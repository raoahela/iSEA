"""
iSEA - SAM 2 Thread
"""

from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker, QWaitCondition
from ultralytics import SAM
import numpy as np
import torch

from .utils import resource_path


class SAM2Thread(QThread):
    """Thread para processamento SAM 2 em background."""

    mask_finished = pyqtSignal(dict, object, int)
    mask_preview = pyqtSignal(dict, object, int)
    error = pyqtSignal(str)

    def __init__(self, model_name="sam2.1_s.pt", parent=None):
        super().__init__(parent)
        self.model = None
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cuda_available = torch.cuda.is_available()

        # Estado do frame atual
        self.current_frame = None
        self.current_frame_num = 0
        self.prompts = []
        self.prompt_type = "points"
        self.preview_mode = False

        # Controle de execução
        self.running = True
        self.model_loaded = False
        self.load_error = None

        # Sincronização
        self.mutex = QMutex()
        self.condition = QWaitCondition()

    # =====================================================================
    # CARREGAMENTO DO MODELO (executa na thread, não na GUI)
    # =====================================================================

    def load_model(self):
        """Carrega o modelo SAM do disco. Executado dentro de run()."""
        try:
            model_path = Path(resource_path("modelos")) / self.model_name

            if not model_path.exists():
                self.load_error = f"Model not found: {model_path}"
                return

            self.model = SAM(str(model_path))
            self.model.to(self.device)
            self.model_loaded = True

        except Exception as e:
            self.load_error = f"Failed to load SAM model: {e}"
            self.model = None

    # =====================================================================
    # API PÚBLICA
    # =====================================================================

    def set_frame_and_prompts(self, frame, frame_num, prompts,
                              prompt_type="points", preview=False):
        """Define frame e prompts para processamento na próxima iteração."""
        with QMutexLocker(self.mutex):
            self.current_frame = frame
            self.current_frame_num = frame_num
            self.prompts = prompts
            self.prompt_type = prompt_type
            self.preview_mode = preview
            self.condition.wakeOne()

    def clear_prompts(self):
        """Limpa prompts e frame pendentes."""
        with QMutexLocker(self.mutex):
            self.prompts = []
            self.current_frame = None

    def stop(self):
        """Sinaliza parada da thread e aguarda término."""
        with QMutexLocker(self.mutex):
            self.running = False
            self.condition.wakeOne()
        self.wait()

    # =====================================================================
    # LOOP PRINCIPAL
    # =====================================================================

    def run(self):
        """Loop principal da thread."""
        self.load_model()

        if self.load_error:
            self.error.emit(self.load_error)
            return

        while True:
            with QMutexLocker(self.mutex):
                while self.current_frame is None and self.running:
                    self.condition.wait(self.mutex)

                if not self.running:
                    break

                frame = self.current_frame
                frame_num = self.current_frame_num
                prompts = list(self.prompts) if self.prompts else []
                prompt_type = self.prompt_type
                is_preview = self.preview_mode
                self.current_frame = None  # evita reprocessar

            if self.model is None:
                self.error.emit("SAM model not available")
                continue

            self._process_frame(frame, frame_num, prompts, prompt_type, is_preview)

    # =====================================================================
    # PROCESSAMENTO DE FRAME
    # =====================================================================

    def _process_frame(self, frame, frame_num, prompts, prompt_type, is_preview):
        """Executa inferência SAM em um frame."""
        try:
            kwargs = self._build_kwargs(prompts, prompt_type)
            if kwargs is None:
                if not is_preview:
                    self.error.emit("Invalid or empty prompts")
                return

            results = self.model.predict(
                source=frame, device=self.device, verbose=False, **kwargs
            )

            if not (results and len(results) > 0):
                if not is_preview:
                    self.error.emit("No results from SAM model")
                return

            self._handle_result(results[0], frame, frame_num,
                                prompts, prompt_type, is_preview)

        except Exception as e:
            msg = f"SAM 2 error: {str(e)}"
            if not is_preview:
                self.error.emit(msg)

    def _build_kwargs(self, prompts, prompt_type):
        """Constrói kwargs para model.predict baseado no tipo de prompt."""
        if prompt_type == "points":
            if not prompts:
                return None
            try:
                normalized = self._normalize_points(prompts)
            except ValueError as e:
                return None
            if not normalized:
                return None
            return {
                "points": normalized,
                "labels": [1] * len(normalized),
            }

        elif prompt_type == "bboxes":
            if not prompts:
                return None
            return {"bboxes": prompts}

        return None

    def _normalize_points(self, prompts):
        """Normaliza qualquer formato de pontos para [[x, y], ...]."""
        arr = np.asarray(prompts, dtype=float)

        if arr.ndim == 1 and arr.size == 2:
            return [[float(arr[0]), float(arr[1])]]

        if arr.ndim == 2 and arr.shape[1] == 2:
            return [[float(x), float(y)] for x, y in arr]

        raise ValueError(f"Invalid point format: shape={arr.shape}")

    # =====================================================================
    # TRATAMENTO DE RESULTADO
    # =====================================================================

    def _handle_result(self, result, frame, frame_num, prompts, prompt_type, is_preview):
        """Processa resultado SAM e emite sinal apropriado."""
        masks = result.masks

        if masks is None or not hasattr(masks, 'data') or masks.data is None:
            if not is_preview:
                self.error.emit("No masks generated")
            return

        data = masks.data
        num_masks = data.shape[0] if hasattr(data, 'shape') else len(data)
        if num_masks == 0:
            if not is_preview:
                self.error.emit("No masks generated")
            return

        mask_data = self._build_mask_data(masks, frame, prompts, prompt_type)

        if is_preview:
            self.mask_preview.emit(mask_data, frame, frame_num)
        else:
            self.mask_finished.emit(mask_data, frame, frame_num)

    def _build_mask_data(self, masks, frame, prompts, prompt_type):
        """Constrói dicionário com dados da máscara."""
        data = masks.data

        first = data[0]
        if torch.is_tensor(first):
            first = first.detach().cpu().numpy()
        else:
            first = np.asarray(first)

        if first.ndim == 3 and first.shape[0] == 1:
            first = first[0]


        if first.dtype in (np.float32, np.float64):
            for thr in (0.3, 0.4, 0.5):
                n = int((first > thr).sum())
            first = (first > 0.5).astype(np.uint8)

        n_mask = int(first.sum())

        all_masks = []
        for m in data:
            if torch.is_tensor(m):
                m_np = m.detach().cpu().numpy()
            else:
                m_np = np.asarray(m)
            if m_np.ndim == 3 and m_np.shape[0] == 1:
                m_np = m_np[0]
            if m_np.dtype in (np.float32, np.float64):
                m_np = (m_np > 0.5).astype(np.uint8)
            all_masks.append(m_np)

        return {
            "segmentation": first,
            "all_masks": all_masks,
            "scores": [0.95] * len(all_masks),
            "orig_shape": getattr(masks, 'orig_shape', frame.shape[:2]),
            "prompts": prompts,
            "prompt_type": prompt_type,
        }