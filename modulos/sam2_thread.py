"""
iSEA - SAM 2 Thread 
"""

from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal, QMutex
from ultralytics import SAM
import numpy as np
import torch

from .utils import resource_path


class SAM2Thread(QThread):
    """Thread para processamento SAM 2 em background."""

    mask_finished = pyqtSignal(dict, object, int)
    mask_preview = pyqtSignal(dict, object, int)
    error = pyqtSignal(str)

    def __init__(self, model_name="sam2.1_b.pt", parent=None):
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
        self.mutex = QMutex()

        self.load_model()

    def load_model(self):
        """Carrega o modelo SAM do disco."""
        try:
            model_path = Path(resource_path("models")) / self.model_name
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")

            self.model = SAM(str(model_path))
            self.model.to(self.device)

        except Exception as e:
            self.error.emit(str(e))
            self.model = None

    def set_frame_and_prompts(self, frame, frame_num, prompts, prompt_type="points", preview=False):
        """Define frame e prompts para processamento na próxima iteração."""
        self.mutex.lock()
        self.current_frame = frame
        self.current_frame_num = frame_num
        self.prompts = prompts
        self.prompt_type = prompt_type
        self.preview_mode = preview
        self.mutex.unlock()

    def clear_prompts(self):
        """Limpa prompts e frame pendentes."""
        self.mutex.lock()
        self.prompts = []
        self.current_frame = None
        self.mutex.unlock()

    def stop(self):
        """Sinaliza parada da thread e aguarda término."""
        self.mutex.lock()
        self.running = False
        self.mutex.unlock()
        self.wait()

    # =====================================================================
    # LOOP PRINCIPAL
    # =====================================================================

    def run(self):
        """Loop principal da thread."""
        while True:
            frame, frame_num, prompts, prompt_type, is_preview, should_run = self._get_work()

            if not should_run:
                break

            if frame is None:
                self.msleep(5)
                continue

            if self.model is None:
                continue

            self._process_frame(frame, frame_num, prompts, prompt_type, is_preview)

    def _get_work(self):
        """Obtém estado atual de forma thread-safe."""
        self.mutex.lock()
        frame = self.current_frame
        frame_num = self.current_frame_num
        prompts = list(self.prompts) if self.prompts else []
        prompt_type = self.prompt_type
        is_preview = self.preview_mode
        should_run = self.running

        # Limpar para não reprocessar
        self.current_frame = None
        self.mutex.unlock()

        return frame, frame_num, prompts, prompt_type, is_preview, should_run

    # =====================================================================
    # PROCESSAMENTO DE FRAME
    # =====================================================================

    def _process_frame(self, frame, frame_num, prompts, prompt_type, is_preview):
        """Executa inferência SAM em um frame."""
        try:
            kwargs = self._build_kwargs(prompts, prompt_type)
            if kwargs is None:
                return

            results = self.model.predict(source=frame, device=self.device, verbose=False, **kwargs)

            if not (results and len(results) > 0):
                if not is_preview:
                    self.error.emit("No results from SAM model")
                return

            self._handle_result(results[0], frame, frame_num, prompts, prompt_type, is_preview)

        except Exception as e:
            if not is_preview:
                self.error.emit(f"SAM 2 error: {str(e)}")

    def _build_kwargs(self, prompts, prompt_type):
        """Constrói kwargs para model.predict baseado no tipo de prompt."""
        if prompt_type == "points":
            if not prompts:
                return None
            normalized = self._normalize_points(prompts)
            return {
                "points": normalized,
                "labels": [1] * len(normalized)
            }

        elif prompt_type == "bboxes":
            return {"bboxes": prompts}

        return None

    def _normalize_points(self, prompts):
        """Normaliza qualquer formato de pontos para [[x, y], ...]."""
        # Formato 1: [[x, y]] — lista de pontos
        if (len(prompts) == 1 and isinstance(prompts[0], (list, tuple))
                and len(prompts[0]) == 2):
            return [[float(prompts[0][0]), float(prompts[0][1])]]

        # Formato 2: [x, y] — lista plana
        if (len(prompts) == 2 and isinstance(prompts[0], (int, float))
                and isinstance(prompts[1], (int, float))):
            return [[float(prompts[0]), float(prompts[1])]]

        # Formato 3: [[x1,y1], [x2,y2]] — múltiplos pontos já normalizados
        return prompts

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

        num_masks = masks.data.shape[0] if hasattr(masks.data, 'shape') else len(masks.data)
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
        mask_tensor = masks.data[0]
        mask_np = mask_tensor.cpu().numpy()

        # Aplicar threshold se for probabilidades (float)
        if mask_np.dtype in (np.float32, np.float64):
            mask_np = (mask_np > 0.5).astype(np.uint8)

        return {
            "segmentation": mask_np,
            "all_masks": [m.cpu().numpy() for m in masks.data],
            "scores": [0.95] * len(masks.data),
            "orig_shape": getattr(masks, 'orig_shape', frame.shape[:2]),
            "prompts": prompts,
            "prompt_type": prompt_type
        }