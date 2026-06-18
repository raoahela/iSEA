from PyQt6.QtCore import QThread, pyqtSignal, QMutex
from ultralytics import SAM
import numpy as np
import torch
from pathlib import Path
from .utils import resource_path
import os

class SAM2Thread(QThread):
    mask_finished = pyqtSignal(dict, object, int)
    mask_preview = pyqtSignal(dict, object, int)
    error = pyqtSignal(str)

    def __init__(self, model_name="sam2.1_b.pt", parent=None):
        super().__init__(parent)
        self.model = None
        self.current_frame = None
        self.current_frame_num = 0
        self.prompts = []
        self.prompt_type = "points"
        self.running = True
        self.mutex = QMutex()
        self.model_name = model_name
        self.cuda_available = torch.cuda.is_available()
        self.preview_mode = False
        self.load_model()

    def load_model(self):
        try:
            model_path = resource_path(os.path.join("models", self.model_name))
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")

            device = 'cuda' if self.cuda_available else 'cpu'
            self.model = SAM(model_path)
            self.model.to(device)

        except Exception as e:
            self.error.emit(str(e))
            self.model = None

    def set_frame_and_prompts(self, frame, frame_num, prompts, prompt_type="points", preview=False):
        self.mutex.lock()
        self.current_frame = frame
        self.current_frame_num = frame_num
        self.prompts = prompts
        self.prompt_type = prompt_type
        self.preview_mode = preview
        self.mutex.unlock()

    def clear_prompts(self):
        self.mutex.lock()
        self.prompts = []
        self.current_frame = None
        self.mutex.unlock()

    def stop(self):
        self.mutex.lock()
        self.running = False
        self.mutex.unlock()
        self.wait()

    def run(self):

        while True:
            self.mutex.lock()
            has_work = self.current_frame is not None
            frame = self.current_frame
            frame_num = self.current_frame_num
            prompts = list(self.prompts) if self.prompts else []
            prompt_type = self.prompt_type
            is_preview = self.preview_mode
            running = self.running
            self.mutex.unlock()

            if not running:
                break

            if not has_work:
                self.msleep(5)  # MUDANÇA: era 30ms, agora 5ms
                continue

            # Limpar para não reprocessar o mesmo frame
            self.mutex.lock()
            self.current_frame = None
            self.mutex.unlock()

            if self.model is None:
                continue

            try:
                device = 'cuda' if self.cuda_available else 'cpu'
                kwargs = {"device": device, "verbose": False}

                if prompt_type == "points":
                    if prompts:
                        # Normalizar formato dos pontos para Ultralytics SAM
                        if len(prompts) == 1 and isinstance(prompts[0], (list, tuple)) and len(prompts[0]) == 2:
                            point = [float(prompts[0][0]), float(prompts[0][1])]
                            kwargs["points"] = [point]   # Lista de pontos
                            kwargs["labels"] = [1]         # 1 = ponto positivo (foreground)
                        elif len(prompts) == 2 and isinstance(prompts[0], (int, float)):
                            # Formato plano [x, y] - converter para lista de pontos
                            point = [float(prompts[0]), float(prompts[1])]
                            kwargs["points"] = [point]
                            kwargs["labels"] = [1]
                        else:
                            # Múltiplos pontos: [[x1,y1], [x2,y2]]
                            kwargs["points"] = prompts
                            kwargs["labels"] = [1] * len(prompts)

                elif prompt_type == "bboxes":
                    kwargs["bboxes"] = prompts
                results = self.model.predict(source=frame, **kwargs)

                if results and len(results) > 0:
                    result = results[0]
                    masks = result.masks

                    if masks is not None and hasattr(masks, 'data') and masks.data is not None:
                        num_masks = masks.data.shape[0] if hasattr(masks.data, 'shape') else len(masks.data)

                        if num_masks > 0:
                            mask_tensor = masks.data[0]
                            mask_np = mask_tensor.cpu().numpy()

                            # Aplicar threshold se for probabilidades (float)
                            if mask_np.dtype in [np.float32, np.float64]:
                                mask_np = (mask_np > 0.5).astype(np.uint8)

                            orig_shape = getattr(masks, 'orig_shape', frame.shape[:2])

                            mask_data = {
                                "segmentation": mask_np,
                                "all_masks": [m.cpu().numpy() for m in masks.data],
                                "scores": [0.95] * num_masks,
                                "orig_shape": orig_shape,
                                "prompts": prompts,
                                "prompt_type": prompt_type
                            }

                            if is_preview:
                                self.mask_preview.emit(mask_data, frame, frame_num)
                            else:
                                self.mask_finished.emit(mask_data, frame, frame_num)
                        else:
                            if not is_preview:
                                self.error.emit("No masks generated")
                    else:
                        if not is_preview:
                            self.error.emit("No masks generated")
                else:
                    pass

            except Exception as e:
                error_msg = f"SAM 2 error: {str(e)}"
                import traceback
                traceback.print_exc()
                if not is_preview:
                    self.error.emit(error_msg)