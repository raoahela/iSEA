from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
from ultralytics import SAM
import numpy as np
import torch

class SAM2Thread(QThread):
    mask_finished = pyqtSignal(object, object, int)  # mask_data, original_frame, frame_num
    error = pyqtSignal(str)
    
    def __init__(self, model_name="sam2.1_b.pt", parent=None):
        super().__init__(parent)
        self.model = None
        self.current_frame = None
        self.current_frame_num = 0
        self.prompts = []  # [(x, y, type)] where type: 1=foreground, 0=background
        self.running = True
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.model_name = model_name
        self.cuda_available = torch.cuda.is_available()
        self.load_model()

    def load_model(self):
        """Load SAM 2 model"""
        try:
            device = 'cuda' if self.cuda_available else 'cpu'
            self.model = SAM(self.model_name)
            self.model.to(device)
            print(f"SAM 2 model loaded on {device}")
        except Exception as e:
            print(f"Failed to load SAM 2: {e}")
            self.error.emit(str(e))

    def set_frame_and_prompts(self, frame: np.ndarray, frame_num: int, prompts: list):
        """Set frame and prompts for segmentation"""
        self.mutex.lock()
        self.current_frame = frame
        self.current_frame_num = frame_num
        self.prompts = prompts
        self.condition.wakeOne()
        self.mutex.unlock()

    def clear_prompts(self):
        """Clear all prompts"""
        self.prompts = []

    def stop(self):
        self.mutex.lock()
        self.running = False
        self.condition.wakeOne()
        self.mutex.unlock()
        self.wait()

    def run(self):
        while True:
            self.mutex.lock()
            while self.current_frame is None and self.running:
                self.condition.wait(self.mutex)
            
            if not self.running:
                self.mutex.unlock()
                break
                
            frame = self.current_frame.copy()
            frame_num = self.current_frame_num
            prompts = self.prompts.copy()
            self.current_frame = None
            self.mutex.unlock()

            if self.model is None or not prompts:
                continue

            try:
                # Prepare points and labels
                points = []
                labels = []
                for x, y, label in prompts:
                    points.append([x, y])
                    labels.append(label)
                
                points = np.array(points) if points else None
                labels = np.array(labels) if labels else None

                # Run SAM 2 inference
                device = 'cuda' if self.cuda_available else 'cpu'
                results = self.model.predict(
                    frame,
                    points=points,
                    labels=labels,
                    device=device,
                    verbose=False
                )

                if results and len(results) > 0:
                    # Extract mask data
                    masks = results[0].masks
                    if masks is not None and len(masks.data) > 0:
                        # SAM 2 doesn't have 'conf' attribute - use fixed confidence
                        mask_data = {
                            "segmentation": masks.data[0].cpu().numpy(),
                            "all_masks": [mask.cpu().numpy() for mask in masks.data],
                            "scores": [0.95] * len(masks.data),  # Valor fixo de confiança
                            "orig_shape": masks.orig_shape if hasattr(masks, 'orig_shape') else frame.shape[:2]
                        }
                        self.mask_finished.emit(mask_data, frame, frame_num)
                    else:
                        self.error.emit("No masks generated")

            except Exception as e:
                error_msg = f"SAM 2 error: {str(e)}"
                print(error_msg)
                self.error.emit(error_msg)