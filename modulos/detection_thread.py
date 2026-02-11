from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition, QMutexLocker
from ultralytics import YOLO
import numpy as np
import torch

class DetectionThread(QThread):
    detection_finished = pyqtSignal(object, object, int)
    
    def __init__(self, model: YOLO, parent=None):
        super().__init__(parent)
        self.model = model
        self.current_frame = None
        self.current_frame_num = 0
        self.running = True
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.cuda_available = torch.cuda.is_available()
        
    def set_frame(self, frame: np.ndarray, frame_num: int):
        with QMutexLocker(self.mutex):
            self.current_frame = frame
            self.current_frame_num = frame_num
            
            self.condition.wakeOne()

    def set_model(self, model: YOLO | None):
        with QMutexLocker(self.mutex):
            self.model = model

    def stop(self):
        with QMutexLocker(self.mutex):
            self.running = False
            self.condition.wakeOne()
        self.wait()

    def run(self):
        while True:
            with QMutexLocker(self.mutex):
                while self.current_frame is None and self.running:
                    self.condition.wait(self.mutex)
                
                if not self.running:
                    break
                    
                frame = self.current_frame
                frame_num = self.current_frame_num
                self.current_frame = None
            
            model = self.model
            if model is None:
                continue

            try:
                with torch.no_grad():
                    results = model.track(
                        frame, 
                        persist=True, 
                        verbose=False, 
                        device='cuda' if self.cuda_available else 'cpu', 
                        tracker="botsort.yaml",
                        conf=0.55,
                        iou=0.65,
                    )
                    
                    if results and len(results) > 0:
                        self.detection_finished.emit(results[0], frame, frame_num)
                        
            except Exception as e:
                print("DetectionThread erro:", e)