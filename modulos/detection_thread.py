from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition, QElapsedTimer
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
        self.is_first_frame = True  
        self.cuda_available = torch.cuda.is_available()

    def set_frame(self, frame: np.ndarray, frame_num: int):
        self.mutex.lock()
        self.current_frame = frame
        self.current_frame_num = frame_num
        self.condition.wakeOne()
        self.mutex.unlock()

    def set_model(self, model: YOLO | None):
        self.mutex.lock()
        self.model = model
        self.mutex.unlock()

    def stop(self):
        self.mutex.lock()
        self.running = False
        self.condition.wakeOne()
        self.mutex.unlock()
        self.wait()  # waits for thread to end 

    def run(self):
        while True:
            self.mutex.lock()
            while self.current_frame is None and self.running:
                self.condition.wait(self.mutex)
            
            if not self.running:
                self.mutex.unlock()
                break
                
            frame = self.current_frame
            frame_num = self.current_frame_num
            self.current_frame = None
            
            model = self.model
            self.mutex.unlock()

            if model is None:
                continue

            try:
                with torch.no_grad():
                    if self.is_first_frame:
                        # For the first frame: only detection, without tracking 
                        results = model.predict(
                            frame, 
                            verbose=False, 
                            device='cuda' if self.cuda_available else 'cpu',
                            conf=0.5
                        )
                        self.is_first_frame = False
                    else:
                        # For next frames: tracking
                        results = model.track(
                            frame, 
                            persist=True, 
                            verbose=False, 
                            device='cuda' if self.cuda_available else 'cpu', 
                            tracker="botsort.yaml"
                        )
                    
                    if results and len(results) > 0:
                        self.detection_finished.emit(results[0], frame, frame_num)
                        
            except Exception as e:
                print("DetectionThread erro:", e)