from PyQt6.QtCore import QThread, pyqtSignal
import os
import traceback
from ultralytics import YOLO

class TrainThread(QThread):
    """Training thread for YOLO detection models"""
    finished = pyqtSignal()
    epoch_progress = pyqtSignal(int)
    
    def __init__(self, train_config):
        super().__init__()
        self.train_config = train_config
        self.success = False
        self.error = ""
        self.model_path = ""

    def run(self):
        try:
            model = YOLO("yolov8n.pt")

            def on_epoch_end(trainer):
                self.epoch_progress.emit(trainer.epoch + 1)

            model.add_callback("on_train_epoch_end", on_epoch_end)
            model.train(**self.train_config)

            self.model_path = os.path.join(
                "runs", "detect",
                self.train_config["name"],
                "weights", "best.pt"
            )
            self.success = True

        except Exception as e:
            self.success = False
            self.error = str(e)
            print(f"Training error: {traceback.format_exc()}")
        finally:
            self.finished.emit()


class TrainSegmentationThread(QThread):
    """Training thread for YOLO segmentation models"""
    finished = pyqtSignal()
    epoch_progress = pyqtSignal(int)
    
    def __init__(self, train_config):
        super().__init__()
        self.train_config = train_config
        self.success = False
        self.error = ""
        self.model_path = ""
        
    def run(self):
        try:
            # Load segmentation-specific model
            model = YOLO("yolov8n-seg.pt")

            def on_epoch_end(trainer):
                self.epoch_progress.emit(trainer.epoch + 1)

            model.add_callback("on_train_epoch_end", on_epoch_end)
            
            # Set segmentation task
            self.train_config["task"] = "segment"
            model.train(**self.train_config)

            self.model_path = os.path.join(
                "runs", "segment",  # Note: 'segment' not 'detect'
                self.train_config["name"],
                "weights", "best.pt"
            )
            self.success = True

        except Exception as e:
            self.success = False
            self.error = str(e)
            print(f"Segmentation training error: {traceback.format_exc()}")
        finally:
            self.finished.emit()