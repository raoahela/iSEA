from PyQt6.QtCore import QThread, pyqtSignal
import os
import traceback
from ultralytics import YOLO

class TrainThread(QThread):
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
            print(f"Erro no treinamento: {traceback.format_exc()}")
        finally:
            self.finished.emit() 