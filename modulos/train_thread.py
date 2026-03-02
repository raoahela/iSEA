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
            model = YOLO("yolo26n.pt")

            def on_epoch_end(trainer):
                self.epoch_progress.emit(trainer.epoch + 1)

            model.add_callback("on_train_epoch_end", on_epoch_end)
            
            aug_params = {
                "hsv_h": 0.05,        # matiz
                "hsv_s": 0.9,          # saturação
                "hsv_v": 0.6,          # valor
                "degrees": 15.0,        # rotação
                "translate": 0.2,      # translação
                "scale": 0.7,          # escala
                "shear": 5.0,          # cisalhamento
                "perspective": 0.001,    # perspectiva
                "flipud": 0.3,         # flip vertical
                "fliplr": 0.5,         # flip horizontal
                "mosaic": 1.0,         # mosaic
                "mixup": 0.1,          # mixup
                "copy_paste": 0.1,     # copy-paste
                "auto_augment": "randaugment",
                "erasing": 0.4,        # random erasing
            }
            
            # Mescla com train_config (train_config tem prioridade)
            final_config = {**aug_params, **self.train_config}
            
            model.train(**final_config)

            self.model_path = os.path.join(
                "runs", "detect",
                final_config["name"],
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
            model = YOLO("yolo26n-seg.pt")

            def on_epoch_end(trainer):
                self.epoch_progress.emit(trainer.epoch + 1)

            model.add_callback("on_train_epoch_end", on_epoch_end)
            
            # Mesmos defaults de augmentation
            aug_params = {
                "hsv_h": 0.05,        # matiz
                "hsv_s": 0.9,          # saturação
                "hsv_v": 0.6,          # valor
                "degrees": 15.0,        # rotação
                "translate": 0.2,      # translação
                "scale": 0.7,          # escala
                "shear": 5.0,          # cisalhamento
                "perspective": 0.001,  # perspectiva
                "flipud": 0.3,         # flip vertical
                "fliplr": 0.5,         # flip horizontal
                "mosaic": 1.0,         # mosaic
                "mixup": 0.1,          # mixup
                "copy_paste": 0.1,     # copy-paste
                "auto_augment": "randaugment",
                "erasing": 0.4,        # random erasing
            }
            
            final_config = {**aug_params, **self.train_config}
            final_config["task"] = "segment"
            
            model.train(**final_config)

            self.model_path = os.path.join(
                "runs", "segment",
                final_config["name"],
                "weights", "best.pt"
            )
            self.success = True

        except Exception as e:
            self.success = False
            self.error = str(e)
            print(f"Segmentation training error: {traceback.format_exc()}")
        finally:
            self.finished.emit()