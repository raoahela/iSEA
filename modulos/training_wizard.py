import os, cv2, shutil
from pathlib import Path
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QInputDialog,
                             QPushButton, QListWidget, QFileDialog, QGroupBox, QApplication,
                             QDialogButtonBox, QProgressDialog, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
from .translations import TEXTS


class TrainingWizard(QDialog):
    frames_ready = pyqtSignal(list)   # list (image_path, frame_num, video_name)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.language = parent.language
        self.texts = TEXTS[self.language]
        self.setWindowTitle(self.texts["create_dataset"])
        self.resize(600, 400)
        self.frame_list = []            
        self.parent = parent
        self.build_ui()

    def build_ui(self):
        layout = QVBoxLayout(self)

        orig_group = QGroupBox(self.texts["image_source"])
        h = QHBoxLayout(orig_group)
        self.video_btn = QPushButton(self.texts["load_video"])
        self.photos_btn = QPushButton(self.texts["load_photos"])
        self.video_btn.clicked.connect(self.load_video)
        self.photos_btn.clicked.connect(self.load_photos)
        h.addWidget(self.video_btn)
        h.addWidget(self.photos_btn)
        layout.addWidget(orig_group)

        self.list_w = QListWidget()
        layout.addWidget(QLabel(self.texts["frames_found"]))
        layout.addWidget(self.list_w)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, self.texts["select_video"], "", 
            "Vídeos (*.mp4 *.avi *.mov *.mkv *.m4v)")
        if not path:
            return
        
        video_name = Path(path).stem
        safe_video_name = "".join(c for c in video_name if c.isalnum() or c in ('_', '-'))
        
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        step, ok = QInputDialog.getInt(
            self, self.texts["extraction_rate"], self.texts["extract_every_n"], 
            100, 1, 1000)
        if not ok:
            return

        dir_name = safe_video_name + "_frames"
        out_dir = Path(QFileDialog.getExistingDirectory(
            self, self.texts["select_output_dir"])) / dir_name
        out_dir.mkdir(parents=True, exist_ok=True)

        progress = QProgressDialog(self.texts["extracting_frames"], 
                                  self.texts["cancel"], 0, total // step, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        cap = cv2.VideoCapture(path)
        frame_counter = 0
        saved_counter = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_counter % step == 0:
                # Temporary frame: [video_name]_frame_000000.jpg
                temp_name = f"{safe_video_name}_frame_{saved_counter:06d}.jpg"
                fname = out_dir / temp_name
                cv2.imwrite(str(fname), frame)
                
                # Stores: (path_temp, original_frame_number, video_name)
                self.frame_list.append((str(fname), frame_counter, safe_video_name))
                
                progress.setValue(saved_counter)
                QApplication.processEvents()
                if progress.wasCanceled():
                    break
                saved_counter += 1
            frame_counter += 1
            
        cap.release()
        progress.close()
        self.fill_list()

    def load_photos(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, self.texts["select_photos"], "", "Imagens (*.jpg *.png *.jpeg)")
        if not files:
            return
        
        # Directory name as identifier
        dir_name = Path(files[0]).parent.stem
        safe_name = "".join(c for c in dir_name if c.isalnum() or c in ('_', '-'))
        
        self.frame_list = [(f, None, safe_name) for f in files]
        self.fill_list()

    def fill_list(self):
        self.list_w.clear()
        for path, num, video_name in self.frame_list:
            self.list_w.addItem(f"{Path(path).name} ({video_name})")

    def accept(self):
        if not self.frame_list:
            QMessageBox.warning(self, self.texts["warning"], 
                              self.texts["no_images_loaded"])
            return
        self.frames_ready.emit(self.frame_list)
        super().accept()

    def update_language(self, lang):
        self.language = lang
        self.texts = TEXTS[lang]
        
        # updates window title 
        self.setWindowTitle(self.texts["create_dataset"])
        
        # updates origin group 
        orig_group = self.findChild(QGroupBox)
        if orig_group:
            orig_group.setTitle(self.texts["image_source"])
        
        # updates buttons 
        self.video_btn.setText(self.texts["load_video"])
        self.photos_btn.setText(self.texts["load_photos"])
        
        # updates list label
        frames_label = self.findChild(QLabel)
        if frames_label:
            frames_label.setText(self.texts["frames_found"])
        
        # updates cancel button
        button_box = self.findChild(QDialogButtonBox)
        if button_box:
            cancel_btn = button_box.button(QDialogButtonBox.StandardButton.Cancel)
            if cancel_btn:
                cancel_btn.setText(self.texts.get("cancel", "Cancelar"))