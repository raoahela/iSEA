import json
import yaml
import os
import sys
import cv2
import hashlib
import pandas as pd
import traceback
from datetime import datetime, timedelta
import csv
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from pathlib import Path
import shutil
from PyQt6 import QtCore
from PyQt6.QtGui import (QPixmap, QImage, QIcon, QPainter, QPen, QAction, 
                         QKeySequence, QColor, QPalette, QDesktopServices)
from PyQt6.QtCore import Qt, QTimer, QRect, QSize, QUrl
from PyQt6.QtWidgets import (QApplication, QLabel, QPushButton, QVBoxLayout,  
                            QHBoxLayout, QWidget, QFileDialog, QMainWindow, QToolBar, QStyle, QMessageBox, 
                            QInputDialog, QSlider, QDockWidget, QDialog, QDialogButtonBox, 
                            QSizePolicy, QFrame, QSpinBox, QFormLayout,
                            QComboBox, QDoubleSpinBox, QProgressDialog)
from .video_label import VideoLabel
from .detections_dock import DetectionsDockWidget
from .train_thread import TrainThread
from .translations import TEXTS
from .taxon_grid import TaxonGrid
from .detection_thread import DetectionThread
from .training_wizard import TrainingWizard

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path).replace("/", os.sep)

class VideoAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.language = "pt"  
        self.texts = TEXTS[self.language] 
        
        self.setWindowTitle("iSEA")
        self.resize(1200, 700)

        self.model = None
        self.custom_classes = []
        self.model_path = None
        self.cap = None
        self.video_path = None
        self.paused = True
        self.continuous_detection = False
        self.annotations = []
        self.current_frame_num = 0
        self.total_frames = 0
        self.all_detections = []
        self.tracking_enabled = True
        self.track_colors = {}
        self.current_tracks = {}
        self.live_mode = False
        self.velocity = False
        self.camera_index = 0
        self.recording = False
        self.video_writer = None
        self.recorded_frames = []
        self.record_start_frame = 0
        self.recorded_detections = [] 
        self.best_confidence = {}  
        self.init_ui()
        self.create_menu()
        self.apply_light_style()  
        self.detection_every_n_frames = 2
        self.detection_thread = DetectionThread(None)
        self.detection_thread.detection_finished.connect(self.on_detection_finished)
        self.detection_thread.start()
        self.last_frame_hash = None
        self.last_frame_small = None
        self.drawing_color = QColor(Qt.GlobalColor.green)
        self.dataset_mode = False              
        self.dataset_frames = []               
        self.dataset_index = 0    
        self.training_wizard = None            
        
        self.taxon_grid = None
        self.taxon_grid = TaxonGrid(self)
        self.taxon_grid_dock = QDockWidget(self.texts["taxons"])
        self.taxon_grid.title_changed.connect(self.taxon_grid_dock.setWindowTitle)
        self.taxon_grid_dock.setWidget(self.taxon_grid)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.taxon_grid_dock)
        initial = []
        if self.model:
            initial = list(self.model.names.values())
        initial += self.custom_classes
        self.taxon_grid.populate(initial)
        self.taxon_grid.taxon_changed.connect(self.change_drawing_class)
        self.load_model()

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.video_label = VideoLabel(self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setText(self.texts["load_drag"])
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.status_label = QLabel(self.texts["waiting_action"])
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: gray")

        self.video_name_label = QLabel(self.texts["no_loaded"])
        self.video_name_label.setStyleSheet("color: gray")
        
        #buttons
        self.load_button = QPushButton("")
        self.load_button.setToolTip(self.texts["load_video"])
        self.load_button.setIcon(QIcon(resource_path("icons/load_icon.png")))
        self.load_button.setIconSize(QtCore.QSize(30, 30))

        self.detect_button = QPushButton("")
        self.detect_button.setToolTip(self.texts["detect_frame"])
        self.detect_button.setIcon(QIcon(resource_path("icons/detect_icon.png")))
        self.detect_button.setIconSize(QtCore.QSize(30, 30))

        self.toggle_button = QPushButton("")
        self.toggle_button.setToolTip(self.texts["toggle_detection"])
        self.toggle_button.setIcon(QIcon(resource_path("icons/toggle_icon.png")))
        self.toggle_button.setIconSize(QtCore.QSize(30, 30))

        self.annotate_button = QPushButton("")
        self.annotate_button.setToolTip(self.texts["annotate_manual"])
        self.annotate_button.setIcon(QIcon(resource_path("icons/annotate_icon.png")))
        self.annotate_button.setIconSize(QtCore.QSize(30, 30))

        self.save_button = QPushButton("")
        self.save_button.setToolTip(self.texts["save_annotations"])
        self.save_button.setIcon(QIcon(resource_path("icons/save_icon.png")))
        self.save_button.setIconSize(QtCore.QSize(30, 30))

        self.save_frame_button = QPushButton("")
        self.save_frame_button.setToolTip(self.texts["save_frame"])
        self.save_frame_button.setIcon(QIcon(resource_path("icons/save_frame_icon.png")))
        self.save_frame_button.setIconSize(QtCore.QSize(30, 30))

        self.live_button = QPushButton("🔴")
        self.live_button.setIconSize(QtCore.QSize(30, 30))
        self.live_button.setToolTip(self.texts["live"])

        self.merge_button = QPushButton("")
        self.merge_button.setToolTip(self.texts["merge_annotations"])
        self.merge_button.setIcon(QIcon(resource_path("icons/merge_geo_icon.png")))
        self.merge_button.setIconSize(QtCore.QSize(30, 30))

        #connections
        self.load_button.clicked.connect(self.load_video)
        self.detect_button.clicked.connect(self.detect_objects)
        self.toggle_button.clicked.connect(self.toggle_detection)
        self.annotate_button.clicked.connect(self.enable_manual_annotation)
        self.save_button.clicked.connect(self.save_annotations)
        self.save_frame_button.clicked.connect(self.save_current_frame_with_annotations)
        self.live_button.clicked.connect(self.toggle_live_mode)
        self.merge_button.clicked.connect(self.merge_annotations)

        #toolbar
        self.tool_bar = QToolBar("Playback")
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, self.tool_bar)

        self.play_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        self.pause_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause)

        self._previous_action = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipBackward),
            self.texts["previous"], self)
        self._play_action = QAction(self.pause_icon, self.texts["pause"], self)
        self._next_action = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipForward),
            self.texts["next"], self)

        self._play_action.setShortcut(QKeySequence(Qt.Key.Key_Space))
        self._previous_action.setShortcut(QKeySequence(Qt.Key.Key_Left))
        self._next_action.setShortcut(QKeySequence(Qt.Key.Key_Right))

        self._play_action.triggered.connect(self.toggle_play_pause)
        self._previous_action.triggered.connect(self.previous_frame)
        self._next_action.triggered.connect(self.next_frame)

        self.tool_bar.addAction(self._previous_action)
        self.tool_bar.addAction(self._play_action)
        self.tool_bar.addAction(self._next_action)

        self.velocity2_button = QPushButton("2.0x")
        self.tool_bar.addWidget(self.velocity2_button)
        self.velocity2_button.clicked.connect(self.velocity2)

        #Detections Dock
        self.detections_dock = DetectionsDockWidget(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.detections_dock)

        #Side layout (buttons)
        button_frame = QFrame()
        button_frame.setFixedWidth(50)
        button_layout = QVBoxLayout(button_frame)
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.live_button)
        button_layout.addWidget(self.toggle_button)
        button_layout.addWidget(self.detect_button)
        button_layout.addWidget(self.annotate_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.save_frame_button)
        button_layout.addWidget(self.merge_button)
        button_layout.addStretch()

        # video container
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.addWidget(self.video_name_label)
        video_layout.addWidget(self.video_label, 1)

        #progress slider
        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setRange(0, 100)
        self.progress_slider.sliderMoved.connect(self.seek_video)
        self.progress_slider.sliderPressed.connect(self.pause_video_for_seeking)
        self.progress_slider.sliderReleased.connect(self.resume_video_after_seeking)

        self.current_time_label = QLabel("00:00:00")
        self.total_time_label = QLabel("00:00:00")

        time_container = QWidget()
        time_layout = QHBoxLayout(time_container)
        time_layout.setContentsMargins(0, 0, 0, 0)
        time_layout.addWidget(self.current_time_label)
        time_layout.addStretch()
        time_layout.addWidget(self.total_time_label)

        video_layout.addWidget(time_container)
        video_layout.addWidget(self.progress_slider)
        video_layout.addWidget(self.status_label)

        # main layout
        main_layout = QHBoxLayout(self.central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        main_layout.addWidget(video_container, 1)
        main_layout.addWidget(button_frame)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.setAcceptDrops(True)

    def recolor_icon(self, standard_icon, color=QColor("white")):
        """Recolor icons from QStyle for the dark mode"""
        pixmap = self.style().standardIcon(standard_icon).pixmap(24, 24) #convert the native icon from the system to pixmap.
        painter = QPainter(pixmap) #creates a pencil that draws directly on the pixmap
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn) #ignores transparent backgroung
        painter.fillRect(pixmap.rect(), color) #fills the entire rect of the pixmap with white
        painter.end()
        return QIcon(pixmap) 

    def set_status_message(self, key, *args):
        """Defines status messages"""
        self.status_label.setText(self.texts[key].format(*args))

    def show_error_message(self, title_key, message_key, *args):
        """Exhibits an error message"""
        QMessageBox.critical(self, self.texts[title_key], self.texts[message_key].format(*args))

    def show_warning_message(self, title_key, message_key, *args):
        """Exhibits an warning message"""
        QMessageBox.warning(self, self.texts[title_key], self.texts[message_key].format(*args))

    def show_info_message(self, title_key, message_key, *args):
        """Exhibits an info message"""
        QMessageBox.information(self, self.texts[title_key], self.texts[message_key].format(*args))
        
    def apply_light_style(self):

        self.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                color: #000000;
            }
            QMenuBar {
                background-color: #ffffff;
                color: #000000;
            }
            QMenu {
                background-color: #ffffff;
                color: #000000;
                border: 1px solid #ccc;
            }
            QMenu::item:selected {
                background-color: #d0d0d0;
            }
            QToolBar {
                background-color: #f0f0f0;
                border: none;
            }
            QDockWidget {
                background-color: #ffffff;
                color: #000000;
            }
            QDockWidget::title {
                background-color: #e0e0e0;
                color: black;
            }
            QListWidget {
                background-color: #ffffff;
                color: black;
                border: 1px solid #ccc;
            }
            QListWidget::item:selected {
                background-color: #cce5ff;
                color: black;
            }
            QGroupBox {
                color: black;
                border: 1px solid #aaa;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QSlider::groove:horizontal {
                background: #ddd;
                height: 6px;
            }
            QSlider::handle:horizontal {
                background: #0078d4;
                width: 12px;
                margin: -3px 0;
                border-radius: 6px;
            }
        """)

        button_style = """
            QPushButton {
                padding: 3px 8px;
                border-radius: 4px;
                border: 1px solid #aaa;
                background-color: #f0f0f0;
                color: black;
                min-width: 23px;
                min-height: 23px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:pressed {
                background-color: #ccc;
            }
        """
        for btn in [
            self.load_button, self.live_button, self.detect_button,
            self.toggle_button, self.annotate_button, self.save_button,
            self.save_frame_button, self.merge_button
        ]:
            btn.setStyleSheet(button_style)

        light_palette = QPalette()
        light_palette.setColor(QPalette.ColorRole.Window, QColor(255, 255, 255))
        light_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.black)
        light_palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
        light_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.black)
        light_palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
        light_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.black)
        self.setPalette(light_palette)

    def set_dark_mode(self, enable=True):
        if enable:
            dark_palette = QPalette()
            dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
            dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
            dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
            dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
            dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
            self.setPalette(dark_palette)
            self._previous_action.setIcon(self.recolor_icon(QStyle.StandardPixmap.SP_MediaSkipBackward))
            self._play_action.setIcon(self.recolor_icon(QStyle.StandardPixmap.SP_MediaPlay))
            self._play_action.setIcon(self.recolor_icon(QStyle.StandardPixmap.SP_MediaPause))
            self._next_action.setIcon(self.recolor_icon(QStyle.StandardPixmap.SP_MediaSkipForward))

            self.setStyleSheet("""
                QWidget {
                    background-color: #353535;
                    color: #ffffff;
                }
            """)

            button_style = """
                QPushButton {
                    padding: 3px 8px;
                    border-radius: 4px;
                    border: 1px solid #666;
                    background-color: #444;
                    color: white;
                    min-width: 23px;
                    min-height: 23px;
                }
                QPushButton:hover {
                    background-color: #555;
                }
                QPushButton:pressed {
                    background-color: #2a82da;
                }
            """
            for btn in [
                self.load_button, self.live_button, self.detect_button,
                self.toggle_button, self.annotate_button, self.save_button,
                self.save_frame_button, self.merge_button
            ]:
                btn.setStyleSheet(button_style)

            self.central_widget.setStyleSheet("")
            self.video_label.setStyleSheet("")


        else:
            #restores the light mode
            self.apply_light_style()
            self._previous_action.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipBackward))
            self._play_action.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
            self._next_action.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipForward))

        if hasattr(self, 'detections_dock'):
            self.detections_dock.set_dark_mode(enable)

        if self.taxon_grid:
            self.taxon_grid.set_dark_mode(enable)
        
        self.update()

    def create_menu(self):
        menubar = self.menuBar()
        
        #file menu
        file_menu = menubar.addMenu(self.texts["arquivo"])
        
        open_action = QAction(self.texts["load_video"], self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self.load_video)
        file_menu.addAction(open_action)
        
        load_model_action = QAction(self.texts["load_model"], self)
        load_model_action.setShortcut(QKeySequence("Ctrl+M"))
        load_model_action.triggered.connect(self.load_custom_model)
        file_menu.addAction(load_model_action)
            
        unload_model_action = QAction(self.texts["unload_model"], self)
        unload_model_action.triggered.connect(self.unload_model)
        file_menu.addAction(unload_model_action)

        load_annotations_action = QAction(self.texts["load_annotations"], self)
        load_annotations_action.setShortcut(QKeySequence("Ctrl+L"))
        load_annotations_action.triggered.connect(self.load_annotations_dialog)
        file_menu.addAction(load_annotations_action)
        
        save_action = QAction(self.texts["save_annotations"], self)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self.save_annotations)
        file_menu.addAction(save_action)

        record_action = QAction(self.texts["start_recording"], self)
        record_action.setShortcut(QKeySequence("Ctrl+R"))
        record_action.triggered.connect(self.start_recording)
        file_menu.addAction(record_action)
        
        stop_record_action = QAction(self.texts["stop_recording"], self)
        stop_record_action.setShortcut(QKeySequence("Ctrl+Shift+R"))
        stop_record_action.triggered.connect(self.stop_recording)
        file_menu.addAction(stop_record_action)
        
        live_action = QAction(self.texts["live"], self)
        live_action.setShortcut(QKeySequence("Ctrl+W"))
        live_action.triggered.connect(self.toggle_live_mode)
        file_menu.addAction(live_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction(self.texts["exit"], self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # view menu
        view_menu = menubar.addMenu(self.texts["visualization"])
        toggle_history_action = QAction(self.texts["history_show"], self)
        toggle_history_action.setShortcut(QKeySequence("Ctrl+H"))
        toggle_history_action.triggered.connect(self.toggle_detections_history)
        view_menu.addAction(toggle_history_action)
        view_menu.addSeparator()
        dark_mode_action = QAction(self.texts["dark_mode"], self)
        dark_mode_action.setCheckable(True)
        dark_mode_action.triggered.connect(self.set_dark_mode)
        view_menu.addAction(dark_mode_action)
            
        # annotation menu
        annotation_menu = menubar.addMenu(self.texts["annotation"])
        
        detect_action = QAction(self.texts["detect_frame"], self)
        detect_action.setShortcut(QKeySequence("D"))
        detect_action.triggered.connect(self.detect_objects)
        annotation_menu.addAction(detect_action)
        
        toggle_action = QAction(self.texts["toggle_detection"], self)
        toggle_action.setShortcut(QKeySequence("T"))
        toggle_action.triggered.connect(self.toggle_detection)
        annotation_menu.addAction(toggle_action)
        
        manual_action = QAction(self.texts["annotate_manual"], self)
        manual_action.setShortcut(QKeySequence("M"))
        manual_action.triggered.connect(self.enable_manual_annotation)
        annotation_menu.addAction(manual_action)

        # training menu
        training_menu = menubar.addMenu(self.texts["train"])
        ("Treino")

        create_ds_action = QAction(self.texts["create_dataset"], self)
        create_ds_action.triggered.connect(self.open_training_wizard)
        training_menu.addAction(create_ds_action)

        export_action = QAction(self.texts["export_yolo"], self)
        export_action.triggered.connect(self.export_yolo_annotations_dialog)
        training_menu.addAction(export_action)

        train_action = QAction(self.texts["train_yolo"], self)
        train_action.triggered.connect(self.train_yolo_model)
        training_menu.addAction(train_action)

        # language menu
        lang_menu = menubar.addMenu(self.texts["language"])

        pt_action = QAction(self.texts["portuguese"], self)
        pt_action.triggered.connect(lambda: self.change_language("pt"))
        lang_menu.addAction(pt_action)

        en_action = QAction(self.texts["english"], self)
        en_action.triggered.connect(lambda: self.change_language("en"))
        lang_menu.addAction(en_action)
            
        # help menu
        help_menu = menubar.addMenu(self.texts["help"])
        
        shortcuts_action = QAction(self.texts["shortcuts"], self)
        shortcuts_action.setShortcut(QKeySequence("F1"))
        shortcuts_action.triggered.connect(self.show_shortcuts)
        help_menu.addAction(shortcuts_action)
        
        about_action = QAction(self.texts["about"], self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        manual_action = QAction("Manual", self)
        manual_action.triggered.connect(self.show_manual)
        help_menu.addAction(manual_action)



    def toggle_detections_history(self):
        """toggle visibility of the detections dock"""
        if self.detections_dock.isVisible():
            self.detections_dock.hide()
        else:
            self.detections_dock.show()

    def load_annotations_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, self.texts["select_annotations_file"], 
                                                   "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            self.load_annotations(file_path)

    def change_language(self, lang):
        self.language = lang
        self.texts = TEXTS[lang]

        self.video_name_label.setText(self.texts["no_loaded"])
        self.video_label.setText(self.texts["load_drag"])
        
        if self.model_path:
            self.status_label.setText(self.texts["model_loaded"].format(self.model_path))
        else:
            self.status_label.setText(self.texts["waiting_action"])

        self.load_button.setToolTip(self.texts["load_video"])
        self.detect_button.setToolTip(self.texts["detect_frame"])
        self.toggle_button.setToolTip(self.texts["toggle_detection"])
        self.annotate_button.setToolTip(self.texts["annotate_manual"])
        self.save_button.setToolTip(self.texts["save_annotations"])
        self.save_frame_button.setToolTip(self.texts["save_frame"])
        self.live_button.setToolTip(self.texts["live"])

        if hasattr(self, 'detections_dock'):
            self.detections_dock.update_language(lang)

        if hasattr(self, 'taxon_grid'):
            self.taxon_grid.update_language(lang)

        if hasattr(self, 'training_wizard') and self.training_wizard is not None:
            self.training_wizard.update_language(lang)

        self.menuBar().clear()
        self.create_menu()
        self.detections_dock.show()
    
    def toggle_live_mode(self):
        if self.live_mode:
            self.stop_recording()
            self.live_mode = False

            is_dark = self.palette().color(QPalette.ColorRole.Window).lightness() < 128
            self.live_button.setStyleSheet(f"""
                QPushButton {{
                    padding: 3px 8px;
                    border-radius: 4px;
                    border: 1px solid {'#666' if is_dark else '#aaa'};
                    background-color: {'#333' if is_dark else '#f0f0f0'};
                    color: {'white' if is_dark else 'black'};
                    min-width: 23px;
                    min-height: 23px;
                }}
                QPushButton:hover {{
                    background-color: {'#555' if is_dark else '#e0e0e0'};
                }}
                QPushButton:pressed {{
                    background-color: {'#2a82da' if is_dark else '#ccc'};
                }}
            """)

            self.status_label.setText(self.texts["webcam_mode_off"])
            if self.cap is not None:
                self.cap.release()
            self.timer.stop()

        else:
            self.live_mode = True

            self.live_button.setStyleSheet("""
                QPushButton {
                    padding: 3px 8px;
                    border-radius: 4px;
                    border: 1px solid #ff5c5c;
                    background-color: #ff5c5c;
                    color: white;
                    min-width: 23px;
                    min-height: 23px;
                }
                QPushButton:hover {
                    background-color: #ff7a7a;
                }
                QPushButton:pressed {
                    background-color: #e04343;
                }
            """)  

            # asking what camera to use
            cameras = self.list_available_cameras()
            if len(cameras) > 1:
                camera, ok = QInputDialog.getItem(
                    self, self.texts["select_camera"], 
                    self.texts["choose_camera"], 
                    cameras, 0, False
                )
                if ok:
                    self.camera_index = cameras.index(camera)
            
            self.start_camera()
            self.status_label.setText(self.texts["webcam_mode_on"].format(self.camera_index))
            
            if self.paused:
                self.toggle_play_pause()
            
    def start_recording(self):
        if not self.live_mode or self.cap is None or not self.cap.isOpened():
            return
            
        # vid configurations
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # default value
        
        # creates a file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_filename = f"recording_{timestamp}.avi"
        
        #creates the video writer 
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(
            self.recording_filename, 
            fourcc, 
            fps, 
            (frame_width, frame_height))
            
        if not self.video_writer.isOpened():
            self.status_label.setText(self.texts["error"] + ": " + self.texts["recording_start_error"])
            self.video_writer = None
            return
            
        self.recording = True
        self.record_start_frame = self.current_frame_num
        self.recorded_detections = []
        self.status_label.setText(self.texts["recording_started"].format(self.recording_filename))

    def stop_recording(self):
        if not self.recording:
            return
            
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            
        self.recording = False
        self.status_label.setText(self.texts["recording_stopped"].format(self.recording_filename))
        
        # adds all the detections to the history (detections dock) 
        for detection in self.recorded_detections:
            self.detections_dock.add_detection(detection)
            
        # asks if they want to save the video
        self.prompt_save_recording()

    def prompt_save_recording(self):
        """Asks if the user wants to save the video"""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setText(self.texts["save_recording_question"])
        msg.setWindowTitle(self.texts["save_recording_title"])
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        retval = msg.exec()
        if retval == QMessageBox.StandardButton.Yes:
            self.save_recording()

    def save_recording(self):
        if not hasattr(self, 'recording_filename') or not os.path.exists(self.recording_filename):
            self.status_label.setText(self.texts["no_recording_to_save"])
            return
            
        options = QFileDialog.Option()
        file_path, _ = QFileDialog.getSaveFileName(
            self, self.texts["save_recording_title"], 
            self.recording_filename,
            "Vídeos (*.avi *.mp4);;Todos os arquivos (*)", 
            options=options)
            
        if file_path:
            try:
                import shutil
                shutil.move(self.recording_filename, file_path)
                self.status_label.setText(self.texts["video_saved"].format(file_path))
                
                # saves the corresponding video annotations
                annotation_path = os.path.splitext(file_path)[0] + "_annotations.json"
                with open(annotation_path, 'w') as f:
                    json.dump({
                        "video_file": os.path.basename(file_path),
                        "detections": self.recorded_detections
                    }, f, indent=4)
                    
                self.status_label.setText(self.texts["video_annotations_saved"].format(os.path.dirname(file_path)))
            except Exception as e:
                self.status_label.setText(self.texts["saving_video_error"].format({str(e)}))

    def list_available_cameras(self):
        cameras = []
        index = 0
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            cameras.append(self.texts["camera_name"].format(index))
            cap.release()
            index += 1
        return cameras if cameras else [self.texts["camera_name"].format(0)]

    def start_camera(self):
        if self.cap is not None:
            self.cap.release()
        
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise RuntimeError(self.texts["camera_open_failed"].format(self.camera_index))
                
            # configurations for better performance and quality
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.video_name_label.setText(self.texts["webcam"].format(self.camera_index))
            self.total_frames = 0
            self.current_frame_num = 0
            self.paused = False
            self._play_action.setIcon(self.pause_icon)
            self._play_action.setText(self.texts["pause"])
            
            self.current_time_label.setText("00:00:00")
            self.total_time_label.setText(self.texts["live_text"])
            
            self.annotations = []
            self.video_label.manual_annotations = []
            if hasattr(self, 'detections_dock'):
                self.detections_dock.all_detections = []
                self.detections_dock.apply_filters()
            
            self.timer.start(30)
            
        except Exception as e:
            self.status_label.setText(self.texts["camera_error"].format(str(e)))
            self.live_mode = False
            self.live_button.setStyleSheet("")
            if self.cap is not None:
                self.cap.release()
                self.cap = None

    def refresh_taxon_grid(self):
        """Clears and repopulates the grid with the current models classes"""
        if self.taxon_grid is None:
            return
        self.taxon_grid.clear()                    
        new_classes = []
        if self.model:
            new_classes = list(self.model.names.values())
        new_classes += [c for c in self.custom_classes if c not in new_classes]
        self.taxon_grid.populate(new_classes)

        is_dark = self.palette().color(QPalette.ColorRole.Window).lightness() < 128
        self.taxon_grid.set_dark_mode(is_dark)

    def load_custom_model(self):
        path, _ = QFileDialog.getOpenFileName(self, self.texts["select_model"], "",  "Arquivos de Modelo (*.pt)")
        if path:
            self.load_model(path)

    def load_model(self, model_path=None):
        try:
            if model_path is None:
                # loads deafault model
                self.model = YOLO(resource_path("yolov8n.pt"))
                self.model_path = "yolov8n.pt"
            else:
                if os.path.exists(model_path):
                    self.model = YOLO(resource_path(model_path))
                    self.model_path = os.path.basename(model_path)
                    if self.detection_thread:
                        self.detection_thread.set_model(self.model)
                else:
                    raise FileNotFoundError(self.texts["model_not_found"].format(model_path))
            
            self.status_label.setText(self.texts["model_loaded"].format(self.model_path))
            
            # updates the classes filter on the dock 
            if hasattr(self, 'detections_dock'):
                self.detections_dock.update_class_filter()

            self.refresh_taxon_grid()

        except Exception as e:
            error_msg = self.texts["model_load_error"].format(str(e))
            self.status_label.setText(error_msg)
            QMessageBox.critical(self, self.texts["error"], error_msg)
            self.model = None
            self.model_path = None

    def unload_model(self):
        self.model = None
        self.model_path = None
        if self.detection_thread:
            self.detection_thread.set_model(None)
        self.status_label.setText(self.texts["model_unloaded"])

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, self.texts["select_video"], "", 
                                                   "Vídeos (*.mp4 *.avi *.mov *.mkv *.m4v *.flv *.wmv);;Todos os arquivos (*)")
        if file_path:
            self.start_video(file_path)

    def start_video(self, file_path):
        """Starts a new video"""
        self.video_path = file_path
        self.video_name_label.setText(self.texts["video_name_format"].format(os.path.basename(file_path)))
        
        if self.cap is not None:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            self.status_label.setText(self.texts["video_load_error"])
            return
            
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_label._aspect_ratio = width / height
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_num = 0
        self.paused = True
        is_dark = self.palette().color(QPalette.ColorRole.Window).lightness() < 128   
        if is_dark:
            self._play_action.setIcon(self.recolor_icon(QStyle.StandardPixmap.SP_MediaPlay))
            self._play_action.setText(self.texts["play"])
             
        else:
            self._play_action.setIcon(self.play_icon)
            self._play_action.setText(self.texts["play"])

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            total_seconds = self.total_frames / fps
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            self.total_video_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            self.total_video_time = "00:00:00"
        
        self.total_time_label.setText(self.total_video_time)
        
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_num = 1
            self.display_frame(frame)
            self.update_time_labels()
            self.status_label.setText(self.texts["video_loaded"])
        else:
            self.status_label.setText(self.texts["video_first_frame_error"])

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith((".mp4", ".avi", ".m4v")):
                self.start_video(file_path)
                break
    
    def capture_current_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return None
        
        # saves the current frame position 
        current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            
        # reads the current frame 
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        # restores the original position 
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
            
        # converts to QImage 
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
        return q_img.copy()
    
    def frame_to_small_and_hash(self, frame: np.ndarray) -> tuple[str, np.ndarray]:
        # reduced/hashed version of the frames that goes to "similar_frames"
        small = cv2.resize(frame, (64, 64))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        h = hashlib.blake2b(gray.tobytes(), digest_size=8).hexdigest()
        return h, small
    
    def similar_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold=2):
        # Compares two frames, if is too similar the model will skip
        if frame1 is None or frame2 is None:
            return False
        diff = cv2.absdiff(frame1, frame2)
        mean_diff = np.mean(diff)
        return mean_diff < threshold
    
    def update_frame(self):
        if self.paused or self.cap is None or not self.cap.isOpened():
            return
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                if self.live_mode:
                    # tries to reconnect camera
                    self.start_camera()
                    return
                else:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame_num = 0
                    self.paused = True
                    self._play_action.setIcon(self.play_icon)
                    self._play_action.setText(self.texts["play"])
                    return
                    
            # Updates the current frame number (only for video file)
            if not self.live_mode:
                self.current_frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.update_progress_slider()

            # if recording, saves the frame 
            if self.recording and self.video_writer is not None:
                self.video_writer.write(frame)

            # Frame skipping logic for continuous detection
            should_detect = True
            if self.continuous_detection and self.velocity:
                self.frame_skip_counter += 1
                should_detect = (self.frame_skip_counter % self.detection_every_n_frames == 0)
                
            # Adds timestamp to frame (live mode only)
            if self.live_mode:
                now = datetime.now()
                timestamp = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                
                # text configs
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2
                text_color = (255, 255, 255)  # white letters
                bg_color = (0, 0, 0)  # black background
                
                # Calculates text size
                (text_width, text_height), _ = cv2.getTextSize(timestamp, font, font_scale, font_thickness)
                
                # Positions text in the lower right corner
                margin = 10
                text_x = frame.shape[1] - text_width - margin
                text_y = frame.shape[0] - margin
                
                #draws a rect for the background 
                cv2.rectangle(frame, 
                            (text_x - margin, text_y - text_height - margin), 
                            (text_x + text_width + margin, text_y + margin), 
                            bg_color, -1)
                
                #writes the text of the timestamp
                cv2.putText(frame, timestamp, 
                            (text_x, text_y), 
                            font, font_scale, text_color, font_thickness)
            
            if should_detect and self.continuous_detection and self.model:
                if frame is None:
                    return

                frame_copy = np.ascontiguousarray(frame.copy())
                if frame_copy is None or frame_copy.size == 0:
                    return

                frame_hash, frame_small = self.frame_to_small_and_hash(frame_copy)
                if self.last_frame_hash is None or not self.similar_frames(self.last_frame_small, frame_small, threshold=2):
                    self.last_frame_hash = frame_hash
                    self.last_frame_small = frame_small
                    frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    self.detection_thread.set_frame(frame_copy, frame_num)
                
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            #Resizing while maintaining aspect ratio
            h, w = frame.shape[:2]
            target_w = self.video_label.width()
            target_h = self.video_label.height()
            
            if w/h > target_w/target_h:
                new_w = target_w
                new_h = int(h * target_w / w)
            else:
                new_h = target_h
                new_w = int(w * target_h / h)
                
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            #frame display
            h, w = frame.shape[:2]
            bytes_per_line = 3 * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.video_label.setPixmap(pixmap)
            self.update_time_labels()
                
        except Exception as e:
            print(self.texts["debug_fatal_update_frame"].format(traceback.format_exc()))
            self.set_status_message("fatal_error")
            self.paused = True
            self.timer.stop()
            self.show_error_message("error", "fatal_error_detail", str(e))

    def detect_objects(self):
        if self.cap is None or not self.cap.isOpened():
            self.status_label.setText(self.texts["no_loaded"])
            return
        
        self.paused = True
        is_dark = self.palette().color(QPalette.ColorRole.Window).lightness() < 128
    
        if is_dark:
            self._play_action.setIcon(self.recolor_icon(QStyle.StandardPixmap.SP_MediaPlay))
        else:
            self._play_action.setIcon(self.play_icon)
        self._play_action.setText(self.texts["play"])
        self.timer.stop()
        
        current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = self.cap.read()
        if not ret:
            self.set_status_message("error_reading_frame")
            return
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        
        if self.model is None:
            self.set_status_message("no_model_loaded")
            return
        
        try:
            self.detect_objects_in_frame(frame)
            self.set_status_message("detection_completed", int(current_pos))
        except Exception as e:
            self.set_status_message("detection_error", str(e))

    def detect_objects_in_frame(self, frame):
        if self.model is None:
            self.set_status_message("no_model_loaded")
            return False
        
        try:
            self.annotations = []
            
            frame_copy = np.ascontiguousarray(frame)
            
            with torch.no_grad():
                results = self.model.track(
                    frame_copy,
                    conf=0.5,
                    iou=0.5,
                    persist=True,
                    verbose=False
                )

            if len(results) > 0:
                plotted_frame = results[0].plot()  
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = self.model.names[cls_id]
                    track_id = int(box.id) if box.id is not None else None
                        
                    detection_frame = self.capture_current_frame()
                    
                    detection = {
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "label": label, 
                        "confidence": conf,
                        "type": "auto", 
                        "class": label,
                        "timestamp": self.get_video_timestamp(self.current_frame_num),
                        "track_id": track_id,
                        "frame_number": self.current_frame_num,
                        "video_path": self.video_path,
                        "frame": detection_frame,
                        "frame_dimensions": f"{frame_copy.shape[1]}x{frame_copy.shape[0]}",
                        "frame_source": (self.video_path or "Live", self.current_frame_num) 
                    }
                        
                    self.annotations.append(detection)
                    self.detections_dock.add_detection(detection)

                self.display_frame(plotted_frame)
                return True
                
            return False
            
        except Exception as e:
            print(self.texts["detectton_error"].format(traceback.format_exc()))
            self.set_status_message("detection_error", str(e))
            return False
    
    def on_detection_finished(self, results, used_frame, frame_num):
        if results is None or not results.boxes:
            return

        plotted = results.plot()          
        self.display_frame(plotted)

        rgb = cv2.cvtColor(used_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id   = int(box.cls[0])
            conf     = float(box.conf[0])
            label    = self.model.names[cls_id]
            track_id = int(box.id) if box.id is not None else None

            detection = {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "label": label,
                "confidence": conf,
                "type": "auto",
                "class": label,
                "timestamp": self.get_video_timestamp(frame_num),
                "track_id": track_id,
                "frame_number": frame_num,
                "video_path": self.video_path,               
                "frame_dimensions": f"{w}x{h}", 
                "frame_source": (self.video_path, frame_num) 
            }
            self.annotations.append(detection)
            self.detections_dock.add_detection(detection)

        self.set_status_message("detection_completed", frame_num)
    
    def toggle_detection(self):
            if not self.cap or not self.cap.isOpened():
                self.set_status_message("no_video_loaded")
                self.continuous_detection = False
                return

            self.continuous_detection = not self.continuous_detection
            
            if self.continuous_detection:
                if not self.model:
                    self.set_status_message("no_model_loaded_error")
                    self.continuous_detection = False
                    return
                
                self.toggle_button.setStyleSheet("""
                    QPushButton {
                        padding: 3px 8px;
                        border-radius: 4px;
                        border: 1px solid #5c9eff;
                        background-color: #5c9eff;
                        color: white;
                        min-width: 23px;
                        min-height: 23px;
                    }
                """)

                self.set_status_message("continuous_detection_on")
                
                if self.paused:
                    self.toggle_play_pause()
            else:
                is_dark = self.palette().color(QPalette.ColorRole.Window).lightness() < 128
                self.toggle_button.setStyleSheet(f"""
                    QPushButton {{
                        padding: 3px 8px;
                        border-radius: 4px;
                        border: 1px solid {'#666' if is_dark else '#aaa'};
                        background-color: {'#333' if is_dark else '#f0f0f0'};
                        color: {'white' if is_dark else 'black'};
                        min-width: 23px;
                        min-height: 23px;
                    }}
                    QPushButton:hover {{
                        background-color: {'#555' if is_dark else '#e0e0e0'};
                    }}
                    QPushButton:pressed {{
                        background-color: {'#2a82da' if is_dark else '#ccc'};
                    }}
                """)
                self.set_status_message("continuous_detection_off")
                

    def enable_manual_annotation(self):
        self.paused = True
        is_dark = self.palette().color(QPalette.ColorRole.Window).lightness() < 128 
        if is_dark:
            self._play_action.setIcon(self.recolor_icon(QStyle.StandardPixmap.SP_MediaPlay))
        else: 
            self._play_action.setIcon(self.play_icon)
        self._play_action.setText(self.texts["play"])
        self.timer.stop()

        self.video_label.drawing_enabled = not self.video_label.drawing_enabled

        if self.video_label.drawing_enabled:
            if not self.video_label.current_class:
                first = next(iter(self.taxon_grid._buttons.keys()), None)
                if first:
                    self.taxon_grid.select(first)
                else:
                    QMessageBox.warning(self, self.texts["warning"], self.texts["no_classes_loaded"])
                    self.video_label.drawing_enabled = False
                    return

            self.annotate_button.setStyleSheet("""
                QPushButton {
                    padding: 3px 8px;
                    border-radius: 4px;
                    border: 1px solid #5c9eff;
                    background-color: #5c9eff;
                    color: white;
                    min-width: 23px;
                    min-height: 23px;
                }
            """)
            self.set_status_message("manual_annotation_on")

        else:
            is_dark = self.palette().color(QPalette.ColorRole.Window).lightness() < 128
            self.annotate_button.setStyleSheet(f"""
                QPushButton {{
                    padding: 3px 8px;
                    border-radius: 4px;
                    border: 1px solid {'#666' if is_dark else '#aaa'};
                    background-color: {'#333' if is_dark else '#f0f0f0'};
                    color: {'white' if is_dark else 'black'};
                    min-width: 23px;
                    min-height: 23px;
                }}
                QPushButton:hover {{
                    background-color: {'#555' if is_dark else '#e0e0e0'};
                }}
                QPushButton:pressed {{
                    background-color: {'#2a82da' if is_dark else '#ccc'};
                }}
            """)
            self.set_status_message("manual_annotation_off")

        self.video_label.update()

    def change_drawing_class(self, name):
        """Updates class, color, and filters whenever the user chooses a taxon in the grid"""
        self.video_label.current_class = name
        hue = hash(name) % 360
        self.video_label.drawing_color = QColor.fromHsv(hue, 255, 200)

        if name not in self.custom_classes and \
           (not self.model or name not in self.model.names.values()):
            self.custom_classes.append(name)

        if hasattr(self, 'detections_dock'):
            if self.detections_dock.class_filter.findText(name) == -1:
                self.detections_dock.class_filter.addItem(name)

    def display_frame(self, frame):
        try:
            if isinstance(frame, np.ndarray):
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                #Smooth resizing
                h, w = frame.shape[:2]
                self.video_label._aspect_ratio = w / h
                
                #Create QImage directly from numpy array
                bytes_per_line = 3 * w
                q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
            else:
                pixmap = frame
            
            self.video_label.active_annotations = []
            self.video_label.update()

            self.video_label._pixmap = pixmap
            self.update_video_display()
            
            if self.paused:
                self.video_label.update()
                
        except Exception as e:
            print(self.texts["frame_error"].format(traceback.format_exc()))
            self.set_status_message("frame_error", str(e))

    def update_video_display(self):
        """Updates the video display while maintaining the aspect ratio"""
        if not hasattr(self.video_label, '_pixmap') or self.video_label._pixmap is None:
            return
        
        target_size = self.video_label.size()
        if self.video_label._aspect_ratio is not None:
            if target_size.width() / target_size.height() > self.video_label._aspect_ratio:
                height = target_size.height()
                width = int(height * self.video_label._aspect_ratio)
            else:
                width = target_size.width()
                height = int(width / self.video_label._aspect_ratio)
            target_size = QSize(width, height)
        
        scaled_pixmap = self.video_label._pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.video_label.video_x = (self.video_label.width() - scaled_pixmap.width()) // 2
        self.video_label.video_y = (self.video_label.height() - scaled_pixmap.height()) // 2
        self.video_label.video_rect = QRect(
            self.video_label.video_x,
            self.video_label.video_y,
            scaled_pixmap.width(),
            scaled_pixmap.height()
        )
        
        self.video_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        """Redraws the frame when the window is resized"""
        super().resizeEvent(event)
        self.update_video_display()

    def toggle_play_pause(self):
        if self.cap is None:
            return
        
        is_dark = self.palette().color(QPalette.ColorRole.Window).lightness() < 128   
        self.paused = not self.paused
        if self.paused:
            if is_dark:
                self._play_action.setIcon(self.recolor_icon(QStyle.StandardPixmap.SP_MediaPlay))
                self._play_action.setText(self.texts["play"])
                self.timer.stop()
            else:
                self._play_action.setIcon(self.play_icon)
                self._play_action.setText(self.texts["play"])
                self.timer.stop()
        else:
            if is_dark:
                self._play_action.setIcon(self.recolor_icon(QStyle.StandardPixmap.SP_MediaPause))
                self._play_action.setText(self.texts["pause"])
                if self.velocity:
                    fps = self.cap.get(cv2.CAP_PROP_FPS)
                    if fps > 0:
                        interval = int(1000 / (fps * 2))
                        self.timer.start(interval)
                    else:
                        self.timer.start(16)  
                else:
                    fps = self.cap.get(cv2.CAP_PROP_FPS)
                    interval = int(1000 / fps) if fps > 0 else 30
                    self.timer.start(interval)
                self.video_label.reset_annotations()
            else:
                self._play_action.setIcon(self.pause_icon)
                self._play_action.setText(self.texts["pause"])
                if self.velocity:
                    fps = self.cap.get(cv2.CAP_PROP_FPS)
                    if fps > 0:
                        interval = int(1000 / (fps * 2))
                        self.timer.start(interval)
                    else:
                        self.timer.start(16)  
                else:
                    fps = self.cap.get(cv2.CAP_PROP_FPS)
                    interval = int(1000 / fps) if fps > 0 else 30
                    self.timer.start(interval)
                self.video_label.reset_annotations()
        
        self.video_label.update()

    def previous_frame(self):
        if self.dataset_mode and self.dataset_frames:
            idx = self.dataset_index - 1
            if idx >= 0:
                self.load_dataset_frame(idx)
            return
        
        if self.cap is None:
            return
            
        new_pos = max(self.current_frame_num - 30, 0) # Calculate new position (30 frames back)
        self.set_current_frame(new_pos)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos) # Defines new position 
        ret, frame = self.cap.read()
        if ret:
            # updates the number of the current frame 
            self.current_frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        
            # exhibits the frame 
            self.display_frame(frame)
            
            # updates the progress bar 
            self.update_progress_slider()
            
            # updates the time labels 
            self.update_time_labels()


    def next_frame(self):
        if self.dataset_mode and self.dataset_frames:
            idx = self.dataset_index + 1
            if idx < len(self.dataset_frames):
                self.load_dataset_frame(idx)
            return
        if self.cap is None:
            return
            
        # calculates new position (30 frames ->)
        new_pos = min(self.current_frame_num + 30, self.total_frames - 1)
        self.set_current_frame(new_pos)
        
        # defines the new position and reads the frame 
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
        ret, frame = self.cap.read()
        
        if ret:
            # updates the number of the current frame 
            self.current_frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # exhibits the frame
            self.display_frame(frame)
            
            # updates the progress bar
            self.update_progress_slider()
            
            # updates the time labels
            self.update_time_labels()
                
    def update_progress_slider(self):
        """Updates the slider position"""
        if self.total_frames > 0:
            progress = int((self.current_frame_num / self.total_frames) * 100)
            self.progress_slider.setValue(progress)            

    def seek_video(self, value):
        if self.live_mode:
            return  
        if self.cap is None:
            return
            
        frame_pos = int((value / 100) * self.total_frames)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        self.current_frame_num = frame_pos
        self.update_frame()
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
        self.update_time_labels()

    def pause_video_for_seeking(self):
        self.was_playing = not self.paused
        if self.was_playing:
            self.toggle_play_pause()

    def resume_video_after_seeking(self):
        if self.was_playing:
            self.toggle_play_pause()

    def update_time_labels(self):
        if self.cap is None:
            return
        
        if self.live_mode:
            self.current_time_label.setText("Live")
            self.total_time_label.setText("")
            return
            
        current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_time = self.get_video_timestamp(current_frame)
        self.current_time_label.setText(current_time)
        
        if not hasattr(self, 'total_video_time'):
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                total_seconds = self.total_frames / fps
                hours = int(total_seconds // 3600)
                minutes = int((total_seconds % 3600) // 60)
                seconds = int(total_seconds % 60)
                self.total_video_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                self.total_video_time = "00:00:00"
        
        self.total_time_label.setText(self.total_video_time)

    def get_video_timestamp(self, frame_num):
        if self.cap is None:
            return "00:00:00"
        
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            return "00:00:00"
        
        total_seconds = frame_num / fps
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
    
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def velocity2(self):
        """Aumenta a velocidade do vídeo"""
        if self.cap is None:
            return
        if self.velocity:
            self.velocity = False
            # stops the current timer if is running
            if self.timer.isActive():
                self.timer.stop()
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            new_interval = int(1000 / fps)
            self.timer.start(new_interval)
            self.velocity2_button.setStyleSheet("background-color: None")

            self.set_status_message("speed_format", "1.0", fps)

        else:
            # activates 2x velocity 
            self.velocity = True
            self.frame_skip_counter = 0
            self.velocity2_button.setStyleSheet("background-color: #5c9eff;")
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # dif configs for continuous detection/non-continuous detection mode  
            if self.continuous_detection:
                new_interval = int(1000 / (fps * 2))  
                self.detection_every_n_frames = 2  
                
                self.set_status_message("speed_detection_format", "2.0", self.detection_every_n_frames)
            else:
                # Normal mode without continuous detection - simply doubles the speed
                new_interval = int(1000 / (fps * 2))
                self.set_status_message("speed_format", "2.0", fps*2)
            
            if self.timer.isActive():
                self.timer.stop()
            self.timer.start(new_interval)

    def export_yolo_annotations(self, output_dir):
        if not hasattr(self, 'all_detections'):
            QMessageBox.warning(self, self.texts["warning"], self.texts["no_annotations_to_export"])
            return
        
        # filters only valid manual annotations
        manual_annotations = [
            ann for ann in self.all_detections 
            if ann.get("type") == "manual" and 
            all(key in ann for key in ["x1", "y1", "x2", "y2", "class"])
        ]
        
        if not manual_annotations:
            self.show_warning_message("warning", "no_manual_annotations")
            return
        
        try:
            images_dir = os.path.join(output_dir, "images")
            labels_dir = os.path.join(output_dir, "labels")
            train_dir_images = os.path.join(images_dir, "train")
            val_dir_images = os.path.join(images_dir, "val")
            train_dir_labels = os.path.join(labels_dir, "train")
            val_dir_labels = os.path.join(labels_dir, "val")
            
            os.makedirs(train_dir_images, exist_ok=True)
            os.makedirs(val_dir_images, exist_ok=True)
            os.makedirs(train_dir_labels, exist_ok=True)
            os.makedirs(val_dir_labels, exist_ok=True)
            
            classes = sorted(list(set(ann["class"] for ann in manual_annotations)))
            class_to_id = {name: idx for idx, name in enumerate(classes)}
            
            frames_dict = defaultdict(list)
            for ann in manual_annotations:
                frame_num = ann.get("frame_number", 0)
                frames_dict[frame_num].append(ann)
            
            all_frames = sorted(frames_dict.keys())
            
            # Convert frame numbers to dataset indexes
            if self.dataset_mode and hasattr(self, 'dataset_frames') and self.dataset_frames:
                # Dataset mode: use direct indexes
                dataset_indices = list(range(len(self.dataset_frames)))
                split_idx = int(0.8 * len(dataset_indices))
                train_indices = set(dataset_indices[:split_idx])
                val_indices = set(dataset_indices[split_idx:])
            else:
                # Video mode: use frame numbers
                split_idx = int(0.8 * len(all_frames))
                train_frames = set(all_frames[:split_idx])
                val_frames = set(all_frames[split_idx:])
                train_indices = train_frames
                val_indices = val_frames
            
            progress = QProgressDialog(self.texts["exporting_frames"], self.texts["cancel"], 0, len(all_frames), self)
            progress.setWindowTitle(self.texts["exporting_dataset"])
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.show()
            
            processed_frames = 0

            video_name_prefix = ""
            if not self.dataset_mode and self.video_path and self.video_path != "Live":
                video_name = Path(self.video_path).stem
                video_name_prefix = "".join(c for c in video_name if c.isalnum() or c in ('_', '-'))
            
            for i, frame_num in enumerate(all_frames):
                progress.setValue(i)
                QApplication.processEvents()
                
                if progress.wasCanceled():
                    break
                
                frame = None
                img_name = None
                
                if self.dataset_mode and hasattr(self, 'dataset_frames') and self.dataset_frames:
                    for dataset_path, original_frame_num, dataset_index in self.dataset_frames:
                        if dataset_index == frame_num:
                            img_name = Path(dataset_path).name
                            frame = cv2.imread(dataset_path)
                            break
                elif self.cap and self.cap.isOpened():
                    current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = self.cap.read()
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
                    
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if video_name_prefix:
                            img_name = f"{video_name_prefix}_{frame_num:06d}.jpg"
                        else:
                            img_name = f"frame_{frame_num:06d}.jpg"
                
                if frame is None:
                    continue
                
                #  Determine subdir based on index
                if self.dataset_mode and hasattr(self, 'dataset_frames'):
                    # find corresponding index
                    frame_index = -1
                    for idx, (_, _, dataset_idx) in enumerate(self.dataset_frames):
                        if dataset_idx == frame_num:
                            frame_index = idx
                            break
                    
                    is_train = frame_index in train_indices
                else:
                    is_train = frame_num in train_frames
                
                img_subdir = "train" if is_train else "val"
                
                # Save image
                img_path = os.path.join(images_dir, img_subdir, img_name)
                cv2.imwrite(img_path, frame)

                processed_frames += 1

                # Save label
                label_name = Path(img_name).stem + ".txt"
                label_path = os.path.join(labels_dir, img_subdir, label_name)
                with open(label_path, 'w') as f:
                    for ann in frames_dict[frame_num]:
                        try:
                            h, w = frame.shape[:2]
                            x1 = max(0, ann["x1"])
                            y1 = max(0, ann["y1"])
                            x2 = min(w, ann["x2"])
                            y2 = min(h, ann["y2"])

                            # convert to yolo format 
                            x_center = ((x1 + x2) / 2) / w
                            y_center = ((y1 + y2) / 2) / h
                            width = (x2 - x1) / w
                            height = (y2 - y1) / h

                            x_center = max(0.0, min(1.0, x_center))
                            y_center = max(0.0, min(1.0, y_center))
                            width = max(0.0, min(1.0, width))
                            height = max(0.0, min(1.0, height))
                            
                            class_id = class_to_id[ann["class"]]
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                        except Exception as e:
                            print(f"❌ Erro na anotação {ann}: {e}")
                            continue

            progress.close()

            if self.dataset_mode:
                bg_count = self.add_background_images_to_dataset(images_dir, train_indices, val_indices)
            else:
                bg_count = 0
            
            # create dataset.yaml 
            yaml_path = os.path.join(output_dir, "dataset.yaml")
            with open(yaml_path, 'w') as f:
                f.write(f"# YOLO Dataset Configuration\n\n")
                out_path = os.path.abspath(output_dir).replace('\\', '/')
                f.write(f"path: {out_path}/\n")
                f.write(f"train: images/train\n")
                f.write(f"val: images/val\n")
                f.write(f"names:\n")
                for idx, name in enumerate(classes):
                    f.write(f"  {idx}: {name}\n")
            
            self.show_info_message(
                "export_completed",
                "export_success",
                processed_frames,
                len(all_frames),
                len(manual_annotations),
                len(classes),
                bg_count,
                output_dir
            )
            
        except Exception as e:
            self.show_error_message(
                "error",
                "export_error",
                str(e),
                traceback.format_exc()
            )

    def export_yolo_annotations_dialog(self):
        """Opens dialog to select output directory""" 
        output_dir = QFileDialog.getExistingDirectory(
            self, self.texts["export_yolo_dialog"])
        
        if output_dir:
            self.export_yolo_annotations(output_dir)

    def train_yolo_model(self):
        """Start training the YOLO model with manual annotations"""
        if not hasattr(self, 'all_detections'):
            self.all_detections = []

        for ann in self.video_label.active_annotations:
            if ann.get("type") == "manual" and ann not in self.all_detections:
                ann.setdefault("frame_number", self.current_frame_num)
                ann.setdefault("video_path", self.video_path or "Live")
                self.all_detections.append(ann)

        if not any(d.get("type") == "manual" for d in self.all_detections):
            self.show_warning_message("warning", "no_manual_annotations_train")
            return

        try:
            models_dir = os.path.join(os.getcwd(), "models")
            os.makedirs(models_dir, exist_ok=True)
            # Asks where to save the dataset and the trained model 
            dataset_dir = QFileDialog.getExistingDirectory(
                self, self.texts["train_dataset_dialog"])
            
            if not dataset_dir:
                return
            
            name, ok = QInputDialog.getText(
                self,
                self.texts["name_model_title"],
                self.texts["name_model_label"]
            )
            if not ok or not name.strip():
                return  # user canceled 
            safe_name = "".join(c for c in name.strip() if c.isalnum() or c in ("_", "-"))
            if not safe_name:
                safe_name = "custom_model"

            # First exports the annotations in YOLO format 
            self.export_yolo_annotations(dataset_dir)
            
            # Get unique classes from manual annotations
            manual_annotations = [
                ann for ann in self.all_detections 
                if ann.get("type") == "manual" and 
                all(key in ann for key in ["x1", "y1", "x2", "y2", "class"])
            ]
            
            classes = sorted(list(set(ann["class"] for ann in manual_annotations)))
            
            # yolo dataset configuration 
            config = {
                "path": os.path.abspath(dataset_dir).replace("\\", "/") + "/",
                "train": "images/train",
                "val": "images/val",
                "names": {i: name for i, name in enumerate(classes)},
                "nc": len(classes)
            }
            
            # saves YAML configuration file 
            config_path = os.path.join(dataset_dir, "dataset.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(config, f)

            images_dir = Path(dataset_dir) / "images"
            
            # verifies is dataset_frames exists 
            is_dataset_mode = self.dataset_mode and hasattr(self, 'dataset_frames') and self.dataset_frames
            
            if is_dataset_mode:
                # Gets indexes form all dataset frames 
                all_indices = set(range(len(self.dataset_frames)))
                
                # identifies which indexes has annotations 
                frames_with_annotations = {ann.get("frame_number", 0) for ann in manual_annotations}
                
                # maps frame numbers for dataset indexes 
                annotated_indices = set()
                for idx, (_, _, dataset_idx) in enumerate(self.dataset_frames):
                    if dataset_idx in frames_with_annotations:
                        annotated_indices.add(idx)
                
                # divides annotadet frames  80/20
                annotated_list = list(annotated_indices)
                split_point = int(len(annotated_list) * 0.8)
                train_indices = set(annotated_list[:split_point])
                val_indices = set(annotated_list[split_point:])
                
                # identifies frames without annotation (background)
                background_indices = all_indices - annotated_indices
                
                if background_indices:
                    bg_list = list(background_indices)
                    bg_split = int(len(bg_list) * 0.8)
                    train_bg = set(bg_list[:bg_split])
                    val_bg = set(bg_list[bg_split:])
                    
                    bg_count = self.add_background_images_to_dataset(images_dir, train_bg, val_bg)
                else:
                    bg_count = 0
            else:
                bg_count = 0
            
            if bg_count > 0:
                self.set_status_message("background_images_added", bg_count)
            
            # training settings 
            train_config = {
                "data": config_path,
                "epochs": 100,
                "imgsz": 640,
                "batch": 8,
                "name": "custom_model",
                "exist_ok": True,
                "patience": 20,
                "optimizer": "auto",
                "lr0": 0.01,
                "device": "0" if torch.cuda.is_available() else "cpu",
                "workers": 4,
                "save_period": 10,
                "single_cls": False,
                "augment": True,
                "name": os.path.join(models_dir, safe_name)
            }
            
            # advanced settings file 
            advanced_dialog = QDialog(self)
            advanced_dialog.setWindowTitle(self.texts["advanced_training_settings"])
            layout = QVBoxLayout()
            
            form_layout = QFormLayout()
            
            epochs_spin = QSpinBox()
            epochs_spin.setRange(1, 1000)
            epochs_spin.setValue(train_config["epochs"])
            form_layout.addRow(self.texts["epochs"], epochs_spin)
            
            batch_spin = QSpinBox()
            batch_spin.setRange(1, 64)
            batch_spin.setValue(train_config["batch"])
            form_layout.addRow(self.texts["batch_size"], batch_spin)
            
            imgsz_spin = QSpinBox()
            imgsz_spin.setRange(320, 1280)
            imgsz_spin.setSingleStep(32)
            imgsz_spin.setValue(train_config["imgsz"])
            form_layout.addRow(self.texts["image_size"], imgsz_spin)
            
            lr_spin = QDoubleSpinBox()
            lr_spin.setRange(0.0001, 0.1)
            lr_spin.setSingleStep(0.001)
            lr_spin.setValue(train_config["lr0"])
            form_layout.addRow(self.texts["learning_rate"], lr_spin)
            
            device_combo = QComboBox()
            device_combo.addItems(["CPU", "GPU"] if torch.cuda.is_available() else ["CPU"])
            form_layout.addRow(self.texts["device"], device_combo)
            
            button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            button_box.accepted.connect(advanced_dialog.accept)
            button_box.rejected.connect(advanced_dialog.reject)
            
            layout.addLayout(form_layout)
            layout.addWidget(button_box)
            advanced_dialog.setLayout(layout)
            
            if advanced_dialog.exec() == QDialog.DialogCode.Accepted:
                train_config.update({
                    "epochs": epochs_spin.value(),
                    "batch": batch_spin.value(),
                    "imgsz": imgsz_spin.value(),
                    "lr0": lr_spin.value(),
                    "device": "0" if device_combo.currentText() == "GPU" else "cpu"
                })
            else:
                return
            
            # Progress dialog configuration
            progress = QProgressDialog(
                self.texts["training_progress"],   
                self.texts["cancel"],
                0,
                train_config["epochs"],       
                self
            )
            progress.setWindowTitle(self.texts["training_model"])
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.show()

            self._last_model_path = os.path.join(models_dir, safe_name, "weights", "best.pt")

            self.train_thread = TrainThread(train_config)
            self.train_thread.epoch_progress.connect(progress.setValue)   
            self.train_thread.finished.connect(lambda: self.on_training_finished(progress))
            self.train_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Erro", self.texts["config_failed"].format(str(e)))
            print(self.texts["debug_config_failed"].format(traceback.format_exc()))

    def on_training_finished(self, progress):
        """Dealing with the end of training"""
        progress.close()
        
        if self.train_thread.success:
            model_path = self._last_model_path
            
            reply = QMessageBox.question(
                self,
                self.texts["training_completed"],
                self.texts["training_success"].format(model_path),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.load_model(model_path)
        else:
            QMessageBox.critical(
                self,
                self.texts["training_error_title"],
                self.texts["training_error"].format(self.train_thread.error)
            ) 

    def save_annotations(self):
        """Export annotations to CSV + save original frames to a user-chosen folder"""
        if not self.video_path and not self.live_mode:
            self.status_label.setText(self.texts["no_loaded"])
            return

        # Determine default filename based on mode
        default_name = "annotations.csv" if self.live_mode else \
            f"{os.path.splitext(os.path.basename(self.video_path))[0]}_annotations.csv"

        # 1. Show save file dialog for CSV
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            self.texts["save_annotations"],
            default_name,
            "CSV Files (*.csv);;All Files (*)"
        )

        if not output_path:
            return

        # 2.  Ask user where to save frames
        frames_dir = QFileDialog.getExistingDirectory(
            self,
            "Choose folder to save frames",
            str(Path(output_path).parent)
        )

        if not frames_dir:  # user cancelled
            return

        frames_dir = Path(frames_dir)  # converts to Path object 

        try:
            # 3. Create frames directory if it doesn't exist
            frames_dir.mkdir(exist_ok=True)

            # 4. Collect all detections from all sources
            all_detections = []
            
            # From video_label (includes manual annotations across all frames)
            if hasattr(self.video_label, 'frame_annotations'):
                for frame_num, annotations in self.video_label.frame_annotations.items():
                    for ann in annotations:
                        # Ensure video_path field exists
                        if "video_path" not in ann:
                            ann["video_path"] = self.video_path if self.video_path else "Live"
                        all_detections.append(ann)

            # From detections dock (includes automated detections)
            if hasattr(self, 'detections_dock'):
                dock_detections = self.detections_dock.filter_all()
                for d in dock_detections:
                    if "video_path" not in d:
                        d["video_path"] = self.video_path if self.video_path else "Live"
                    all_detections.append(d)

            # 5. Filter out training annotations (keep all other types)
            filtered_detections = [d for d in all_detections if d.get("type") != "training"]

            # 6. Remove duplicate detections based on unique identifiers
            unique_detections = []
            seen = set()
            for d in filtered_detections:
                key = (
                    d.get("video_path"),
                    d.get("frame_number"),
                    d.get("timestamp"),
                    d.get("class"),
                    d.get("x1"), d.get("y1"), d.get("x2"), d.get("y2")
                )
                if key not in seen:
                    seen.add(key)
                    unique_detections.append(d)

            # 7. Group by track_id, keeping highest confidence per tracked object
            best_by_track = {}
            for d in unique_detections:
                tid = d.get("track_id")
                if tid is not None:
                    # For tracked objects, keep the highest confidence detection
                    if tid not in best_by_track or d.get("confidence", 0) > best_by_track[tid].get("confidence", 0):
                        best_by_track[tid] = d
                else:
                    # Keep all manual/untacked detections
                    key = f"manual_{d.get('frame_number', 0)}_{id(d)}"
                    best_by_track[key] = d

            # 8. Extract and save frames 
            saved_frames = {}  # Cache: (video_path, frame_num) -> relative_path
            progress = QProgressDialog(self.texts["exporting_frames"], self.texts["cancel"], 0, len(best_by_track), self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.show()

            for i, ann in enumerate(best_by_track.values()):
                if progress.wasCanceled():
                    break
                    
                frame_num = ann.get("frame_number")
                video_path = ann.get("video_path", "Live")
                
                # Check if frame was already saved to avoid duplication
                frame_key = (video_path, frame_num)
                if frame_key in saved_frames:
                    ann["frame_path"] = saved_frames[frame_key]
                    continue

                # Determine frame save path
                if video_path == "Live":
                    # For Live mode: save current frame from memory
                    if hasattr(self, 'current_frame') and self.current_frame is not None:
                        frame_name = f"live_frame_{frame_num or 0}.jpg"
                        frame_path = frames_dir / frame_name
                        cv2.imwrite(str(frame_path), self.current_frame)
                        saved_frames[frame_key] = str(frame_path)  
                    else:
                        saved_frames[frame_key] = "N/A"
                else:
                    # For video files: extract specific frame
                    video_name = Path(video_path).stem
                    frame_name = f"{video_name}_frame_{frame_num:06d}.jpg"
                    frame_path = frames_dir / frame_name
                    
                    # Avoid re-extracting if frame already exists
                    if not frame_path.exists():
                        cap = cv2.VideoCapture(str(video_path))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                        ret, frame = cap.read()
                        if ret:
                            cv2.imwrite(str(frame_path), frame)
                        cap.release()
                    
                    saved_frames[frame_key] = str(frame_path)  

                ann["frame_path"] = saved_frames[frame_key]
                progress.setValue(i)

            progress.close()

            # 9. Prepare CSV data with frame paths
            export_data = []
            for ann in best_by_track.values():
                video_path = ann.get("video_path", "")
                video_name = "Live" if video_path == "Live" else os.path.basename(str(video_path)) if video_path else "Unknown"
                
                confidence = ann.get('confidence', 0)
                confidence_str = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else str(confidence)
                
                export_data.append({
                    "Video": video_name, 
                    "Timestamp": ann.get("timestamp", ""),
                    "Taxon": ann.get("class", "Unknown"),
                    "Confidence": confidence_str,
                    "Type": ann.get("type", "unknown"),
                    "Track_ID": ann.get("track_id", ""),
                    "x1": ann.get("x1", ""),
                    "y1": ann.get("y1", ""),
                    "x2": ann.get("x2", ""),
                    "y2": ann.get("y2", ""),
                    "Frame_Number": ann.get("frame_number", ""),
                    "Photo": ann.get("frame_path", "")
                })

            # 10. Save CSV file
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ["Video", "Timestamp", "Taxon", "Confidence", "Type", "Track_ID",
                            "x1", "y1", "x2", "y2", "Frame_Number", "Photo"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(export_data)

            self.status_label.setText(self.texts["annotations_saved"].format(output_path))
            QMessageBox.information(
                self, 
                self.texts['export_completed'], 
                f"{self.texts['annotations_saved'].format(output_path)}\n"
                f"{self.texts['frames_saved'].format(frames_dir)}\n"
            )

        except Exception as e:
            self.set_status_message("saving_error")
            print(f"{self.texts['error_colon']} {traceback.format_exc()}")
            QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def load_annotations(self, file_path):
        """Load annotations from a JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # checks if it is a valid annotation file 
            if 'frames' not in data:
                raise ValueError(self.texts["invalid_annotations"])
                
            # clears existing annotations 
            self.annotations = []
            self.video_label.manual_annotations = []
            self.detections_dock.all_detections = []
            self.detections_dock.detections_list.clear()
            
            # loads custom classes
            if 'custom_classes' in data:
                self.custom_classes = data['custom_classes']
                
            # loads annotations
            for frame_data in data['frames']:
                # Adds automated annotations
                if 'auto_annotations' in frame_data:
                    for ann in frame_data['auto_annotations']:
                        self.annotations.append(ann)
                        self.detections_dock.add_detection(ann)
                
                # Adds manual annotations
                if 'manual_annotations' in frame_data:
                    for ann in frame_data['manual_annotations']:
                        self.video_label.manual_annotations.append(ann)
                        self.detections_dock.add_detection(ann)
            
            # updates model informations 
            if 'model_used' in data:
                self.model_path = data['model_used']
                self.set_status_message("annotations_loaded_with_model",
                                        os.path.basename(file_path),
                                        self.model_path)
            else:
                self.set_status_message("annotations_loaded",
                                        os.path.basename(file_path))
                
            # updates class filter 
            self.detections_dock.update_class_filter()
            
            # If a video is loaded, display the annotations on the current frame
            if self.cap is not None and self.cap.isOpened():
                self.display_frame(self.video_label._pixmap)
                
            self.detections_dock.apply_filters()
                
        except Exception as e:
            self.show_error_message("error", "load_annotations_error", str(e))
            self.set_status_message("load_annotations_status_error", str(e))
            print(self.texts["debug_load_annotations_error"].format(traceback.format_exc()))

    def save_current_frame_with_annotations(self):
        # Make sure we have something on screen
        if not hasattr(self.video_label, '_pixmap') or self.video_label._pixmap is None:
            self.status_label.setText(self.texts["no_frame_to_save"])
            return

        # Start from the pixmap already shown (includes YOLO auto-boxes)
        pixmap = self.video_label._pixmap.copy()

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Get the rectangle where the video is drawn in the label
        video_rect = self.video_label.video_rect
        if not video_rect:
            video_rect = QRect(0, 0, pixmap.width(), pixmap.height())

        # Draw only manual annotations on top
        for ann in self.video_label.active_annotations:
            # Annotation coordinates are in original-frame pixels
            x1_px = int(ann["x1"])
            y1_px = int(ann["y1"])
            x2_px = int(ann["x2"])
            y2_px = int(ann["y2"])

            # Map original-frame pixels to pixmap pixels
            scale_x = video_rect.width() / (self.video_label.original_width or video_rect.width())
            scale_y = video_rect.height() / (self.video_label.original_height or video_rect.height())

            x1 = int(x1_px * scale_x)
            y1 = int(y1_px * scale_y)
            x2 = int(x2_px * scale_x)
            y2 = int(y2_px * scale_y)

            color = QColor(ann.get("color", self.drawing_color.name()))
            pen = QPen(color, 4)
            painter.setPen(pen)
            painter.drawRect(QRect(x1, y1, x2 - x1, y2 - y1))

            # Label background
            if "class" in ann:
                font = painter.font()
                font.setPixelSize(20)
                painter.setFont(font)
                text_width = painter.fontMetrics().horizontalAdvance(ann["class"]) + 10
                text_height = painter.fontMetrics().height()
                text_rect = QRect(x1, max(0, y1 - text_height - 4), text_width, text_height)
                painter.fillRect(text_rect, color)

                painter.setPen(QPen(Qt.GlobalColor.white, 1))
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, ann["class"])

        painter.end()

        # Default file name
        if self.video_path:
            video_name = os.path.splitext(os.path.basename(self.video_path))[0]
            default_name = f"{video_name}_frame_{self.current_frame_num}.png"
        else:
            default_name = f"frame_{self.current_frame_num}.png"

        # Ask where to save
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            self.texts["save_frame_dialog"],
            default_name,
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;All Files (*)"
        )

        if output_path:
            pixmap.save(output_path)
            self.status_label.setText(self.texts["frame_saved"].format(output_path))

    def show_shortcuts(self):
        shortcuts = [
            f"{self.texts['space']}: {self.texts['pause']}",
            f"Setas: {self.texts['navigate_frames']}",
            f"D: {self.texts['detect_frame']}",
            f"T: {self.texts['toggle_detection']}",
            f"M: {self.texts['annotate_manual']}",
            f"Ctrl+O: {self.texts['load_video']}",
            f"Ctrl+M: {self.texts['load_model']}",
            f"Ctrl+W: {self.texts['live']}",
            f"Ctrl+S: {self.texts['save_annotations']}",
            f"Ctrl+R: {self.texts['start_recording']}",
            f"Ctrl+Shift+R: {self.texts['stop_recording']}",
            f"Ctrl+Q: {self.texts['exit']}",
            f"Ctrl+L: {self.texts['load_annotations']}",
            f"F1: {self.texts['shortcuts']}"
        ]
        QMessageBox.information(self, self.texts["shortcuts"], "\n".join(shortcuts))

    def show_about(self):
        QMessageBox.about(self, self.texts["about"], self.texts["about_text"])

    def show_manual(self):
        # Determine which manual to open based on current language
        manual_filename = f"manual/manual_{self.language}.pdf"
        manual_path = resource_path(manual_filename)
        
        # Open PDF with system default viewer
        url = QUrl.fromLocalFile(manual_path)
        QDesktopServices.openUrl(url)
    
    def set_current_frame(self, frame_num):
        """Centralizes frame switching and synchronizes annotations"""
        if self.cap is None or self.live_mode:
            return

        self.current_frame_num = frame_num
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        self.video_label.current_frame_num = frame_num
        self.video_label.update_active_annotations()

        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
            self.update_progress_slider()
            self.update_time_labels()

    def add_manual_annotation_to_history(self, annotation):
        if not hasattr(self, 'all_detections'):
            self.all_detections = []
        
        annotation["frame_number"] = self.current_frame_num
        annotation["frame"] = self.capture_current_frame()
        
        self.all_detections.append(annotation)
        self.detections_dock.add_detection(annotation)

    def robust_read_csv(self, path):
        for enc in ('utf-8', 'latin1', 'iso-8859-1', 'cp1252'):
            for sep in (',', ';', '\t'):
                try:
                    with open(path, newline='', encoding=enc) as f:
                        sep = csv.Sniffer().sniff(f.read(1024)).delimiter
                    return pd.read_csv(path, sep=sep, encoding=enc)
                except Exception:
                    continue
            raise ValueError("Nenhum encoding ou separador compatível encontrado.")
        
    @staticmethod
    def parse_time_string(t: str) -> timedelta:
        try:
            return datetime.strptime(t, "%H:%M:%S.%f") - datetime.strptime("00:00:00.000", "%H:%M:%S.%f")
        except ValueError:
            return datetime.strptime(t, "%H:%M:%S") - datetime.strptime("00:00:00", "%H:%M:%S")
        
    def merge_annotations(self):
        # select annotation file
        self.show_info_message(
                "warning",
                "select_annotations_file"
            )
        
        file_path1, _ = QFileDialog.getOpenFileName(self, self.texts["select_annotations_file"], 
                                                   "", "CSV Files (*.csv);;All Files (*)")
        if not file_path1:
            return
        
        self.show_info_message(
                "warning",
                "select_georeferencing_file"
            )
        
        # select georeferencing file
        file_path2, _ = QFileDialog.getOpenFileName(self, self.texts["select_georeferencing_file"], 
                                                   "", "CSV Files (*.csv);;All Files (*)")
        if not file_path2:
            return
        
        try:
            df1 = self.robust_read_csv(file_path1)
            df2 = self.robust_read_csv(file_path2)

            dialog = QDialog(self)
            dialog.setWindowTitle(self.texts["choose_merge_columns"])
            layout = QFormLayout(dialog)

            combo1 = QComboBox()
            combo1.addItems(df1.columns)
            combo2 = QComboBox()
            combo2.addItems(df2.columns)

            layout.addRow(self.texts["key_column_left"], combo1)
            layout.addRow(self.texts["key_column_right"], combo2)

            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            layout.addWidget(buttons)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)

            if dialog.exec() != QDialog.DialogCode.Accepted:
                return

            col_left = combo1.currentText()
            col_right = combo2.currentText()

            # Convert both columns to timedelta.
            df1["time_delta"] = pd.to_timedelta(df1[col_left])
            df2["time_delta"] = pd.to_timedelta(df2[col_right])

            df1 = df1.sort_values("time_delta")
            df2 = df2.sort_values("time_delta")

            merged = pd.merge_asof(
                df1,
                df2,
                left_on="time_delta",
                right_on="time_delta",
                direction="nearest",
                tolerance=pd.Timedelta("1s")
            )

            base_name = os.path.splitext(os.path.basename(file_path1))[0]
            default_name = f"{base_name}_georreferenciado.csv"
            save_path, _ = QFileDialog.getSaveFileName(
                self,
                self.texts["save_merged_annotations"],
                default_name,
                "CSV Files (*.csv);;All Files (*)"
            )
            if not save_path:
                return

            merged.to_csv(save_path, index=False, encoding="utf-8")
            self.show_info_message("success", "merge_completed", save_path)

        except Exception as e:
            self.show_error_message("error", "merge_error", str(e))
            print(self.texts["debug_merge_error"].format(traceback.format_exc()))

    def open_training_wizard(self):
        wizard = TrainingWizard(self)
        if wizard.exec() != QDialog.DialogCode.Accepted:
            return
            
        frames = wizard.frame_list
        if not frames:
            return
        
        ds_dir = Path(QFileDialog.getExistingDirectory(
            self, "Onde guardar o projeto YOLO?"))
        if not ds_dir:
            return
            
        images_dir = ds_dir / "images"
        labels_dir = ds_dir / "labels"
        for d in (images_dir, labels_dir):
            d.mkdir(parents=True, exist_ok=True)

        progress = QProgressDialog("Copiando frames…", "Cancelar", 0, len(frames), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        QApplication.processEvents()
        
        self.dataset_frames = []
        for i, (src, num, video_name) in enumerate(frames):
            if progress.wasCanceled():
                break
                
            src_path = Path(src)
            frame_num = num if num is not None else i
            
            new_name = f"{video_name}_{frame_num:06d}{src_path.suffix}"
            dst = images_dir / new_name
            
            try:
                shutil.copy2(src, dst)
                self.dataset_frames.append((str(dst), frame_num, i))
            except Exception as e:
                QMessageBox.warning(self, "Erro de Cópia", 
                                f"Falha ao copiar:\n{src}\n→ {e}")
                continue
                
            progress.setValue(i + 1)
            QApplication.processEvents()
            
        progress.close()
        
        if self.dataset_frames:
            first_path = self.dataset_frames[0][0]
            self.start_video(first_path)
            self.current_frame_num = 0
            self.total_frames = len(self.dataset_frames)
            self.paused = True
            self.annotations = []
            self.detections_dock.all_detections = []
            self.detections_dock.apply_filters()
            self.dataset_mode = True
            self.dataset_index = 0

            QMessageBox.information(
                self, 
                self.texts["dataset_ready"],
                f"{self.texts['frames_loaded'].format(num_frames=len(self.dataset_frames))}\n"
                f"{self.texts['annotation_instruction']}"
            )

    def load_dataset_frame(self, index):
        if not (0 <= index < len(self.dataset_frames)):
            return

        self.dataset_index = index
        image_path, original_frame_num, dataset_index = self.dataset_frames[index]

        frame = cv2.imread(image_path)
        if frame is None:
            self.status_label.setText("Erro ao carregar imagem: " + str(image_path))
            return

        self.current_frame_num = index
        self.display_frame(frame)
        self.update_time_labels()
        self.video_name_label.setText(f"[Dataset] {Path(image_path).name}")

    def add_background_images_to_dataset(self, images_dir, train_indices: set, val_indices: set) -> int:
      
        #  Complete validation
        if not self.dataset_mode or not hasattr(self, 'dataset_frames') or not self.dataset_frames:
            return 0
        
        background_count = 0
        images_dir = Path(images_dir)
        
        # Set of all image paths in the dataset
        all_dataset_paths = {Path(path) for path, _, _ in self.dataset_frames}
        
        # Access only VALID INDEXES in the dataset
        annotated_paths = set()
        for idx in (train_indices | val_indices):
            if 0 <= idx < len(self.dataset_frames):  
                annotated_paths.add(Path(self.dataset_frames[idx][0]))
        
        # background
        unannotated = list(all_dataset_paths - annotated_paths)
        
        if not unannotated:
            return 0
        
        # Divides 80/20
        split_point = int(len(unannotated) * 0.8)
        train_bg = unannotated[:split_point]
        val_bg = unannotated[split_point:]
        
        # Copy with try/except for safety
        for img_path in train_bg:
            dst = images_dir / "train" / img_path.name
            if not dst.exists():
                try:
                    shutil.copy2(img_path, dst)
                    background_count += 1
                except Exception as e:
                    print(f"Erro ao copiar {img_path}: {e}")
        
        for img_path in val_bg:
            dst = images_dir / "val" / img_path.name
            if not dst.exists():
                try:
                    shutil.copy2(img_path, dst)
                    background_count += 1
                except Exception as e:
                    print(f"Erro ao copiar {img_path}: {e}")
        
        return background_count