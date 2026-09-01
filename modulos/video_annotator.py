"""
iSEA - Intelligent Seafloor & Animal Image Annotator
"""

import os
import csv
import yaml
import shutil
import hashlib
import platform
import getpass
import random
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from contextlib import contextmanager

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from PyQt6 import QtCore
from PyQt6.QtGui import (
    QPixmap, QImage, QIcon, QPainter, QPen, QAction,
    QKeySequence, QColor, QPalette, QDesktopServices, QCursor
)
from PyQt6.QtCore import Qt, QTimer, QRect, QSize, QUrl, QMutex, QMutexLocker
from PyQt6.QtWidgets import (
    QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QFileDialog, QMainWindow, QToolBar, QStyle, QMessageBox,
    QInputDialog, QSlider, QDockWidget, QDialog, QDialogButtonBox,
    QSizePolicy, QFrame, QSpinBox, QFormLayout, QComboBox,
    QDoubleSpinBox, QProgressDialog
)

from .video_label import VideoLabel
from .detections_dock import DetectionsDockWidget
from .train_thread import TrainThread, TrainSegmentationThread
from .translations import TEXTS
from .taxon_grid import TaxonGrid
from .detection_thread import DetectionThread
from .training_wizard import TrainingWizard
from .sam2_thread import SAM2Thread
from .taxonomy_enrichment import TaxonomyEnrichmentDialog
from .utils import resource_path
from .seafloor_classifier import (
    SeafloorClassifier,
    SeafloorClassificationThread,
    SeafloorClassificationDialog
)


# =============================================================================
# CONSTANTES
# =============================================================================

ACTIVE_BTN_STYLE = """
    QPushButton {
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid #5c9eff;
        background-color: #5c9eff;
        color: white;
        min-width: 23px;
        min-height: 23px;
    }
"""

INACTIVE_BTN_STYLE_TEMPLATE = """
    QPushButton {{
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid {border};
        background-color: {bg};
        color: {fg};
        min-width: 23px;
        min-height: 23px;
    }}
    QPushButton:hover {{
        background-color: {hover};
    }}
    QPushButton:pressed {{
        background-color: {pressed};
    }}
"""

LIVE_BTN_ACTIVE_STYLE = """
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
"""

SAM_BTN_ACTIVE_STYLE = """
    QPushButton {
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid #00c8ff;
        background-color: #00c8ff;
        color: white;
        min-width: 23px;
        min-height: 23px;
    }
"""

INFO_PANEL_STYLE = """
    QLabel {{
        background-color: {bg}33;
        color: {fg};
        border: 1px solid {border}66;
        border-radius: 4px;
        padding: 4px 12px;
        font-size: 13px;
        font-weight: bold;
        min-height: 28px;
    }}
"""

DEFAULT_INFO_COLORS = {"bg": "#E3F2FD", "fg": "#1565C0", "border": "#90CAF9"}

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".flv", ".wmv")
MODEL_EXTS = "Model files (*.pt *.onnx *.engine);;PyTorch (*.pt);;ONNX (*.onnx);;TensorRT (*.engine)"
CSV_EXTS = "CSV Files (*.csv);;All Files (*)"

SEAFLOOR_SHORTCUTS = {
    "S": "Sedimento",
    "F": "Coral_Fragmento",
    "R": "Recife_Coral",
}


class VideoAnnotator(QMainWindow):
    """Janela principal do iSEA Video Annotator."""

    # -------------------------------------------------------------------------
    # INICIALIZAÇÃO
    # -------------------------------------------------------------------------

    def __init__(self):
        super().__init__()

        self.language = "en"
        self.texts = TEXTS[self.language]

        self.setWindowTitle("iSEA")
        self.setWindowIcon(QIcon(resource_path("icons/iSEA_icon.png")))
        self.resize(1200, 700)

        # --- Estado do vídeo ---
        self.cap = None
        self.video_path = None
        self.paused = True
        self.current_frame_num = 0
        self.total_frames = 0
        self.current_frame = None
        self.live_mode = False
        self.live_start_time = None
        self.velocity = False
        self.camera_index = 0
        self.was_playing = False

        # --- Modelo e detecção ---
        self.model = None
        self.model_path = None
        self.custom_classes = []
        self.continuous_detection = False
        self.annotations = []
        self.all_detections = []
        self.detection_every_n_frames = 0
        self.frame_skip_counter = 0
        self.tracking_enabled = True
        self.track_colors = {}
        self.current_tracks = {}
        self.best_confidence = {}
        self._last_plotted_result = None  

        # --- Gravação ---
        self.recording = False
        self.video_writer = None
        self.recorded_frames = []
        self.record_start_frame = 0
        self.recorded_detections = []
        self.recording_filename = None
        self.recording_start_time = None

        # --- Dataset ---
        self.dataset_mode = False
        self.dataset_frames = []
        self.dataset_index = 0
        self.current_dataset_yaml = None

        # --- SAM 2 ---
        self.sam2_thread = None
        self.sam2_masks = {}
        self.current_sam2_mask = None
        self.hover_segmentation_mode = False
        self.sam2_use_preview = True
        self.sam2_cache_enabled = True
        self.sam2_min_quality = 0.7

        # --- Seafloor ---
        self.seafloor_classifier = None
        self.seafloor_thread = None
        self.seafloor_enabled = False
        self.seafloor_collecting = False
        self.current_seafloor_class = None
        self.current_seafloor_shortcut = None
        self.seafloor_annotation_frames = []
        self.seafloor_annotation_start_frame = None
        self.seafloor_frame_save_interval = 30
        self.seafloor_frame_counter = 0
        self.seafloor_training_dir = Path("seafloor_training_data")
        self.seafloor_realtime_counter = 0
        self.seafloor_realtime_skip = 5
        self.current_seafloor_result = None

        # --- Segmentação ---
        self.segmentation_annotations = []

        # --- UI ---
        self.drawing_color = QColor(Qt.GlobalColor.green)
        self.info_panel = None
        self.taxon_grid = None
        self.taxon_grid_dock = None
        self.detections_dock = None
        self.training_wizard = None
        self._seafloor_custom_actions = []
        self._seafloor_stop_action = None

        # --- Cache e otimização ---
        self.frame_cache = {}
        self.cache_size = 30
        self.use_fp16 = True

        # --- Modo headless (sem renderização de vídeo) ---
        self.headless_mode = False
        self.headless_stats = {
            "frames_processed": 0,
            "detections_found": 0,
            "start_time": None,
            "output_path": None,
            "all_detections": []
        }
        self.last_frame_hash = None
        self.last_frame_small = None

        # --- Threads ---
        self.detection_thread = DetectionThread(None)
        self.detection_thread.detection_finished.connect(self.on_detection_finished)
        self.detection_thread.start()
        self.pending_detection_result = None
        self.detection_mutex = QMutex()
        self.detection_frame_buffer = []

        # --- Inicialização ---
        self.init_ui()
        self.init_sam2()
        self.create_menu()
        self.apply_light_style()
        self.init_taxon_grid()
        self.load_model()

    def init_ui(self):
        """Inicializa todos os widgets da interface."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Labels
        self.video_label = VideoLabel(self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setText(self.texts["load_drag"])
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.status_label = QLabel(self.texts["waiting_action"])
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: gray")

        self.video_name_label = QLabel(self.texts["no_loaded"])
        self.video_name_label.setStyleSheet("color: gray")

        self.info_panel = QLabel("")
        self.info_panel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._apply_info_panel_style()
        self.info_panel.setVisible(False)

        # Botões
        self._create_toolbar_buttons()

        # Toolbar de playback
        self.tool_bar = QToolBar(self.texts.get("playback", "Playback"))
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

        # Dock de detecções
        self.detections_dock = DetectionsDockWidget(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.detections_dock)

        # Layout lateral de botões
        button_frame = QFrame()
        button_frame.setFixedWidth(50)
        button_layout = QVBoxLayout(button_frame)
        for btn in self._toolbar_buttons:
            button_layout.addWidget(btn)
        button_layout.addStretch()

        # Container de vídeo
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.addWidget(self.video_name_label)
        video_layout.addWidget(self.info_panel)
        video_layout.addWidget(self.video_label, 1)

        # Slider de progresso
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

        # Layout principal
        main_layout = QHBoxLayout(self.central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        main_layout.addWidget(video_container, 1)
        main_layout.addWidget(button_frame)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.setAcceptDrops(True)

    def _create_toolbar_buttons(self):
        """Cria e configura os botões da toolbar lateral."""
        buttons_config = [
            ("load", "load_video", "icons/load_icon.png", self.load_video),
            ("live", "live", None, self.toggle_live_mode, "🔴"),
            ("toggle", "toggle_detection", "icons/toggle_icon.png", self.toggle_detection),
            ("detect", "detect_frame", "icons/detect_icon.png", self.detect_objects),
            ("annotate", "annotate_manual", "icons/annotate_icon.png", self.enable_manual_annotation),
            ("sam", "segment_with_sam2", "icons/sam_icon.png", self.toggle_hover_segmentation),
            ("save", "save_annotations", "icons/save_icon.png", self.save_annotations),
            ("save_frame", "save_frame", "icons/save_frame_icon.png", self.save_current_frame_with_annotations),
            ("merge", "merge_annotations", "icons/merge_geo_icon.png", self.merge_annotations),
        ]

        self._toolbar_buttons = []
        for name, tooltip_key, icon_path, callback, *extra in buttons_config:
            btn = QPushButton(extra[0] if extra else "")
            btn.setToolTip(self.texts[tooltip_key])
            if icon_path:
                btn.setIcon(QIcon(resource_path(icon_path)))
            btn.setIconSize(QtCore.QSize(30, 30))
            btn.clicked.connect(callback)
            setattr(self, f"{name}_button", btn)
            self._toolbar_buttons.append(btn)

    def _apply_info_panel_style(self, color_hex=None):
        """Aplica estilo ao painel informativo. Texto sempre branco."""
        # Fallback: se cor inválida/branca, usa azul padrão
        if not color_hex or not isinstance(color_hex, str) or not color_hex.startswith("#"):
            color_hex = "#1565C0"
        c = color_hex.strip().upper()
        if c in ("#FFFFFF", "#FFF", "WHITE", ""):
            color_hex = "#1565C0"
        else:
            try:
                r = int(c[1:3], 16); g = int(c[3:5], 16); b = int(c[5:7], 16)
                if (0.299*r + 0.587*g + 0.114*b) / 255.0 > 0.82:
                    color_hex = "#1565C0"
            except Exception:
                color_hex = "#1565C0"
        self.info_panel.setStyleSheet(f"""
            QLabel {{
                background-color: {color_hex};
                color: #FFFFFF;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 13px;
                font-weight: bold;
                min-height: 28px;
            }}
        """)

    def init_taxon_grid(self):
        """Inicializa o grid de taxons."""
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


    # -------------------------------------------------------------------------
    # ESTILOS E UI
    # -------------------------------------------------------------------------

    def recolor_icon(self, standard_icon, color=QColor("white")):
        """Recolore um ícone padrão do sistema."""
        pixmap = self.style().standardIcon(standard_icon).pixmap(24, 24)
        painter = QPainter(pixmap)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(pixmap.rect(), color)
        painter.end()
        return QIcon(pixmap)

    def _is_dark_mode(self):
        """Retorna True se o tema escuro está ativo."""
        return self.palette().color(QPalette.ColorRole.Window).lightness() < 128

    def _get_button_style(self, active=False):
        """Retorna o estilo CSS para botões conforme tema."""
        if active:
            return ACTIVE_BTN_STYLE
        dark = self._is_dark_mode()
        colors = {
            "border": "#666" if dark else "#81D4FA",
            "bg": "#333" if dark else "#E1F5FE",
            "fg": "white" if dark else "black",
            "hover": "#555" if dark else "#81D4FA",
            "pressed": "#2a82da" if dark else "#29B6F6",
        }
        return INACTIVE_BTN_STYLE_TEMPLATE.format(**colors)

    def apply_light_style(self):
        """Aplica o tema claro à aplicação."""
        self.setStyleSheet("""
            QWidget {
                background-color: #FAFCFD;
                color: #37474F;
            }
            QMenuBar {
                background-color: #F5FAFC;
                color: #37474F;
                border-bottom: 1px solid #D6EAF8;
            }
            QMenu {
                background-color: #FFFFFF;
                color: #37474F;
                border: 1px solid #D6EAF8;
            }
            QMenu::item:selected {
                background-color: #E1F5FE;
                color: #0277BD;
            }
            QMenu::item:pressed {
                background-color: #B3E5FC;
            }
            QToolBar {
                background-color: #F0F9FF;
                border: none;
                padding: 2px;
            }
            QDockWidget {
                background-color: #FAFCFD;
                color: #37474F;
                border: 1px solid #D6EAF8;
            }
            QDockWidget::title {
                background-color: #E3F2FD;
                color: #37474F;
                padding: 4px;
            }
            QListWidget {
                background-color: #FFFFFF;
                color: #37474F;
                border: 1px solid #D6EAF8;
            }
            QListWidget::item:selected {
                background-color: #E1F5FE;
                color: #0277BD;
            }
            QListWidget::item:hover {
                background-color: #F5FAFC;
            }
            QGroupBox {
                color: #37474F;
                border: 1px solid #B3E5FC;
                border-radius: 4px;
                margin-top: 10px;
                font-weight: bold;
                background-color: transparent;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #0288D1;
            }
            QSlider::groove:horizontal {
                background: #E3F2FD;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #B3E5FC;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4FC3F7;
                width: 12px;
                margin: -3px 0;
                border-radius: 6px;
                border: 1px solid #29B6F6;
            }
            QSlider::handle:horizontal:hover {
                background: #29B6F6;
            }
            QLabel {
                color: #37474F;
                background-color: transparent;
            }
            QStatusBar {
                background-color: #F0F9FF;
                color: #37474F;
                border-top: 1px solid #D6EAF8;
            }
        """)

        for btn in self._toolbar_buttons:
            btn.setStyleSheet(self._get_button_style())

    def set_dark_mode(self, enable=True):
        """Alterna entre tema claro e escuro."""
        if enable:
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

            for btn in self._toolbar_buttons:
                btn.setStyleSheet(self._get_button_style())

            self.central_widget.setStyleSheet("")
            self.video_label.setStyleSheet("")
        else:
            self.apply_light_style()
            self._previous_action.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipBackward))
            self._play_action.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
            self._next_action.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipForward))

        if hasattr(self, 'detections_dock'):
            self.detections_dock.set_dark_mode(enable)

        if self.taxon_grid:
            self.taxon_grid.set_dark_mode(enable)

        self.update()

    # -------------------------------------------------------------------------
    # MENSAGENS E UTILITÁRIOS
    # -------------------------------------------------------------------------

    def set_status_message(self, key, *args):
        self.status_label.setText(self.texts[key].format(*args))

    def sync_frame_num(self, frame_num=None):
        """Sincroniza frame_num entre VideoAnnotator e VideoLabel."""
        if frame_num is not None:
            self.current_frame_num = frame_num
        if self.video_label is not None:
            self.video_label.current_frame_num = self.current_frame_num

    def show_error_message(self, title_key, message_key, *args):
        QMessageBox.critical(self, self.texts[title_key], self.texts[message_key].format(*args))

    def show_warning_message(self, title_key, message_key, *args):
        QMessageBox.warning(self, self.texts[title_key], self.texts[message_key].format(*args))

    def show_info_message(self, title_key, message_key, *args):
        QMessageBox.information(self, self.texts[title_key], self.texts[message_key].format(*args))

    def get_system_timestamp(self):
        """Retorna timestamp completo do sistema para anotações."""
        now = datetime.now()
        tz = now.astimezone().tzname()
        return {
            'system_datetime': now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            'system_date': now.strftime("%Y-%m-%d"),
            'system_time': now.strftime("%H:%M:%S.%f")[:-3],
            'system_timezone': tz,
            'computer_name': platform.node(),
            'user': getpass.getuser(),
            'os': f"{platform.system()} {platform.release()}"
        }

    def get_video_timestamp(self, frame_num):
        """Converte número do frame para timestamp HH:MM:SS."""
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

    def _update_info_panel(self, message, color_hex=None):
        """Atualiza o painel informativo acima do vídeo."""
        if not message:
            self.info_panel.setVisible(False)
            return
        self.info_panel.setText(message)
        self._apply_info_panel_style(color_hex)
        self.info_panel.setVisible(True)

    @contextmanager
    def wait_cursor(self, widget=None):
        """Context manager para cursor de espera durante operações longas."""
        target = widget or self.video_label
        target.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        try:
            yield
        finally:
            target.restoreOverrideCursor()


    # -------------------------------------------------------------------------
    # MENU
    # -------------------------------------------------------------------------

    def create_menu(self):
        """Cria a barra de menus completa."""
        menubar = self.menuBar()
        self._create_file_menu(menubar)
        self._create_view_menu(menubar)
        self._create_annotation_menu(menubar)
        self._create_training_menu(menubar)
        self._create_seafloor_menu(menubar)
        self._create_language_menu(menubar)
        self._create_help_menu(menubar)

    def _create_file_menu(self, menubar):
        file_menu = menubar.addMenu(self.texts["arquivo"])
        actions = [
            (self.texts["load_video"], "Ctrl+O", self.load_video),
            (self.texts["load_image_folder"], None, self.load_image_folder_for_inference),
            (self.texts["load_model"], "Ctrl+M", self.load_custom_model),
            (self.texts["unload_model"], None, self.unload_model),
            (self.texts["load_yaml"], "Ctrl+Y", lambda: self.load_dataset_taxons()),
            (self.texts["load_annotations"], "Ctrl+L", self.load_annotations_dialog),
            (self.texts["save_annotations"], "Ctrl+S", self.save_annotations),
            (self.texts["export_batch_results"], None, self.export_batch_results),
            (self.texts["start_recording"], "Ctrl+R", self.start_recording),
            (self.texts["stop_recording"], "Ctrl+Shift+R", self.stop_recording),
            (self.texts["live"], "Ctrl+W", self.toggle_live_mode),
            (self.texts["process_video_headless"], "Ctrl+Shift+H", self.run_headless_video_processing),
        ]
        for text, shortcut, callback in actions:
            act = QAction(text, self)
            if shortcut:
                act.setShortcut(QKeySequence(shortcut))
            act.triggered.connect(callback)
            file_menu.addAction(act)

        file_menu.addSeparator()
        exit_act = QAction(self.texts["exit"], self)
        exit_act.setShortcut(QKeySequence("Ctrl+Q"))
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

    def _create_view_menu(self, menubar):
        view_menu = menubar.addMenu(self.texts["visualization"])
        toggle_history = QAction(self.texts["history_show"], self)
        toggle_history.setShortcut(QKeySequence("Ctrl+H"))
        toggle_history.triggered.connect(self.toggle_detections_history)
        view_menu.addAction(toggle_history)

        toggle_taxon = QAction(self.texts["taxon_show"], self)
        toggle_taxon.setShortcut(QKeySequence("Ctrl+T"))
        toggle_taxon.triggered.connect(self.toggle_taxon_grid)
        view_menu.addAction(toggle_taxon)

        dark_mode = QAction(self.texts["dark_mode"], self)
        dark_mode.setCheckable(True)
        dark_mode.triggered.connect(self.set_dark_mode)
        view_menu.addAction(dark_mode)

        view_menu.addSeparator()
        headless_action = QAction(self.texts.get("headless_mode", "Modo rápido (sem vídeo)"), self)
        headless_action.setCheckable(True)
        headless_action.setToolTip(self.texts.get("headless_mode_tooltip", "Processa frames sem renderizar — ideal para batch e benchmark"))
        headless_action.triggered.connect(self.toggle_headless_mode)
        view_menu.addAction(headless_action)

    def _create_annotation_menu(self, menubar):
        annotation_menu = menubar.addMenu(self.texts["annotation"])
        actions = [
            (self.texts["detect_frame"], "D", self.detect_objects),
            (self.texts["toggle_detection"], "T", self.toggle_detection),
            (self.texts["annotate_manual"], "M", self.enable_manual_annotation),
            (self.texts["segment_with_sam2"], "A", self.toggle_hover_segmentation),
            (self.texts["enrich_taxonomy"], "E", self.open_enrichment_dialog),
        ]
        for text, shortcut, callback in actions:
            act = QAction(text, self)
            act.setShortcut(QKeySequence(shortcut))
            act.triggered.connect(callback)
            annotation_menu.addAction(act)

    def _create_training_menu(self, menubar):
        training_menu = menubar.addMenu(self.texts["train"])
        actions = [
            (self.texts["create_dataset"], None, self.open_training_wizard),
            (self.texts.get("import_dataset", "Import Dataset"), None, self.import_yolo_dataset),
            (self.texts["export_yolo"], None, self.export_yolo_annotations_dialog),
            (self.texts["train_yolo"], None, self.train_yolo_model),
            (self.texts["export_segmentation"], None, self.export_segmentation_dialog),
            (self.texts["train_segmentation_model"], None, self.train_segmentation_model),
        ]
        for text, shortcut, callback in actions:
            act = QAction(text, self)
            if shortcut:
                act.setShortcut(QKeySequence(shortcut))
            act.triggered.connect(callback)
            training_menu.addAction(act)

    def _create_seafloor_menu(self, menubar):
        seafloor_menu = menubar.addMenu(self.texts.get("seafloor_menu", "Seafloor"))

        classify_folder = QAction(self.texts.get("seafloor_classify_folder", "Classificar Pasta"), self)
        classify_folder.triggered.connect(self.open_seafloor_dialog)
        seafloor_menu.addAction(classify_folder)

        classify_frame = QAction(self.texts.get("seafloor_classify_frame", "Classificar Frame Atual"), self)
        classify_frame.setShortcut(QKeySequence("Ctrl+F"))
        classify_frame.triggered.connect(self.classify_current_frame)
        seafloor_menu.addAction(classify_frame)

        manage = QAction(self.texts.get("manage_categories", "Manage Categories..."), self)
        manage.triggered.connect(self.open_seafloor_class_manager)
        seafloor_menu.addAction(manage)

        seafloor_menu.addSeparator()
        seafloor_menu.addSection(self.texts.get("quick_classification", "Quick Classification (during video)"))

        # Classes fixas: S, F, R
        for key, class_name in SEAFLOOR_SHORTCUTS.items():
            act = QAction(f"{key} - {class_name}", self)
            act.setShortcut(QKeySequence(key))
            act.triggered.connect(lambda checked, c=class_name, k=key: self.quick_classify_seafloor(c, k))
            seafloor_menu.addAction(act)

        # Classes customizadas (atualizadas dinamicamente)
        self._seafloor_custom_actions = []
        self._update_seafloor_custom_menu(seafloor_menu)

        seafloor_menu.addSeparator()
        self._seafloor_stop_action = QAction(self.texts.get("stop_annotation", "Stop Annotation (Shift+S)"), self)
        self._seafloor_stop_action.setShortcut(QKeySequence("Shift+S"))
        self._seafloor_stop_action.triggered.connect(self.stop_seafloor_annotation)
        seafloor_menu.addAction(self._seafloor_stop_action)

        seafloor_menu.addSeparator()
        train_act = QAction(self.texts.get("train_from_collected", "Train from Collected Data"), self)
        train_act.triggered.connect(self.train_seafloor_from_collected)
        seafloor_menu.addAction(train_act)

        toggle_rt = QAction(self.texts.get("seafloor_realtime", "Real-time Classification"), self)
        toggle_rt.setCheckable(True)
        toggle_rt.triggered.connect(self.toggle_seafloor_realtime)
        seafloor_menu.addAction(toggle_rt)

    def _create_language_menu(self, menubar):
        lang_menu = menubar.addMenu(self.texts["language"])
        for code, label in [("pt", self.texts["portuguese"]), ("en", self.texts["english"])]:
            act = QAction(label, self)
            act.triggered.connect(lambda checked, c=code: self.change_language(c))
            lang_menu.addAction(act)

    def _create_help_menu(self, menubar):
        help_menu = menubar.addMenu(self.texts["help"])
        shortcuts = QAction(self.texts["shortcuts"], self)
        shortcuts.setShortcut(QKeySequence("F1"))
        shortcuts.triggered.connect(self.show_shortcuts)
        help_menu.addAction(shortcuts)

        about = QAction(self.texts["about"], self)
        about.triggered.connect(self.show_about)
        help_menu.addAction(about)

        manual = QAction(self.texts.get("manual_menu", "Manual"), self)
        manual.triggered.connect(self.show_manual)
        help_menu.addAction(manual)

    def toggle_detections_history(self):
        self.detections_dock.setVisible(not self.detections_dock.isVisible())

    def toggle_taxon_grid(self):
        self.taxon_grid_dock.setVisible(not self.taxon_grid_dock.isVisible())


    # -------------------------------------------------------------------------
    # MODELO E CARREGAMENTO
    # -------------------------------------------------------------------------

    def load_custom_model(self):
        path, _ = QFileDialog.getOpenFileName(self, self.texts["select_model"], "", MODEL_EXTS)
        if path:
            self.load_model(path)

    def load_model(self, model_path=None):
        """Carrega modelo YOLO (padrão ou customizado)."""
        try:
            if model_path is None:
                self.model_path = r"modelos\corais.pt"
                self.model = YOLO(resource_path(self.model_path))
            else:
                if os.path.exists(model_path):
                    self.model = YOLO(resource_path(model_path))
                    self.model_path = os.path.basename(model_path)
                else:
                    raise FileNotFoundError(self.texts["model_not_found"].format(model_path))

            if self.detection_thread:
                self.detection_thread.set_model(self.model)

            self.status_label.setText(self.texts["model_loaded"].format(self.model_path))

            if self.detections_dock:
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

    def refresh_taxon_grid(self):
        """Atualiza o grid de taxons com classes do modelo e customizadas."""
        if self.taxon_grid is None:
            return
        self.taxon_grid.clear()
        new_classes = []
        if self.model:
            new_classes = list(self.model.names.values())
        new_classes += [c for c in self.custom_classes if c not in new_classes]
        self.taxon_grid.populate(new_classes)
        self.taxon_grid.set_dark_mode(self._is_dark_mode())

    def load_dataset_taxons(self, yaml_path=None):
        """Carrega classes de um arquivo dataset.yaml."""
        try:
            if yaml_path is None:
                yaml_path, _ = QFileDialog.getOpenFileName(
                    self, "Select dataset.yaml", "", "YAML Files (*.yaml *.yml);;All Files (*)")
                if not yaml_path:
                    return

            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            names = data.get('names', {})
            if isinstance(names, dict):
                class_names = [names[i] for i in sorted(names.keys())]
            elif isinstance(names, list):
                class_names = names
            else:
                raise ValueError("Invalid 'names' format in YAML")

            if self.taxon_grid:
                self.taxon_grid.clear()
                for class_name in class_names:
                    self.taxon_grid.add_taxon(class_name)

            self.current_dataset_yaml = yaml_path
            self.status_label.setText(self.texts.get("taxons_loaded_format", "Taxons loaded: {} classes from {}").format(len(class_names), os.path.basename(yaml_path)))
            QMessageBox.information(self, self.texts.get("success_title", "Success"), self.texts.get("dataset_import_success", "Imported {} images\nClasses: {}\nAnnotations: {}").format(len(class_names), '', ''))

        except FileNotFoundError:
            QMessageBox.critical(self, self.texts.get("error_title", "Error"), self.texts.get("file_not_found_format", "File not found: {}").format(yaml_path))
        except yaml.YAMLError as e:
            QMessageBox.critical(self, self.texts.get("error_title", "Error"), self.texts.get("invalid_yaml_format", "Invalid YAML file: {}").format(str(e)))
        except Exception as e:
            QMessageBox.critical(self, self.texts.get("error_title", "Error"), self.texts.get("error_loading_taxons_format", "Error loading taxons:{}").format(str(e)))

    def change_drawing_class(self, name):
        """Altera a classe ativa para anotação manual."""
        self.video_label.current_class = name

        # --- CORES FIXAS DO SEAFLOOR (pt + en) ---
        seafloor_colors = {
            "Sedimento": "#8B4513", "Sediment": "#8B4513",
            "Coral_Fragmento": "#FF8C00", "Coral Fragment": "#FF8C00",
            "Recife_Coral": "#00CED1", "Coral Reef": "#00CED1",
        }

        if name in seafloor_colors:
            self.video_label.drawing_color = QColor(seafloor_colors[name])
        else:
            # HSV controlado: saturação alta, valor moderado → NUNCA branco
            hue = abs(hash(name)) % 360
            sat = 180 + (abs(hash(name + "_sat")) % 75)   # 180–255
            val = 140 + (abs(hash(name + "_val")) % 80)   # 140–220
            self.video_label.drawing_color = QColor.fromHsv(hue, sat, val)

        if name not in self.custom_classes and (not self.model or name not in self.model.names.values()):
            self.custom_classes.append(name)

        if self.detections_dock:
            if self.detections_dock.class_filter.findText(name) == -1:
                self.detections_dock.class_filter.addItem(name)

    def change_language(self, lang):
        """Altera o idioma da interface."""
        self.language = lang
        self.texts = TEXTS[lang]

        self.video_name_label.setText(self.texts["no_loaded"])
        self.video_label.setText(self.texts["load_drag"])

        if self.model_path:
            self.status_label.setText(self.texts["model_loaded"].format(self.model_path))
        else:
            self.status_label.setText(self.texts["waiting_action"])

        for btn in self._toolbar_buttons:
            # Atualizar tooltips conforme o nome do botão
            pass  # Tooltips são definidos no create, precisam ser redefinidos

        # Reconstruir menu
        self.menuBar().clear()
        self.create_menu()

        if self.detections_dock:
            self.detections_dock.update_language(lang)
        if self.taxon_grid:
            self.taxon_grid.update_language(lang)
        if self.training_wizard:
            self.training_wizard.update_language(lang)

        self.detections_dock.show()

    # -------------------------------------------------------------------------
    # CARREGAMENTO DE VÍDEO
    # -------------------------------------------------------------------------

    def load_video(self):
        """Abre diálogo para selecionar vídeo."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, self.texts["select_video"], "",
            self.texts.get("video_files_filter", "Videos") + " (*.mp4 *.avi *.mov *.mkv *.m4v *.flv *.wmv);;" +
            self.texts.get("all_files", "All Files") + " (*)")

        if not file_path:
            return

        # Reset estado se estava em modo dataset
        if self.dataset_mode:
            self._reset_dataset_state()

        if self.live_mode:
            self.live_mode = False

        self.video_label.current_frame_num = 0
        self.current_frame_num = 0
        self.sync_frame_num()
        self.start_video(file_path)

    def _reset_dataset_state(self):
        """Reseta o estado quando saindo do modo dataset."""
        self.detections_dock.detections_list.clear()
        self.detections_dock.all_detections.clear()
        self.dataset_mode = False
        self.video_label.frame_annotations = {}
        self.video_label.active_annotations = []
        self.video_label.segmentation_annotations = []
        self.all_detections = []
        self.annotations = []

    def start_video(self, file_path):
        """Inicializa reprodução de vídeo a partir do arquivo."""
        if self.seafloor_enabled:
            self.toggle_seafloor_realtime(False)

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
        self.sync_frame_num()
        self.paused = True

        self._update_play_icon()

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            total_seconds = self.total_frames / fps
            self.total_video_time = self._format_time(total_seconds)
        else:
            self.total_video_time = "00:00:00"

        self.total_time_label.setText(self.total_video_time)

        ret, frame = self.cap.read()
        if ret:
            self.current_frame_num = 1
            self.sync_frame_num()
            self.display_frame(frame)
            self.update_time_labels()
            self.status_label.setText(self.texts["video_loaded"])
        else:
            self.status_label.setText(self.texts["video_first_frame_error"])

    def _format_time(self, total_seconds):
        """Formata segundos totais para HH:MM:SS."""
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _update_play_icon(self):
        """Atualiza o ícone do botão play/pause conforme tema."""
        if self._is_dark_mode():
            self._play_action.setIcon(self.recolor_icon(QStyle.StandardPixmap.SP_MediaPlay))
        else:
            self._play_action.setIcon(self.play_icon)
        self._play_action.setText(self.texts["play"])

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(VIDEO_EXTS):
                self.start_video(file_path)
                break


    # -------------------------------------------------------------------------
    # PLAYBACK
    # -------------------------------------------------------------------------

    def toggle_play_pause(self):
        if self.cap is None:
            return

        self.paused = not self.paused

        # Pausar/resumir seafloor realtime
        if self.seafloor_thread:
            self.seafloor_thread._paused = self.paused

        if self.paused:
            self._update_play_icon()
            self.timer.stop()
        else:
            if self._is_dark_mode():
                self._play_action.setIcon(self.recolor_icon(QStyle.StandardPixmap.SP_MediaPause))
            else:
                self._play_action.setIcon(self.pause_icon)
            self._play_action.setText(self.texts["pause"])

            fps = self.cap.get(cv2.CAP_PROP_FPS)
            multiplier = 2 if self.velocity else 1
            interval = int(1000 / (fps * multiplier)) if fps > 0 else (16 if self.velocity else 30)
            self.timer.start(interval)
            self.video_label.reset_annotations()

        self.video_label.update()

    def previous_frame(self):
        if self.dataset_mode and self.dataset_frames:
            idx = self.dataset_index - 1
            if idx >= 0:
                self.load_dataset_frame(idx)
                self.on_dataset_frame_changed(idx)
            return

        if self.cap is None:
            return

        new_pos = max(self.current_frame_num - 30, 0)
        self._seek_and_display(new_pos)

    def next_frame(self):
        if self.dataset_mode and self.dataset_frames:
            idx = self.dataset_index + 1
            if idx < len(self.dataset_frames):
                self.load_dataset_frame(idx)
                self.on_dataset_frame_changed(idx)
            return

        if self.cap is None:
            return

        new_pos = min(self.current_frame_num + 30, self.total_frames - 1)
        self._seek_and_display(new_pos)

    def _seek_and_display(self, frame_num):
        """Seek para frame e exibe."""
        self.set_current_frame(frame_num)
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
            self.update_progress_slider()
            self.update_time_labels()

    def set_current_frame(self, frame_num):
        if self.cap is None or self.live_mode:
            return
        self.sync_frame_num(frame_num)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        self.video_label.current_frame_num = frame_num
        self.video_label.update_active_annotations()
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
            self.update_progress_slider()
            self.update_time_labels()

    def seek_video(self, value):
        if self.live_mode or self.cap is None:
            return
        frame_pos = int((value / 100) * self.total_frames)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        self.sync_frame_num(frame_pos)
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

    def update_progress_slider(self):
        if self.total_frames > 0:
            progress = int((self.current_frame_num / self.total_frames) * 100)
            self.progress_slider.setValue(progress)

    def update_time_labels(self):
        if self.cap is None:
            return
        if self.live_mode:
            self.current_time_label.setText(self.texts.get("live_status", "Live"))
            self.total_time_label.setText("")
            return

        current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.current_time_label.setText(self.get_video_timestamp(current_frame))
        if hasattr(self, 'total_video_time'):
            self.total_time_label.setText(self.total_video_time)

    def velocity2(self):
        if self.cap is None:
            return
        self.velocity = not self.velocity
        self.timer.stop()

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.velocity:
            self.detection_every_n_frames = 2
            self.frame_skip_counter = 0
            interval = int(1000 / (fps * 2)) if fps > 0 else 16
            self.velocity2_button.setStyleSheet("background-color: #5c9eff;")
            self.set_status_message("speed_detection_format", "2.0", self.detection_every_n_frames)
        else:
            self.detection_every_n_frames = 0
            self.last_frame_hash = None
            self.last_frame_small = None
            interval = int(1000 / fps) if fps > 0 else 30
            self.velocity2_button.setStyleSheet("background-color: None")
            self.set_status_message("speed_format", "1.0", fps)

        self.timer.start(interval)

    # -------------------------------------------------------------------------
    # FRAME DISPLAY
    # -------------------------------------------------------------------------

    def display_frame(self, frame):
        """Exibe um frame (numpy array ou QPixmap) no video_label."""
        try:
            if isinstance(frame, np.ndarray):
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                h, w = frame.shape[:2]
                self.video_label._aspect_ratio = w / h
                bytes_per_line = 3 * w
                q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
            else:
                pixmap = frame

            # >>> CORREÇÃO: não limpar anotações no modo dataset <<<
            if not getattr(self, 'dataset_mode', False):
                self.video_label.active_annotations = []
            self.video_label.update()
            self.video_label._pixmap = pixmap
            self.update_video_display()

            if self.paused:
                self.video_label.update()

        except Exception as e:
            self.set_status_message("frame_error", str(e))

    def update_video_display(self):
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
            self.video_label.video_x, self.video_label.video_y,
            scaled_pixmap.width(), scaled_pixmap.height()
        )
        self.video_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_video_display()

    # -------------------------------------------------------------------------
    # DETECÇÃO UNIFICADA
    # -------------------------------------------------------------------------

    def _build_detection_dict(self, box, frame, frame_num, detection_type="auto"):
        """Constrói o dicionário de detecção padronizado."""
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = self.model.names[cls_id]
        track_id = int(box.id) if box.id is not None else None
        system_info = self.get_system_timestamp()

        return {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "label": label,
            "confidence": conf,
            "type": detection_type,
            "class": label,
            "timestamp": self.get_video_timestamp(frame_num),
            "system_datetime": system_info['system_datetime'],
            "system_date": system_info['system_date'],
            "system_time": system_info['system_time'],
            "system_timezone": system_info['system_timezone'],
            "computer_name": system_info['computer_name'],
            "user": system_info['user'],
            "os": system_info['os'],
            "track_id": track_id,
            "frame_number": frame_num,
            "video_path": self.video_path or "Live",
            "frame_dimensions": f"{frame.shape[1]}x{frame.shape[0]}",
            "frame_source": (self.video_path or "Live", frame_num)
        }

    def detect_objects(self):
        """Executa detecção no frame atual (modo manual)."""
        if self.cap is None or not self.cap.isOpened():
            self.status_label.setText(self.texts["no_loaded"])
            return

        self._pause_for_operation()

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
            self._run_detection_on_frame(frame)
            self.set_status_message("detection_completed", int(current_pos))
        except Exception as e:
            self.set_status_message("detection_error", str(e))

    def _run_detection_on_frame(self, frame):
        """Executa inferência YOLO em um frame e processa resultados."""
        if self.model is None:
            return False

        self.annotations = []
        frame_copy = np.ascontiguousarray(frame)

        with torch.no_grad():
            results = self.model.track(frame_copy, conf=0.5, iou=0.5, persist=True, verbose=False)

        if len(results) > 0 and results[0].boxes:
            plotted_frame = results[0].plot()
            for box in results[0].boxes:
                detection = self._build_detection_dict(box, frame, self.current_frame_num)
                detection["frame"] = self.capture_current_frame()
                self.annotations.append(detection)
                self.detections_dock.add_detection(detection)
                if self.recording:
                    self.recorded_detections.append(detection)

            self.display_frame(plotted_frame)
            return True
        return False

    def on_detection_finished(self, results, used_frame, frame_num):
        """Callback quando a thread de detecção assíncrona termina."""
        if results is None or not results.boxes:
            return

        try:
            for box in results.boxes:
                detection = self._build_detection_dict(box, used_frame, frame_num)
                self.annotations.append(detection)
                self.detections_dock.add_detection(detection)
                if self.recording:
                    self.recorded_detections.append(detection)

            # NÃO chama display_frame aqui — o frame já foi processado sincronamente
            self.set_status_message("detection_completed", frame_num)

        except Exception:
            import traceback
            traceback.print_exc()

    def toggle_detection(self):
        """Ativa/desativa detecção contínua."""
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
            self.toggle_button.setStyleSheet(ACTIVE_BTN_STYLE)
            self.set_status_message("continuous_detection_on")
            if self.paused:
                self.toggle_play_pause()
        else:
            self.toggle_button.setStyleSheet(self._get_button_style())
            self.set_status_message("continuous_detection_off")

    def enable_manual_annotation(self):
        """Ativa/desativa modo de anotação manual (bounding boxes)."""
        if self.hover_segmentation_mode:
            self.toggle_hover_segmentation()

        if not self.live_mode:
            self._pause_for_operation()

        self.video_label.drawing_enabled = not self.video_label.drawing_enabled

        if self.video_label.drawing_enabled:
            self.video_label.setCursor(self.video_label.cursor_drawing)
            if not self.video_label.current_class:
                first = next(iter(self.taxon_grid._buttons.keys()), None)
                if first:
                    self.taxon_grid.select(first)
                else:
                    QMessageBox.warning(self, self.texts["warning"], self.texts["no_classes_loaded"])
                    self.video_label.drawing_enabled = False
                    return
            self.annotate_button.setStyleSheet(ACTIVE_BTN_STYLE)
            self.set_status_message("manual_annotation_on")
        else:
            self.annotate_button.setStyleSheet(self._get_button_style())
            self.set_status_message("manual_annotation_off")

        self.video_label.update()

    def _pause_for_operation(self):
        """Pausa o vídeo para operações que requerem frame estático."""
        self.paused = True
        self._update_play_icon()
        self.timer.stop()


    # -------------------------------------------------------------------------
    # LOOP PRINCIPAL DE FRAME (update_frame)
    # -------------------------------------------------------------------------

    def update_frame(self):
        """Loop principal de atualização de frames (chamado pelo timer)."""
        if self.paused or self.cap is None or not self.cap.isOpened():
            return

        try:
            ret, frame = self.cap.read()
            if not ret:
                self._handle_frame_read_failure()
                return

            self.current_frame = frame.copy()
            self._handle_seafloor_collection()
            self._handle_seafloor_realtime(frame)
            self._update_frame_number()

            # === DETECÇÃO CONTÍNUA ===
            if self.continuous_detection and self.model:
                frame = self._process_continuous_detection(frame)

            # === GRAVAÇÃO ===
            if self.recording and self.video_writer is not None:
                self.video_writer.write(frame)

            # === TIMESTAMP LIVE ===
            if self.live_mode:
                frame = self._add_live_timestamp(frame)

            # === DISPLAY ===
            if self.headless_mode:
                self.headless_stats["frames_processed"] += 1
                if self.annotations:
                    self.headless_stats["all_detections"].extend(self.annotations)
                    self.headless_stats["detections_found"] += len(self.annotations)

                if self.headless_stats["frames_processed"] % 30 == 0:
                    elapsed = (datetime.now() - self.headless_stats["start_time"]).total_seconds()
                    fps = self.headless_stats["frames_processed"] / elapsed if elapsed > 0 else 0
                    self.status_label.setText(
                        f"⚡ {self.texts.get('headless_status', 'Headless')} | "
                        f"Frames: {self.headless_stats['frames_processed']} | "
                        f"FPS: {fps:.1f} | "
                        f"Detecções: {self.headless_stats['detections_found']}"
                    )
            else:
                self._display_frame_resized(frame)
                self.update_time_labels()

        except Exception as e:
            self.set_status_message("fatal_error")
            self.paused = True
            self.timer.stop()
            self.show_error_message("error", "fatal_error_detail", str(e))

    def _handle_frame_read_failure(self):
        """Lida com falha na leitura do frame (fim do vídeo ou erro)."""
        if self.headless_mode and self.headless_stats.get("output_path"):
            self._save_headless_results()
            self.headless_mode = False
            self.toggle_headless_mode(False)
            self.continuous_detection = False
            self.toggle_button.setStyleSheet(self._get_button_style())
            self.paused = True
            self._update_play_icon()
            self.timer.stop()
            return
        if self.seafloor_enabled:
            self.toggle_seafloor_realtime(False)
        if self.live_mode:
            self.start_camera()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame_num = 0
            self.sync_frame_num()
            self.paused = True
            self._update_play_icon()
    def _handle_seafloor_collection(self):
        """Coleta frames para treinamento de seafloor durante playback."""
        if self.seafloor_collecting and self.current_seafloor_class:
            self.seafloor_frame_counter += 1
            if self.seafloor_frame_counter % self.seafloor_frame_save_interval == 0:
                self._save_seafloor_training_frame()

    def _handle_seafloor_realtime(self, frame):
        """Envia frame para classificação de seafloor em tempo real."""
        if not (self.seafloor_enabled and self.seafloor_thread):
            return
        self.seafloor_realtime_counter += 1
        if self.seafloor_realtime_counter % self.seafloor_realtime_skip == 0:
            if hasattr(self.seafloor_thread, 'add_frame'):
                self.seafloor_thread.add_frame(frame.copy(), self.current_frame_num)

    def _update_frame_number(self):
        """Atualiza o número do frame atual e timestamp."""
        if self.live_mode:
            self.current_frame_num += 1
            if not hasattr(self, 'live_start_time'):
                self.live_start_time = datetime.now()
        else:
            self.current_frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.sync_frame_num()
            self.update_progress_slider()

    def _process_continuous_detection(self, frame):
        """Processa detecção contínua no frame atual."""
        if self.model is None:
            return frame

        # ── Skip de frames quando velocity está ativo ──
        self.frame_skip_counter += 1
        skip = max(1, self.detection_every_n_frames) if self.velocity else 1
        should_run_model = (self.frame_skip_counter % skip == 0)

        frame_copy = np.ascontiguousarray(frame.copy())

        # Deduplicação por similaridade (só em 2×)
        if self.velocity and should_run_model:
            frame_hash, frame_small = self.frame_to_small_and_hash(frame_copy)
            if (self.last_frame_hash is not None and 
                self.similar_frames(self.last_frame_small, frame_small)):
                should_run_model = False   # frame muito parecido, não roda modelo
            else:
                self.last_frame_hash = frame_hash
                self.last_frame_small = frame_small

        # Se não vai rodar modelo, reutiliza o último frame plotado (evita piscar)
        if not should_run_model:
            if self._last_plotted_result is not None:
                
                return self._last_plotted_result.copy()
            return frame

        # ── Roda detecção ──
        with torch.no_grad():
            results = self.model.track(
                frame_copy,
                persist=True,
                verbose=False,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                tracker="botsort.yaml",
                conf=0.55,
                iou=0.65,
            )

        if results and len(results) > 0 and results[0].boxes:
            frame = results[0].plot(img=frame)
            self._last_plotted_result = frame.copy()   # guarda para reusar nos skips
            for box in results[0].boxes:
                detection = self._build_detection_dict(box, frame, self.current_frame_num)
                self.annotations.append(detection)
                self.detections_dock.add_detection(detection)
                if self.recording:
                    self.recorded_detections.append(detection)
        else:
            # Nenhuma detecção neste frame → limpa cache visual
            self._last_plotted_result = None

        return frame

    def _add_live_timestamp(self, frame):
        """Adiciona timestamp ao frame no modo live."""
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale, thickness = 0.7, 2
        text_color, bg_color = (255, 255, 255), (0, 0, 0)

        (text_width, text_height), _ = cv2.getTextSize(timestamp, font, font_scale, thickness)
        margin = 10
        text_x = frame.shape[1] - text_width - margin
        text_y = frame.shape[0] - margin

        cv2.rectangle(frame,
                      (text_x - margin, text_y - text_height - margin),
                      (text_x + text_width + margin, text_y + margin),
                      bg_color, -1)
        cv2.putText(frame, timestamp, (text_x, text_y), font, font_scale, text_color, thickness)
        return frame

    def _display_frame_resized(self, frame):
        """Redimensiona e exibe o frame na interface."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        target_w, target_h = self.video_label.width(), self.video_label.height()

        if w / h > target_w / target_h:
            new_w, new_h = target_w, int(h * target_w / w)
        else:
            new_h, new_w = target_h, int(w * target_h / h)

        frame_resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        bytes_per_line = 3 * new_w
        q_img = QImage(frame_resized.data, new_w, new_h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap)

    # -------------------------------------------------------------------------
    # UTILITÁRIOS DE FRAME
    # -------------------------------------------------------------------------

    def capture_current_frame(self):
        """Captura o frame atual como QImage."""
        if self.cap is None or not self.cap.isOpened():
            return None

        current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)

        if not ret:
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return q_img.copy()

    def frame_to_small_and_hash(self, frame: np.ndarray) -> tuple:
        """Gera hash e versão reduzida do frame para comparação."""
        small = cv2.resize(frame, (64, 64))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        h = hashlib.blake2b(gray.tobytes(), digest_size=8).hexdigest()
        return h, small

    def similar_frames(self, frame1: np.ndarray, frame2: np.ndarray, threshold=4):
        """Verifica se dois frames são visualmente similares."""
        if frame1 is None or frame2 is None:
            return False
        diff = cv2.absdiff(frame1, frame2)
        return np.mean(diff) < threshold


    # -------------------------------------------------------------------------
    # LIVE MODE
    # -------------------------------------------------------------------------

    def toggle_live_mode(self):
        if self.seafloor_enabled:
            self.toggle_seafloor_realtime(False)

        if self.live_mode:
            self._stop_live_mode()
        else:
            self._start_live_mode()

    def _stop_live_mode(self):
        self.stop_recording()
        self.live_mode = False
        self.live_button.setStyleSheet(self._get_button_style())
        self.status_label.setText(self.texts["webcam_mode_off"])
        if self.cap is not None:
            self.cap.release()
        self.timer.stop()

    def _start_live_mode(self):
        self.live_mode = True
        self.live_button.setStyleSheet(LIVE_BTN_ACTIVE_STYLE)

        cameras = self.list_available_cameras()
        if len(cameras) > 1:
            camera, ok = QInputDialog.getItem(
                self, self.texts["select_camera"],
                self.texts["choose_camera"], cameras, 0, False)
            if ok:
                self.camera_index = cameras.index(camera)

        self.start_camera()
        self.status_label.setText(self.texts["webcam_mode_on"].format(self.camera_index))
        if self.paused:
            self.toggle_play_pause()

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

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.video_name_label.setText(self.texts["webcam"].format(self.camera_index))
            self.total_frames = 0
            self.current_frame_num = 0
            self.sync_frame_num()
            self.paused = False
            self._play_action.setIcon(self.pause_icon)
            self._play_action.setText(self.texts["pause"])
            self.current_time_label.setText("00:00:00")
            self.total_time_label.setText(self.texts["live_text"])
            self.annotations = []
            self.video_label.manual_annotations = []
            if self.detections_dock:
                self.detections_dock.all_detections = []
                self.detections_dock.apply_filters()
            self.timer.start(30)

        except Exception as e:
            self.status_label.setText(self.texts["camera_error"].format(str(e)))
            self.live_mode = False
            self.live_button.setStyleSheet(self._get_button_style())
            if self.cap is not None:
                self.cap.release()
                self.cap = None

    # -------------------------------------------------------------------------
    # GRAVAÇÃO
    # -------------------------------------------------------------------------

    def start_recording(self):
        if not self.live_mode or self.cap is None or not self.cap.isOpened():
            return

        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 30:
            fps = 20

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_filename = f"recording_{timestamp}.avi"
        self.current_frame_num = 0
        self.sync_frame_num()
        self.live_start_time = datetime.now()
        self.recording_start_time = datetime.now()

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(
            self.recording_filename, fourcc, fps, (frame_width, frame_height))

        if not self.video_writer.isOpened():
            self.status_label.setText(self.texts["error"] + ": " + self.texts["recording_start_error"])
            self.video_writer = None
            return

        self.recording = True
        self.record_start_frame = self.current_frame_num
        self.recorded_detections = []

    def stop_recording(self):
        if self.current_seafloor_class and self.seafloor_annotation_frames:
            self._save_seafloor_annotation_segment()

        if not self.recording:
            return

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        self.recording = False
        self.status_label.setText(self.texts["recording_stopped"].format(self.recording_filename))

        for detection in self.recorded_detections:
            self.detections_dock.add_detection(detection)

        self.prompt_save_recording()

    def prompt_save_recording(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setText(self.texts["save_recording_question"])
        msg.setWindowTitle(self.texts["save_recording_title"])
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if msg.exec() == QMessageBox.StandardButton.Yes:
            self.save_recording()

    def save_recording(self):
        if not hasattr(self, 'recording_filename') or not os.path.exists(self.recording_filename):
            self.status_label.setText(self.texts["no_recording_to_save"])
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, self.texts["save_recording_title"], self.recording_filename,
            self.texts["video_files_filter"] + " (*.avi *.mp4);;" + self.texts["all_files"] + " (*)")

        if not file_path:
            return

        try:
            shutil.move(self.recording_filename, file_path)
            self.status_label.setText(self.texts["video_saved"].format(file_path))
            self._export_recording_annotations(file_path)
        except Exception as e:
            self.status_label.setText(self.texts["saving_video_error"].format(str(e)))

    def _export_recording_annotations(self, video_path):
        """Exporta anotações da gravação para CSV e frames."""
        video_name = Path(video_path).stem
        frames_dir = Path(video_path).parent / f"{video_name}_frames"
        frames_dir.mkdir(exist_ok=True)

        # Deduplicação por track_id
        best_by_track = {}
        for detection in self.recorded_detections:
            track_id = detection.get("track_id")
            if track_id is not None:
                if track_id not in best_by_track or detection.get("confidence", 0) > best_by_track[track_id].get("confidence", 0):
                    best_by_track[track_id] = detection
            else:
                key = f"manual_{detection.get('frame_number', 0)}_{detection.get('x1', 0)}_{detection.get('y1', 0)}"
                best_by_track[key] = detection

        unique_detections = list(best_by_track.values())

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.status_label.setText(self.texts["error_opening_video"])
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 20

        frames_dict = defaultdict(list)
        for det in unique_detections:
            frames_dict[det.get("frame_number", 0)].append(det)

        saved_frames = {}
        for frame_num, detections in sorted(frames_dict.items()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue

            for det in detections:
                if det.get("type") == "manual":
                    self._draw_manual_box_on_frame(frame, det)

            frame_filename = f"{video_name}_frame_{frame_num:06d}.jpg"
            frame_path = frames_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
            saved_frames[frame_num] = str(frame_path)

            seconds = frame_num / fps
            video_timestamp = self._format_time(seconds)
            for det in detections:
                det["frame_path"] = str(frame_path)
                det["video_timestamp"] = video_timestamp

        cap.release()

        # Salvar CSV
        annotation_path = os.path.splitext(video_path)[0] + "_annotations.csv"
        self._save_detections_csv(unique_detections, annotation_path, video_path)

        self.recorded_detections = []
        QMessageBox.information(
            self, self.texts["recording_saved_title"],
            f"Salvo {len(unique_detections)} anotações e {len(saved_frames)} frames")

    def _draw_manual_box_on_frame(self, frame, det):
        """Desenha bounding box manual em um frame OpenCV."""
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        cls = det.get("class", "unknown")
        conf = det.get("confidence", 1.0)
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{cls} {conf:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    def _save_detections_csv(self, detections, output_path, video_path):
        """Salva lista de detecções em arquivo CSV."""
        export_data = []
        for ann in detections:
            confidence = ann.get('confidence', 0)
            confidence_str = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else str(confidence)
            export_data.append({
                "Video": os.path.basename(video_path),
                "Timestamp": ann.get("video_timestamp", ""),
                "System_Date": ann.get("system_date", ""),
                "System_Time": ann.get("system_time", ""),
                "Taxon": ann.get("class", "Unknown"),
                "Confidence": confidence_str,
                "Type": ann.get("type", "unknown"),
                "Track_ID": ann.get("track_id", ""),
                "x1": ann.get("x1", ""), "y1": ann.get("y1", ""),
                "x2": ann.get("x2", ""), "y2": ann.get("y2", ""),
                "Frame_Number": ann.get("frame_number", ""),
                "Photo": ann.get("frame_path", "")
            })

        export_data.sort(key=lambda x: x["Frame_Number"] if x["Frame_Number"] != "" else 0)

        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["Video", "Timestamp", "System_Date", "System_Time",
                          "Taxon", "Confidence", "Type", "Track_ID",
                          "x1", "y1", "x2", "y2", "Frame_Number", "Photo"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(export_data)

    # -------------------------------------------------------------------------
    # DATASET
    # -------------------------------------------------------------------------

    def open_training_wizard(self):
        wizard = TrainingWizard(self)
        if wizard.exec() != QDialog.DialogCode.Accepted:
            return

        frames = wizard.frame_list
        if not frames:
            return

        ds_dir = Path(QFileDialog.getExistingDirectory(self, self.texts.get("select_yolo_project_dir", "Where to save the YOLO project?")))
        if not ds_dir:
            return

        images_dir = ds_dir / "images"
        labels_dir = ds_dir / "labels"
        for subdir in ["train", "val"]:
            (images_dir / subdir).mkdir(parents=True, exist_ok=True)
            (labels_dir / subdir).mkdir(parents=True, exist_ok=True)

        progress = QProgressDialog(self.texts.get("copying_frames", "Copying frames…"), self.texts.get("cancel_pt", "Cancel"), 0, len(frames), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        QApplication.processEvents()

        indices = list(range(len(frames)))
        random.seed(42)
        random.shuffle(indices)
        split_idx = int(0.8 * len(indices))
        train_indices = set(indices[:split_idx])

        self.dataset_frames = []
        for i, (src, num, video_name) in enumerate(frames):
            if progress.wasCanceled():
                break
            src_path = Path(src)
            split = "train" if i in train_indices else "val"
            new_name = f"frame_{i:06d}{src_path.suffix}"
            dst = images_dir / split / new_name
            try:
                shutil.copy2(src, dst)
                self.dataset_frames.append((str(dst), num, i))
            except Exception as e:
                QMessageBox.warning(self, self.texts.get("error_title", "Error"), self.texts.get("copy_failed_format", "Failed to copy: {}").format(e))
            progress.setValue(i + 1)
            QApplication.processEvents()

        progress.close()

        if self.dataset_frames:
            self._setup_dataset_mode()

    def _setup_dataset_mode(self):
        """Configura o estado para modo dataset."""
        first_path = self.dataset_frames[0][0]
        self.start_video(first_path)
        self.current_frame_num = 0
        self.sync_frame_num()
        self.total_frames = len(self.dataset_frames)
        self.paused = True
        self.dataset_mode = True
        self.dataset_index = 0
        self.detections_dock.detections_list.clear()
        self.detections_dock.all_detections.clear()
        self.video_label.frame_annotations = {}
        self.video_label.active_annotations = []
        self.video_label.segmentation_annotations = []
        self.all_detections = []
        self.annotations = []
        QMessageBox.information(self, self.texts.get("dataset_ready_title", "Dataset Ready"), self.texts.get("dataset_frames_loaded", "{} frames loaded").format(len(self.dataset_frames)))

    def load_dataset_frame(self, index):
        if not (0 <= index < len(self.dataset_frames)):
            return

        self.dataset_index = index
        image_path, original_num, dataset_idx = self.dataset_frames[index]
        frame = cv2.imread(image_path)
        if frame is None:
            return

        self.current_frame = frame.copy()
        self.sync_frame_num(index)
        h, w = frame.shape[:2]
        self.video_label.original_width = w
        self.video_label.original_height = h
        self.video_label.current_frame_num = dataset_idx
        self.video_label.update_active_annotations()
        self.display_frame(frame)
        self.update_time_labels()
        self.video_name_label.setText(self.texts.get("dataset_prefix", "[Dataset] {}").format(Path(image_path).name))

    def on_dataset_frame_changed(self, frame_num):
        self.sync_frame_num(frame_num)
        self.video_label.current_frame_num = frame_num
        self.video_label.update_active_annotations()
        self.update()


    # -------------------------------------------------------------------------
    # EXPORTAÇÃO YOLO
    # -------------------------------------------------------------------------

    def export_yolo_annotations_dialog(self):
        output_dir = QFileDialog.getExistingDirectory(self, self.texts["export_yolo_dialog"])
        if output_dir:
            self.export_yolo_annotations(output_dir)

    def export_yolo_annotations(self, output_dir):
        if not hasattr(self, 'all_detections'):
            QMessageBox.warning(self, self.texts["warning"], self.texts["no_annotations_to_export"])
            return

        manual_annotations = [
            ann for ann in self.all_detections
            if ann.get("type") == "manual" and all(key in ann for key in ["x1", "y1", "x2", "y2", "class"])
        ]

        # Agrupa anotações por frame
        frames_dict = defaultdict(list)
        for ann in manual_annotations:
            frames_dict[ann.get("frame_number", 0)].append(ann)

        try:
            output_path = Path(output_dir)
            images_dir = output_path / "images"
            labels_dir = output_path / "labels"
            for subdir in ["train", "val"]:
                (images_dir / subdir).mkdir(parents=True, exist_ok=True)
                (labels_dir / subdir).mkdir(parents=True, exist_ok=True)

            # Carrega classes existentes
            yaml_path = output_path / "dataset.yaml"
            existing_classes = self._load_existing_classes(yaml_path)

            # Mescla classes
            new_classes = set(ann["class"] for ann in manual_annotations)
            merged_classes = dict(existing_classes)
            next_id = max(merged_classes.values()) + 1 if merged_classes else 0
            for cls in new_classes:
                if cls not in merged_classes:
                    merged_classes[cls] = next_id
                    next_id += 1

            class_to_id = merged_classes

            is_dataset_mode = self.dataset_mode and self.dataset_frames

            if is_dataset_mode:
                # >>> Exportar TODOS os frames (com e sem labels) <<<
                all_frames = list(range(len(self.dataset_frames)))
                dataset_index_to_path = self._build_dataset_index_map(images_dir)
                get_split = lambda fn: dataset_index_to_path.get(fn, ('', 'train'))[1]
            else:
                all_frames = sorted(frames_dict.keys())
                get_split = lambda fn: 'train' if (fn % 10) < 8 else 'val'

            if not all_frames:
                self.show_warning_message("warning", "no_manual_annotations")
                return

            progress = QProgressDialog(self.texts["exporting_frames"], self.texts["cancel"], 0, len(all_frames), self)
            progress.setWindowTitle(self.texts["exporting_dataset"])
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.show()

            processed = 0
            video_name_prefix = ""
            if not is_dataset_mode and self.video_path and self.video_path != "Live":
                video_name = Path(self.video_path).stem
                video_name_prefix = "".join(c for c in video_name if c.isalnum() or c in ('_', '-'))

            for i, frame_num in enumerate(all_frames):
                progress.setValue(i)
                QApplication.processEvents()
                if progress.wasCanceled():
                    break

                frame, img_name, split, label_path = self._prepare_frame_for_export(
                    frame_num, images_dir, labels_dir, get_split, is_dataset_mode, video_name_prefix
                )
                if frame is None:
                    continue

                # Copia imagem SEMPRE
                img_dest = images_dir / split / img_name
                if not img_dest.exists():
                    cv2.imwrite(str(img_dest), frame)

                # Só cria .txt se houver anotações para este frame
                if frame_num in frames_dict and frames_dict[frame_num]:
                    new_lines = self._generate_yolo_labels(
                        frames_dict[frame_num], frame.shape[1], frame.shape[0], class_to_id
                    )
                    with open(label_path, 'w') as f:
                        for line in sorted(new_lines):
                            f.write(line + "\n")

                processed += 1

            progress.close()
            self._write_dataset_yaml(yaml_path, output_path, class_to_id)
            self._show_export_summary(images_dir, labels_dir, len(manual_annotations), processed, list(class_to_id.keys()), output_dir)

        except Exception as e:
            QMessageBox.critical(self, self.texts["error"], f"Export failed: {str(e)}")

    def _load_existing_classes(self, yaml_path):
        """Carrega classes existentes de um dataset.yaml."""
        existing = {}
        if yaml_path.exists():
            try:
                with open(yaml_path, 'r') as f:
                    yaml_data = yaml.safe_load(f)
                if yaml_data and 'names' in yaml_data:
                    names = yaml_data['names']
                    if isinstance(names, dict):
                        for idx, name in names.items():
                            existing[str(name)] = int(idx) if str(idx).isdigit() else idx
                    elif isinstance(names, list):
                        for idx, name in enumerate(names):
                            existing[str(name)] = idx
            except Exception as e:
                print(f"Warning: Could not load existing dataset.yaml: {e}")
        return existing

    def _build_dataset_index_map(self, images_dir):
        """Constrói mapeamento de índice do dataset para caminho e split."""
        mapping = {}
        for idx, (img_path, orig_num, dataset_idx) in enumerate(self.dataset_frames):
            p = Path(img_path)
            split = None
            for sp in ['train', 'val']:
                if (images_dir / sp / p.name).exists():
                    split = sp
                    break
            if split is None:
                split = p.parent.name if p.parent.name in ('train', 'val') else 'train'
            mapping[dataset_idx] = (p.name, split, str(p))
        return mapping

    def _prepare_frame_for_export(self, frame_num, images_dir, labels_dir, get_split, is_dataset_mode, video_name_prefix):
        """Prepara frame e caminhos para exportação YOLO."""
        if is_dataset_mode:
            if frame_num >= len(self.dataset_frames):
                return None, None, None, None
            img_path, _, _ = self.dataset_frames[frame_num]
            img = cv2.imread(img_path)
            if img is None:
                return None, None, None, None
            h, w = img.shape[:2]
            img_name = Path(img_path).name
            split = get_split(frame_num)
            
            # copiar imagem para o novo diretório de treinamento 
            dest_img_path = images_dir / split / img_name
            if not dest_img_path.exists():
                dest_img_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dest_img_path), img)
            
            label_name = Path(img_name).stem + ".txt"
            return img, img_name, split, labels_dir / split / label_name
        else:
            if not (self.cap and self.cap.isOpened()):
                return None, None, None, None
            current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
            if not ret:
                return None, None, None, None
            h, w = frame.shape[:2]
            split = get_split(frame_num)
            img_name = f"{video_name_prefix}_{frame_num:06d}.jpg" if video_name_prefix else f"frame_{frame_num:06d}.jpg"
            img_path = images_dir / split / img_name
            if not img_path.exists():
                cv2.imwrite(str(img_path), frame)
            label_name = Path(img_name).stem + ".txt"
            return frame, img_name, split, labels_dir / split / label_name

    def _generate_yolo_labels(self, annotations, w, h, class_to_id):
        """Gera linhas YOLO formatadas a partir de anotações."""
        lines = set()
        for ann in annotations:
            try:
                x1 = max(0, ann["x1"])
                y1 = max(0, ann["y1"])
                x2 = min(w, ann["x2"])
                y2 = min(h, ann["y2"])
                if x2 <= x1 or y2 <= y1:
                    continue
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                box_w = (x2 - x1) / w
                box_h = (y2 - y1) / h
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                box_w = max(0.0, min(1.0, box_w))
                box_h = max(0.0, min(1.0, box_h))
                class_id = class_to_id[ann["class"]]
                line = f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
                lines.add(line)
            except Exception:
                continue
        return lines

    def _write_dataset_yaml(self, yaml_path, output_path, class_to_id):
        """Escreve/atualiza o arquivo dataset.yaml."""
        config = {
            "path": str(output_path.absolute()).replace("\\", "/") + "/",
            "train": "images/train",
            "val": "images/val",
            "nc": len(class_to_id),
            "names": {cid: name for name, cid in class_to_id.items()}
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def _show_export_summary(self, images_dir, labels_dir, n_annotations, processed, classes, output_dir):
        """Exibe resumo da exportação."""
        train_imgs = len(list((images_dir / "train").glob('*.*')))
        val_imgs = len(list((images_dir / "val").glob('*.*')))
        train_labels = len(list((labels_dir / "train").glob('*.txt')))
        val_labels = len(list((labels_dir / "val").glob('*.txt')))

        QMessageBox.information(
            self, "Export Complete",
            f"Dataset exportado com sucesso!\n\n"
            f"Train: {train_imgs} images, {train_labels} labels\n"
            f"Val: {val_imgs} images, {val_labels} labels\n"
            f"Anotações exportadas: {n_annotations}\n"
            f"Frames processados: {processed}\n"
            f"Classes: {classes}\n\n"
            f"Salvo em: {output_dir}"
        )


    # -------------------------------------------------------------------------
    # IMPORTAÇÃO DE DATASET YOLO
    # -------------------------------------------------------------------------

    def import_yolo_dataset(self):
        dataset_dir = QFileDialog.getExistingDirectory(
            self, self.texts.get("import_dataset", "Import YOLO Dataset"))
        if not dataset_dir:
            return

        try:
            dataset_path = Path(dataset_dir)
            images_dir = dataset_path / "images"
            labels_dir = dataset_path / "labels"

            if not images_dir.exists():
                self.show_warning_message("warning", "invalid_dataset_structure")
                return

            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
            image_files = []
            for split in ['train', 'val']:
                split_dir = images_dir / split
                if split_dir.exists():
                    for f in split_dir.iterdir():
                        if f.suffix.lower() in image_extensions:
                            image_files.append((f, split))

            if not image_files:
                self.show_warning_message("warning", "no_images_found")
                return

            image_files.sort(key=lambda x: x[0].name)
            classes, class_id_map = self._load_classes_from_yaml_or_labels(dataset_path, labels_dir)

            self.dataset_frames = []
            for i, (img_path, split) in enumerate(image_files):
                self.dataset_frames.append((str(img_path), i, i))

            self.all_detections = []
            self.video_label.frame_annotations = {}
            self.video_label.confirmed_masks = {}

            valid_count = self._load_annotations_from_labels(image_files, labels_dir, class_id_map)

            self.dataset_mode = True
            if self.dataset_frames:
                self._setup_imported_dataset(classes, len(image_files), valid_count)

        except Exception as e:
            self.show_error_message("error", "dataset_import_error", str(e))

    def _load_classes_from_yaml_or_labels(self, dataset_path, labels_dir):
        """Carrega classes do dataset.yaml ou infere dos arquivos de label.
        Retorna: (lista_nomes, dict_id_para_nome)
        """
        yaml_path = dataset_path / "dataset.yaml"
        classes = []
        class_id_map = {}

        if yaml_path.exists():
            try:
                for encoding in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']:
                    try:
                        with open(yaml_path, 'r', encoding=encoding) as f:
                            yaml_data = yaml.safe_load(f)
                        if yaml_data is not None:
                            break
                    except UnicodeDecodeError:
                        continue

                if yaml_data and 'names' in yaml_data:
                    names = yaml_data['names']
                    if isinstance(names, dict):
                        for k, v in names.items():
                            try:
                                idx = int(k)
                            except (ValueError, TypeError):
                                continue
                            name = str(v)
                            class_id_map[idx] = name
                            classes.append(name)
                    elif isinstance(names, list):
                        for idx, name in enumerate(names):
                            class_id_map[idx] = str(name)
                            classes.append(str(name))
            except Exception:
                pass

        if not classes and labels_dir.exists():
            all_labels = set()
            for split in ['train', 'val']:
                split_dir = labels_dir / split
                if split_dir.exists():
                    for label_file in split_dir.glob('*.txt'):
                        try:
                            with open(label_file, 'r') as f:
                                for line in f:
                                    line = line.strip()
                                    if line:
                                        parts = line.split()
                                        if parts and parts[0].isdigit():
                                            all_labels.add(int(parts[0]))
                        except Exception:
                            continue
            if all_labels:
                for i in range(max(all_labels) + 1):
                    name = f"class_{i}"
                    class_id_map[i] = name
                    classes.append(name)

        return (classes, class_id_map) if classes else (["unknown"], {})

    def _load_annotations_from_labels(self, image_files, labels_dir, class_id_map):
        """Carrega anotações YOLO dos arquivos .txt para memória."""
        valid_count = 0
        for dataset_idx, (img_path, split) in enumerate(image_files):
            img_path_obj = Path(img_path)
            label_file = labels_dir / split / f"{img_path_obj.stem}.txt"
            if not label_file.exists():
                continue

            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                frame_annotations = []

                with open(label_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) != 5:
                            continue
                        try:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                        except (ValueError, IndexError):
                            continue

                        x1 = int((x_center - width / 2) * w)
                        y1 = int((y_center - height / 2) * h)
                        x2 = int((x_center + width / 2) * w)
                        y2 = int((y_center + height / 2) * h)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w - 1, x2), min(h - 1, y2)
                        if x2 <= x1 or y2 <= y1:
                            continue

                        # >>> lookup direto pelo ID numérico do YAML <<<
                        class_name = class_id_map.get(class_id, f"class_{class_id}")
                        annotation = {
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "label": class_name, "class": class_name,
                            "confidence": 1.0, "type": "manual",
                            "frame_number": dataset_idx,
                            "video_path": str(img_path), "timestamp": "00:00:00",
                            "frame_source": (str(img_path), dataset_idx),
                            "frame_dimensions": f"{w}x{h}",
                            "bbox_id": str(uuid.uuid4())[:8]
                        }
                        self.all_detections.append(annotation)
                        frame_annotations.append(annotation)
                        valid_count += 1

                if frame_annotations:
                    self.video_label.frame_annotations[dataset_idx] = frame_annotations

            except Exception:
                continue

        return valid_count

    def _setup_imported_dataset(self, classes, total_images, valid_annotations):
        """Configura UI após importação bem-sucedida."""
        first_path = self.dataset_frames[0][0]
        self.start_video(first_path)
        self.current_frame_num = 0
        self.sync_frame_num()
        self.total_frames = len(self.dataset_frames)
        self.paused = True
        self.dataset_mode = True
        self.dataset_index = 0
        self.video_label.current_frame_num = 0
        self.video_label.update_active_annotations()
        self.custom_classes = classes
        self.refresh_taxon_grid()
        self.detections_dock.all_detections = self.all_detections.copy()
        self.detections_dock.apply_filters()
        self.set_status_message("dataset_imported", total_images, valid_annotations)
        self.show_info_message("dataset_ready", "dataset_import_success",
                               total_images, ', '.join(classes), valid_annotations)

    # -------------------------------------------------------------------------
    # TREINAMENTO YOLO
    # -------------------------------------------------------------------------

    def train_yolo_model(self):
        has_dataset = self.dataset_frames and len(self.dataset_frames) > 0

        if not hasattr(self, 'all_detections'):
            self.all_detections = []

        for ann in self.video_label.active_annotations:
            if ann.get("type") == "manual" and ann not in self.all_detections:
                ann.setdefault("frame_number", self.current_frame_num)
                ann.setdefault("video_path", self.video_path or "Live")
                self.all_detections.append(ann)

        manual_annotations = [
            ann for ann in self.all_detections
            if ann.get("type") == "manual" and all(key in ann for key in ["x1", "y1", "x2", "y2", "class"])
        ]

        if not manual_annotations and has_dataset:
            self._train_from_existing_dataset()
            return
        elif not manual_annotations:
            self.show_warning_message("warning", "no_manual_annotations_train")
            return

        self._train_from_annotations(manual_annotations)

    def _train_from_existing_dataset(self):
        """Treina a partir de um dataset YOLO existente."""
        reply = QMessageBox.question(
            self, self.texts["train_from_dataset_title"],
            self.texts["train_from_dataset_question"],
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if reply != QMessageBox.StandardButton.Yes:
            return

        dataset_dir = QFileDialog.getExistingDirectory(
            self, self.texts["select_dataset_folder"],
            str(Path(self.dataset_frames[0][0]).parent.parent) if self.dataset_frames else "")
        if not dataset_dir:
            return

        self._run_training(dataset_dir)

    def _train_from_annotations(self, manual_annotations):
        """Treina exportando anotações manuais primeiro."""
        dataset_dir = QFileDialog.getExistingDirectory(self, self.texts["train_dataset_dialog"])
        if not dataset_dir:
            return

        self.export_yolo_annotations(dataset_dir)
        self._run_training(dataset_dir, manual_annotations)

    def _run_training(self, dataset_dir, manual_annotations=None):
        """Executa o treinamento YOLO com configuração de diálogo."""
        try:
            models_dir = os.path.join(os.getcwd(), "modelos")
            os.makedirs(models_dir, exist_ok=True)

            name, ok = QInputDialog.getText(self, self.texts["name_model_title"], self.texts["name_model_label"])
            if not ok or not name.strip():
                return
            safe_name = "".join(c for c in name.strip() if c.isalnum() or c in ("_", "-"))
            if not safe_name:
                safe_name = "custom_model"

            # Carrega classes
            classes, class_to_id = self._resolve_classes_for_training(dataset_dir, manual_annotations)
            if not classes:
                QMessageBox.warning(self, self.texts.get("error_title", "Error"), self.texts.get("no_classes_found", "No classes found."))
                return

            config_path = self._create_or_update_dataset_yaml(dataset_dir, class_to_id)

            train_config = self._show_training_dialog(dataset_dir, safe_name, models_dir)
            if not train_config:
                return

            progress = QProgressDialog(
                self.texts["training_progress"], self.texts["cancel"],
                0, train_config["epochs"], self)
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
            self.show_error_message("error", "config_failed", str(e))

    def _resolve_classes_for_training(self, dataset_dir, manual_annotations=None):
        """Resolve classes preservando IDs existentes do dataset.yaml."""
        yaml_path = os.path.join(dataset_dir, "dataset.yaml")
        existing_classes = {}

        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, 'r') as f:
                    yaml_data = yaml.safe_load(f)
                if yaml_data and 'names' in yaml_data:
                    names = yaml_data['names']
                    if isinstance(names, dict):
                        existing_classes = {str(name): int(idx) for idx, name in names.items()}
                    elif isinstance(names, list):
                        existing_classes = {str(name): idx for idx, name in enumerate(names)}
            except Exception:
                pass

        if manual_annotations:
            new_classes = set(ann["class"] for ann in manual_annotations)
            merged = dict(existing_classes)
            next_id = max(merged.values()) + 1 if merged else 0
            for cls in new_classes:
                if cls not in merged:
                    merged[cls] = next_id
                    next_id += 1
            return list(merged.keys()), merged

        return list(existing_classes.keys()), existing_classes

    def _create_or_update_dataset_yaml(self, dataset_dir, class_to_id):
        """Cria ou atualiza dataset.yaml."""
        config = {
            "path": os.path.abspath(dataset_dir).replace("\\", "/") + "/",
            "train": "images/train",
            "val": "images/val",
            "names": {cid: name for name, cid in class_to_id.items()},
            "nc": len(class_to_id)
        }
        config_path = os.path.join(dataset_dir, "dataset.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        return config_path

    def _show_training_dialog(self, dataset_dir, safe_name, models_dir):
        """Exibe diálogo de configuração avançada do treinamento."""
        train_config = {
            "data": os.path.join(dataset_dir, "dataset.yaml"),
            "epochs": 100, "imgsz": 640, "batch": 8,
            "project": models_dir, "name": safe_name,
            "exist_ok": True, "patience": 20,
            "optimizer": "auto", "lr0": 0.01,
            "device": "0" if torch.cuda.is_available() else "cpu",
            "workers": 4, "save_period": 10,
            "single_cls": False, "augment": True,
        }

        dialog = QDialog(self)
        dialog.setWindowTitle(self.texts["advanced_training_settings"])
        layout = QVBoxLayout()
        form = QFormLayout()

        epochs_spin = QSpinBox()
        epochs_spin.setRange(1, 1000)
        epochs_spin.setValue(train_config["epochs"])
        form.addRow(self.texts["epochs"], epochs_spin)

        batch_spin = QSpinBox()
        batch_spin.setRange(1, 64)
        batch_spin.setValue(train_config["batch"])
        form.addRow(self.texts["batch_size"], batch_spin)

        imgsz_spin = QSpinBox()
        imgsz_spin.setRange(320, 1280)
        imgsz_spin.setSingleStep(32)
        imgsz_spin.setValue(train_config["imgsz"])
        form.addRow(self.texts["image_size"], imgsz_spin)

        lr_spin = QDoubleSpinBox()
        lr_spin.setRange(0.0001, 0.1)
        lr_spin.setSingleStep(0.001)
        lr_spin.setValue(train_config["lr0"])
        form.addRow(self.texts["learning_rate"], lr_spin)

        device_combo = QComboBox()
        device_combo.addItems(["CPU", "GPU"] if torch.cuda.is_available() else ["CPU"])
        form.addRow(self.texts["device"], device_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        layout.addLayout(form)
        layout.addWidget(buttons)
        dialog.setLayout(layout)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        train_config.update({
            "epochs": epochs_spin.value(),
            "batch": batch_spin.value(),
            "imgsz": imgsz_spin.value(),
            "lr0": lr_spin.value(),
            "device": "0" if device_combo.currentText() == "GPU" else "cpu"
        })
        return train_config

    def on_training_finished(self, progress):
        progress.close()
        if self.train_thread.success:
            reply = QMessageBox.question(
                self, self.texts["training_completed"],
                self.texts["training_success"].format(self._last_model_path),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.load_model(self._last_model_path)
        else:
            QMessageBox.critical(self, self.texts["training_error_title"],
                                 self.texts["training_error"].format(self.train_thread.error))


    # -------------------------------------------------------------------------
    # SAM 2 - SEGMENTAÇÃO
    # -------------------------------------------------------------------------

    def init_sam2(self):
        try:
            self.sam2_thread = SAM2Thread("sam2.1_b.pt", self)
            self.sam2_thread.mask_preview.connect(self.on_hover_mask_preview)
            self.sam2_thread.error.connect(self.on_sam2_error)
            self.sam2_thread.start()
        except Exception:
            self.status_label.setText(self.texts.get("sam2_init_failed", "SAM 2 initialization failed"))

    def on_sam2_error(self, error_msg):
        self.status_label.setText(self.texts["sam2_error"].format(error_msg))

    def toggle_hover_segmentation(self):
        """Ativa/desativa modo hover-to-segment via SAM 2."""
        self.hover_segmentation_mode = not self.hover_segmentation_mode

        if self.hover_segmentation_mode:
            if self.video_label.drawing_enabled:
                self.enable_manual_annotation()
            self.video_label.hover_segmentation_enabled = True
            self.video_label.setCursor(self.video_label.cursor_sam)
            self.sam_button.setStyleSheet(SAM_BTN_ACTIVE_STYLE)
            self.set_status_message("hover_segmentation_on")
            self._connect_sam_signals()
        else:
            self.video_label.hover_segmentation_enabled = False
            self.video_label._clear_preview()
            self.video_label.unsetCursor()
            self.sam_button.setStyleSheet(self._get_button_style())
            self.set_status_message("hover_segmentation_off")
            self._disconnect_sam_signals()

    def _connect_sam_signals(self):
        """Conecta sinais do video_label para SAM."""
        for signal, slot in [
            (self.video_label.hover_point, self.on_hover_point),
            (self.video_label.hover_cleared, self.on_hover_cleared),
            (self.video_label.mask_confirmed, self.on_mask_confirmed),
        ]:
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
            signal.connect(slot)

    def _disconnect_sam_signals(self):
        """Desconecta sinais do video_label do SAM."""
        for signal, slot in [
            (self.video_label.hover_point, self.on_hover_point),
            (self.video_label.hover_cleared, self.on_hover_cleared),
            (self.video_label.mask_confirmed, self.on_mask_confirmed),
        ]:
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass

    def on_hover_point(self, x, y, frame_num):
        if not self.hover_segmentation_mode:
            return

        actual_frame = getattr(self, 'dataset_index', 0) if getattr(self, 'dataset_mode', False) else self.current_frame_num
        frame = self._get_frame_for_sam()
        if frame is None:
            return

        h, w = frame.shape[:2]
        x = max(0, min(int(x), w - 1))
        y = max(0, min(int(y), h - 1))

        self.video_label.set_sam_processing(True)
        self.sam2_thread.set_frame_and_prompts(
            frame, actual_frame, [[float(x), float(y)]],
            prompt_type="points", preview=True
        )

    def on_hover_cleared(self):
        pass

    def on_hover_mask_preview(self, mask_data, original_frame, frame_num):
        self.video_label.set_sam_processing(False)
        if not self.hover_segmentation_mode:
            return

        current = getattr(self, 'dataset_index', 0) if getattr(self, 'dataset_mode', False) else self.current_frame_num
        if frame_num != current:
            return
        self.video_label.set_preview_mask(mask_data)

    def on_mask_confirmed(self, seg_annotation):
        """Callback quando segmentação SAM é confirmada."""
        if not hasattr(self, 'segmentation_annotations'):
            self.segmentation_annotations = []
        self.segmentation_annotations.append(seg_annotation)
        self.set_status_message("segmentation_confirmed",
                                seg_annotation.get("class", self.texts.get("unknown", "Unknown")))

    def _get_frame_for_sam(self):
        """Obtém frame atual para processamento SAM (com cache)."""
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            return self.current_frame.copy()

        if getattr(self, 'dataset_mode', False) and self.dataset_frames:
            idx = getattr(self, 'dataset_index', 0)
            if 0 <= idx < len(self.dataset_frames):
                image_path, _, _ = self.dataset_frames[idx]
                if os.path.exists(image_path):
                    frame = cv2.imread(image_path)
                    if frame is not None:
                        self.current_frame = frame
                        return frame

        elif hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = self.cap.read()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
            if ret:
                self.current_frame = frame
                return frame

        return None


    # -------------------------------------------------------------------------
    # SEAFLOOR CLASSIFICATION
    # -------------------------------------------------------------------------

    def open_seafloor_dialog(self):
        dialog = SeafloorClassificationDialog(self, self.language)
        dialog.exec()

    def classify_current_frame(self):
        """Classifica frame atual usando modelo treinado (Ctrl+F)."""
        if self.cap is None or not self.cap.isOpened():
            self.status_label.setText(self.texts.get("no_video_loaded", "No video loaded!"))
            return

        current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        if not ret:
            self.status_label.setText(self.texts.get("error_reading_frame", "Error reading frame!"))
            return

        if self.seafloor_classifier is None:
            self.seafloor_classifier = SeafloorClassifier(normalize_colors=False, fast_mode=True, language=self.language)

        if not self.seafloor_classifier.has_trained_model():
            model_path = self.seafloor_training_dir / "seafloor_model.pt"
            config_path = self.seafloor_training_dir / "classifier_config.json"
            if model_path.exists():
                try:
                    self.status_label.setText(self.texts.get("loading_trained_model", "Loading trained model..."))
                    self.seafloor_classifier.load_model(str(model_path))
                    if config_path.exists():
                        self.seafloor_classifier.load_config(str(config_path))
                except Exception as e:
                    self.status_label.setText(self.texts.get("model_load_error_format", "Error loading model: {}").format(str(e)))
                    return
            else:
                self.status_label.setText(self.texts.get("no_trained_model", "No trained model found. Use 'Train from Collected Data' first."))
                return

        try:
            result = self.seafloor_classifier.predict_single_frame(frame)
            self._display_seafloor_result(frame, result)
        except Exception as e:
            self.status_label.setText(self.texts.get("classification_error_format", "Classification error: {}").format(str(e)))

    def _display_seafloor_result(self, frame, result):
        """Exibe resultado de classificação de seafloor no frame."""
        class_name = result["class_name"]      # já traduzido
        confidence = result["confidence"]
        # Usa internal_name para buscar cor, com fallback seguro
        internal_name = result.get("internal_name", class_name)
        color_hex = result.get("color") or self.seafloor_classifier.class_colors.get(internal_name, "#1565C0")
        color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

        h, w = frame.shape[:2]
        label = f"{class_name}: {confidence:.2f}"
        cv2.rectangle(frame, (10, 10), (300, 50), color_bgr, -1)
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        self.display_frame(frame)
        self._update_info_panel(f" {class_name}: {confidence:.2f}", color_hex=color_hex)

        detection = {
            "x1": 10, "y1": 10, "x2": 300, "y2": 50,
            "class": class_name, "confidence": confidence,
            "type": "seafloor", "frame_number": int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)),
            "timestamp": self.get_video_timestamp(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))),
            "video_path": self.video_path or "Live"
        }
        self.detections_dock.add_detection(detection)

    def toggle_seafloor_realtime(self, enabled):
        """Ativa/desativa classificação em tempo real de fundo marinho."""
        self.seafloor_enabled = enabled

        if enabled:
            if self.seafloor_classifier is None:
                self.seafloor_classifier = SeafloorClassifier(normalize_colors=False, fast_mode=True, language=self.language)

            if not self.seafloor_classifier.has_trained_model():
                model_path = self.seafloor_training_dir / "seafloor_model.pt"
                if model_path.exists():
                    try:
                        self.seafloor_classifier.load_model(str(model_path))
                        config_path = self.seafloor_training_dir / "classifier_config.json"
                        if config_path.exists():
                            self.seafloor_classifier.load_config(str(config_path))
                    except Exception as e:
                        self.status_label.setText(self.texts.get("model_load_error_format", "Error loading model: {}").format(str(e)))
                        self.seafloor_enabled = False
                        return
                else:
                    self.status_label.setText(self.texts.get("model_not_found_train", "Model not found. Train first."))
                    self.seafloor_enabled = False
                    return

            self.seafloor_thread = SeafloorClassificationThread(self.seafloor_classifier)
            self.seafloor_thread.set_realtime_mode()
            self.seafloor_thread.frame_classified.connect(self.on_seafloor_frame_classified)
            self.seafloor_thread.start()
            self.status_label.setText(self.texts.get("seafloor_activated", "Seafloor classification: ON"))
        else:
            if self.seafloor_thread:
                self.seafloor_thread.stop()
                self.seafloor_thread.wait(1000)
                self.seafloor_thread = None
            self.status_label.setText(self.texts.get("seafloor_deactivated", "Seafloor classification: OFF"))

    def on_seafloor_frame_classified(self, result, frame_num):
        """Handle real-time classification result."""
        self.current_seafloor_result = result
        class_name = result["class_name"]
        confidence = result["confidence"]
        internal_name = result.get("internal_name", class_name)
        shortcut = result.get("shortcut", "")
        # Cor oficial hardcoded — nunca falha, nunca branca
        OFFICIAL_COLORS = {
            "Sedimento": "#941363", "Sediment": "#941363",
            "Coral_Fragmento": "#0E7607", "Coral Fragment": "#0E7607",
            "Recife_Coral": "#1F00D1", "Coral Reef": "#1F00D1",
        }
        color_hex = OFFICIAL_COLORS.get(internal_name) or OFFICIAL_COLORS.get(class_name, "#1565C0")

        self.status_label.setText(self.texts.get("seafloor_label", "Seafloor: {} ({:.2f})").format(class_name, confidence))
        label_text = f" [{shortcut}] {class_name}: {confidence:.2f}" if shortcut else f" {class_name}: {confidence:.2f}"
        self._update_info_panel(label_text, color_hex=color_hex)

    def _update_seafloor_custom_menu(self, menu=None):
        """Atualiza o menu com classes customizadas do classificador."""
        if menu is None:
            menubar = self.menuBar()
            for action in menubar.actions():
                if action.text() == self.texts.get("seafloor_menu", "Seafloor"):
                    menu = action.menu()
                    break
        if not menu:
            return

        for action in self._seafloor_custom_actions:
            menu.removeAction(action)
        self._seafloor_custom_actions = []

        if self.seafloor_classifier:
            for cls_info in self.seafloor_classifier.list_all_classes():
                if cls_info["type"] == "custom":
                    shortcut = cls_info["shortcut"]
                    name = cls_info["name"]
                    action = QAction(f"{shortcut} - {name}", self)
                    action.setShortcut(QKeySequence(shortcut))
                    action.triggered.connect(lambda checked, c=name, k=shortcut: self.quick_classify_seafloor(c, k))
                    menu.insertAction(self._seafloor_stop_action, action)
                    self._seafloor_custom_actions.append(action)

    def quick_classify_seafloor(self, class_name, shortcut):
        """Atalho rápido para classificar fundo durante anotação de vídeo."""
        if not self.cap or not self.cap.isOpened():
            self.status_label.setText(self.texts.get("no_video_loaded_excl", "No video loaded!"))
            return

        if self.current_seafloor_class and self.current_seafloor_class != class_name:
            self._save_seafloor_annotation_segment()

        self.current_seafloor_class = class_name
        self.current_seafloor_shortcut = shortcut
        self.seafloor_annotation_start_frame = self.current_frame_num
        self.seafloor_frame_counter = 0

        # Busca cor: primeiro no classificador, depois no mapping fixo
        color_hex = None
        if self.seafloor_classifier:
            color_hex = self.seafloor_classifier.class_colors.get(class_name)
            # Fallback: tentar nome em português se o classificador normalizou
            if not color_hex and class_name == "Coral Reef":
                color_hex = self.seafloor_classifier.class_colors.get("Recife_Coral")
            elif not color_hex and class_name == "Sediment":
                color_hex = self.seafloor_classifier.class_colors.get("Sedimento")
            elif not color_hex and class_name == "Coral Fragment":
                color_hex = self.seafloor_classifier.class_colors.get("Coral_Fragmento")
        # Fallback final: cores oficiais fixas
        if not color_hex:
            official = {
                "Sedimento": "#8B4513", "Sediment": "#8B4513",
                "Coral_Fragmento": "#FF8C00", "Coral Fragment": "#FF8C00",
                "Recife_Coral": "#00CED1", "Coral Reef": "#00CED1",
            }
            color_hex = official.get(class_name)

        self._update_info_panel(f" [{shortcut}] {class_name} — COLETANDO | Shift+S para parar", color_hex=color_hex)
        self.seafloor_collecting = True

    def stop_seafloor_annotation(self):
        """Para a coleta de frames e salva o último segmento."""
        if self.current_seafloor_class:
            self._save_seafloor_annotation_segment()
            self.current_seafloor_class = None
            self.current_seafloor_shortcut = None
            self.seafloor_collecting = False
            self._update_info_panel("✓ Anotação de fundo finalizada")
            QTimer.singleShot(3000, lambda: self._update_info_panel(""))

    def _save_seafloor_annotation_segment(self):
        """Finaliza anotação do trecho anterior e salva metadados."""
        if not self.seafloor_annotation_frames:
            return

        segment_file = self.seafloor_training_dir / "annotations.csv"
        segment_file.parent.mkdir(parents=True, exist_ok=True)

        file_exists = segment_file.exists()
        with open(segment_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "video_path", "start_frame", "end_frame",
                "class", "shortcut", "frames_count", "start_timestamp", "end_timestamp"
            ])
            if not file_exists:
                writer.writeheader()

            start_frame = self.seafloor_annotation_frames[0]["frame_number"]
            end_frame = self.seafloor_annotation_frames[-1]["frame_number"]
            writer.writerow({
                "video_path": self.video_path or "Live",
                "start_frame": start_frame, "end_frame": end_frame,
                "class": self.current_seafloor_class,
                "shortcut": self.current_seafloor_shortcut or "",
                "frames_count": len(self.seafloor_annotation_frames),
                "start_timestamp": self.get_video_timestamp(start_frame),
                "end_timestamp": self.get_video_timestamp(end_frame)
            })

        self.status_label.setText(self.texts.get("segment_saved_format", "Segment saved: {} frames of {}").format(len(self.seafloor_annotation_frames), self.current_seafloor_class))
        self.seafloor_annotation_frames = []

    def _save_seafloor_training_frame(self):
        """Salva frame atual para dataset de treinamento do fundo."""
        if not hasattr(self, 'current_frame') or self.current_frame is None or not self.current_seafloor_class:
            return

        class_dir = self.seafloor_training_dir / self.current_seafloor_class
        class_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(self.video_path).stem if self.video_path else "live"
        filename = f"{video_name}_frame{self.current_frame_num}_{timestamp}.jpg"
        filepath = class_dir / filename
        cv2.imwrite(str(filepath), self.current_frame)

        self.seafloor_annotation_frames.append({
            "frame_number": self.current_frame_num,
            "class": self.current_seafloor_class,
            "shortcut": self.current_seafloor_shortcut,
            "filepath": str(filepath),
            "video_path": self.video_path or "Live",
            "timestamp": self.get_video_timestamp(self.current_frame_num)
        })

        total_saved = len(self.seafloor_annotation_frames)
        if total_saved % 5 == 0:
            color_hex = None
            if self.seafloor_classifier and self.current_seafloor_class in self.seafloor_classifier.class_colors:
                color_hex = self.seafloor_classifier.class_colors[self.current_seafloor_class]
            self._update_info_panel(
                f" [{self.current_seafloor_shortcut}] {self.current_seafloor_class} | "
                f"{total_saved} frames coletados | Shift+S para parar",
                color_hex=color_hex
            )

    def open_seafloor_class_manager(self):
        """Abre diálogo para gerenciar categorias do classificador de fundo."""
        if self.seafloor_classifier is None:
            self.seafloor_classifier = SeafloorClassifier(n_clusters=3, normalize_colors=False, fast_mode=True, language=self.language)

        from .seafloor_classifier import SeafloorClassManager
        dialog = SeafloorClassManager(self.seafloor_classifier, self)
        dialog.classes_changed.connect(lambda: self._update_seafloor_custom_menu())
        dialog.exec()

        classes_str = ", ".join(self.seafloor_classifier.class_names)
        self.status_label.setText(self.texts.get("seafloor_categories_format", "Seafloor categories: {}").format(classes_str))

    def train_seafloor_from_collected(self):
        """Treina classificador com frames coletados durante anotação de vídeo."""
        if not self.seafloor_training_dir.exists():
            QMessageBox.warning(self, self.texts.get("warning", "Warning"),
                "Nenhum dado coletado. Use os atalhos S/F/R/1-9/0 durante o vídeo primeiro.")
            return

        class_dirs = [d for d in self.seafloor_training_dir.iterdir()
                     if d.is_dir() and any(d.glob("*.jpg"))]

        if len(class_dirs) < 2:
            QMessageBox.warning(self, self.texts.get("warning", "Warning"),
                f"Dados insuficientes. Encontradas {len(class_dirs)} classes com imagens. São necessárias pelo menos 2.")
            return

        class_names = sorted([d.name for d in class_dirs])
        if self.seafloor_classifier is None:
            self.seafloor_classifier = SeafloorClassifier(
                custom_classes=[n for n in class_names if n not in SeafloorClassifier.FIXED_CLASSES.values()],
                normalize_colors=False, fast_mode=True, language=self.language
            )

        try:
            self.status_label.setText(self.texts.get("training_classifier", "Training classifier..."))
            QApplication.processEvents()

            results = self.seafloor_classifier.train_from_collected_data(
                data_dir=str(self.seafloor_training_dir),
                min_samples_per_class=5, parent_widget=self
            )

            self._update_seafloor_custom_menu()
            self.status_label.setText(self.texts.get("training_completed_format", "Training completed! Classes: {}").format(', '.join(results['class_names'])))
            QMessageBox.information(self, self.texts.get("training_completed_title", "Completed"),
                f"Modelo treinado com sucesso!\n"
                f"Classes: {', '.join(results['class_names'])}\n"
                f"Amostras: {results['n_samples']}\n"
                f"Modelo: {results['model_path']}")
        except Exception as e:
            QMessageBox.critical(self, self.texts.get("error_title", "Error"), self.texts.get("training_failed_format", "Training failed: {}").format(str(e)))


    # -------------------------------------------------------------------------
    # SALVAMENTO DE ANOTAÇÕES
    # -------------------------------------------------------------------------

    def save_annotations(self):
        if self.current_seafloor_class and self.seafloor_annotation_frames:
            self._save_seafloor_annotation_segment()

        if not self.video_path and not self.live_mode:
            self.status_label.setText(self.texts["no_loaded"])
            return

        default_name = "annotations.csv" if self.live_mode else f"{os.path.splitext(os.path.basename(self.video_path))[0]}_annotations.csv"
        output_path, _ = QFileDialog.getSaveFileName(
            self, self.texts["save_annotations"], default_name, "CSV Files (*.csv);;All Files (*)")
        if not output_path:
            return

        frames_dir = QFileDialog.getExistingDirectory(
            self, "Choose folder to save frames", str(Path(output_path).parent))
        if not frames_dir:
            return
        frames_dir = Path(frames_dir)

        try:
            frames_dir.mkdir(exist_ok=True)
            all_detections = self._collect_all_detections()
            seafloor_by_frame = self._load_seafloor_annotations()
            unique_detections = self._deduplicate_detections(all_detections)
            saved_frames = self._save_detection_frames(unique_detections, frames_dir)
            self._write_annotations_csv(unique_detections, saved_frames, seafloor_by_frame, output_path)

            self.status_label.setText(self.texts["annotations_saved"].format(output_path))
            QMessageBox.information(self, self.texts['export_completed'],
                f"{self.texts['annotations_saved'].format(output_path)}\n"
                f"{self.texts['frames_saved'].format(frames_dir)}\n")

        except Exception as e:
            self.set_status_message("saving_error")
            QMessageBox.critical(self, self.texts.get("error_title", "Error"), self.texts.get("export_failed_format", "Export failed: {}").format(str(e)))

    def _collect_all_detections(self):
        """Coleta todas as detecções de todas as fontes."""
        all_detections = []
        if hasattr(self.video_label, 'frame_annotations'):
            for frame_num, annotations in self.video_label.frame_annotations.items():
                for ann in annotations:
                    ann.setdefault("video_path", self.video_path or "Live")
                    all_detections.append(ann)
        if self.detections_dock:
            for d in self.detections_dock.filter_all():
                d.setdefault("video_path", self.video_path or "Live")
                all_detections.append(d)
        return [d for d in all_detections if d.get("type") != "training"]

    def _load_seafloor_annotations(self):
        """Carrega anotações de seafloor do CSV."""
        seafloor_by_frame = {}
        seafloor_csv = Path("seafloor_training_data") / "annotations.csv"
        seafloor_csv = seafloor_csv.resolve()

        if seafloor_csv.exists():
            with open(seafloor_csv, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        s = int(row.get("start_frame", 0))
                        e = int(row.get("end_frame", 0))
                        cls = row.get("class", "")
                        if cls:
                            for fn in range(s, e + 1):
                                seafloor_by_frame[fn] = cls
                    except (ValueError, TypeError):
                        continue

        if self.current_seafloor_class and self.seafloor_annotation_frames:
            for entry in self.seafloor_annotation_frames:
                seafloor_by_frame[entry.get("frame_number", 0)] = self.current_seafloor_class

        return seafloor_by_frame

    def _deduplicate_detections(self, detections):
        """Remove detecções duplicadas e mantém a melhor por track_id."""
        unique = []
        seen = set()
        for d in detections:
            key = (d.get("video_path"), d.get("frame_number"), d.get("timestamp"),
                   d.get("class"), d.get("x1"), d.get("y1"), d.get("x2"), d.get("y2"))
            if key not in seen:
                seen.add(key)
                unique.append(d)

        best = {}
        for d in unique:
            tid = d.get("track_id")
            if tid is not None:
                if tid not in best or d.get("confidence", 0) > best[tid].get("confidence", 0):
                    best[tid] = d
            else:
                k = f"{d.get('type', 'unknown')}_{d.get('frame_number', 0)}_{id(d)}"
                best[k] = d
        return list(best.values())

    def _save_detection_frames(self, detections, frames_dir):
        """Extrai e salva frames das detecções."""
        saved = {}
        progress = QProgressDialog(self.texts["exporting_frames"], self.texts["cancel"], 0, len(detections), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        for i, ann in enumerate(detections):
            if progress.wasCanceled():
                break
            fn = ann.get("frame_number")
            vp = ann.get("video_path", "Live")
            fk = (vp, fn)
            if fk in saved:
                ann["frame_path"] = saved[fk]
                continue

            if vp == "Live":
                if hasattr(self, 'current_frame') and self.current_frame is not None:
                    fp = frames_dir / f"live_frame_{fn or 0}.jpg"
                    cv2.imwrite(str(fp), self.current_frame)
                    saved[fk] = str(fp)
                else:
                    saved[fk] = "N/A"
            else:
                vn = Path(vp).stem
                fp = frames_dir / f"{vn}_frame_{fn:06d}.jpg"
                if not fp.exists():
                    c = cv2.VideoCapture(str(vp))
                    c.set(cv2.CAP_PROP_POS_FRAMES, fn)
                    r, fr = c.read()
                    if r:
                        cv2.imwrite(str(fp), fr)
                    c.release()
                saved[fk] = str(fp)

            ann["frame_path"] = saved[fk]
            progress.setValue(i)

        progress.close()
        return saved

    def _write_annotations_csv(self, detections, saved_frames, seafloor_by_frame, output_path):
        """Escreve arquivo CSV final com todas as anotações."""
        export_data = []
        for ann in detections:
            vp = ann.get("video_path", "")
            vn = "Live" if vp == "Live" else os.path.basename(str(vp)) if vp else "Unknown"
            conf = ann.get('confidence', 0)
            conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf)
            fn = ann.get("frame_number", 0)
            export_data.append({
                "Video": vn, "Timestamp": ann.get("timestamp", ""),
                "System_Date": ann.get("system_date", ""),
                "System_Time": ann.get("system_time", ""),
                "Taxon": ann.get("class", "Unknown"),
                "Confidence": conf_str, "Type": ann.get("type", "unknown"),
                "Track_ID": ann.get("track_id", ""),
                "x1": ann.get("x1", ""), "y1": ann.get("y1", ""),
                "x2": ann.get("x2", ""), "y2": ann.get("y2", ""),
                "Frame_Number": fn,
                "Seafloor": seafloor_by_frame.get(fn, ""),
                "Photo": ann.get("frame_path", "")
            })

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=[
                "Video", "Timestamp", "System_Date", "System_Time",
                "Taxon", "Confidence", "Type", "Track_ID",
                "x1", "y1", "x2", "y2", "Frame_Number", "Seafloor", "Photo"
            ])
            w.writeheader()
            w.writerows(export_data)

    def save_current_frame_with_annotations(self):
        """Salva o frame atual com anotações manuais desenhadas."""
        if not hasattr(self.video_label, '_pixmap') or self.video_label._pixmap is None:
            self.status_label.setText(self.texts["no_frame_to_save"])
            return

        pixmap = self.video_label._pixmap.copy()
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        video_rect = self.video_label.video_rect or QRect(0, 0, pixmap.width(), pixmap.height())

        for ann in self.video_label.active_annotations:
            self._draw_annotation_on_pixmap(painter, ann, video_rect)

        painter.end()

        default_name = f"{Path(self.video_path).stem}_frame_{self.current_frame_num}.png" if self.video_path else f"frame_{self.current_frame_num}.png"
        output_path, _ = QFileDialog.getSaveFileName(
            self, self.texts["save_frame_dialog"], default_name,
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;All Files (*)")

        if output_path:
            pixmap.save(output_path)
            self.status_label.setText(self.texts["frame_saved"].format(output_path))

    def _draw_annotation_on_pixmap(self, painter, ann, video_rect):
        """Desenha uma anotação manual em um QPixmap."""
        scale_x = video_rect.width() / (self.video_label.original_width or video_rect.width())
        scale_y = video_rect.height() / (self.video_label.original_height or video_rect.height())

        x1 = int(ann["x1"] * scale_x)
        y1 = int(ann["y1"] * scale_y)
        x2 = int(ann["x2"] * scale_x)
        y2 = int(ann["y2"] * scale_y)

        color = QColor(ann.get("color", self.drawing_color.name()))
        painter.setPen(QPen(color, 4))
        painter.drawRect(QRect(x1, y1, x2 - x1, y2 - y1))

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

    # -------------------------------------------------------------------------
    # EXPORTAÇÃO DE SEGMENTAÇÃO
    # -------------------------------------------------------------------------

    def export_segmentation_dialog(self):
        output_dir = QFileDialog.getExistingDirectory(self, self.texts.get("select_seg_output_dir", "Select output folder for segmentation dataset"))
        if output_dir:
            self.export_yolo_segmentation_annotations(output_dir)

    def export_yolo_segmentation_annotations(self, output_dir):
        if not hasattr(self, 'segmentation_annotations') or not self.segmentation_annotations:
            QMessageBox.warning(self, self.texts["warning"], self.texts.get("no_seg_annotations", "No segmentation annotations to export"))
            return 0

        try:
            images_dir = Path(output_dir) / "images"
            labels_dir = Path(output_dir) / "labels"
            for subdir in ["train", "val"]:
                (images_dir / subdir).mkdir(parents=True, exist_ok=True)
                (labels_dir / subdir).mkdir(parents=True, exist_ok=True)

            existing_classes = self._load_existing_classes(Path(output_dir) / "dataset.yaml")
            new_classes = set(ann["class"] for ann in self.segmentation_annotations)
            merged = dict(existing_classes)
            next_id = max(merged.values()) + 1 if merged else 0
            for cls in new_classes:
                if cls not in merged:
                    merged[cls] = next_id
                    next_id += 1
            class_to_id = merged

            frames_dict = defaultdict(list)
            for ann in self.segmentation_annotations:
                frames_dict[ann.get("frame_number", 0)].append(ann)

            is_dataset_mode = self.dataset_mode and self.dataset_frames

            # >>> CORREÇÃO: exportar TODOS os frames no modo dataset (incluindo backgrounds) <<<
            if is_dataset_mode:
                all_frames = list(range(len(self.dataset_frames)))
                train_frames, val_frames = set(), set()
                for fn in all_frames:
                    if fn < len(self.dataset_frames):
                        orig_split = Path(self.dataset_frames[fn][0]).parent.name
                        if orig_split == 'val':
                            val_frames.add(fn)
                        else:
                            train_frames.add(fn)
            else:
                all_frames = sorted(frames_dict.keys())
                train_frames, val_frames = self._split_frames_for_export(all_frames, is_dataset_mode)

            if not all_frames:
                return 0

            index_offset = self._detect_max_existing_index(images_dir)
            exported_count = self._export_segmentation_frames(
                all_frames, frames_dict, images_dir, labels_dir, class_to_id,
                is_dataset_mode, train_frames, val_frames, index_offset
            )

            self._write_dataset_yaml(Path(output_dir) / "dataset.yaml", output_dir, class_to_id)
            self._show_segmentation_export_summary(images_dir, labels_dir, exported_count, class_to_id)
            return exported_count

        except Exception as e:
            QMessageBox.critical(self, self.texts.get("error_title", "Error"), self.texts.get("export_failed_format", "Export failed: {}").format(str(e)))
            return 0

    def _split_frames_for_export(self, all_frames, is_dataset_mode):
        """Divide frames entre train e val."""
        if is_dataset_mode:
            train_frames = set()
            val_frames = set()
            for fn in all_frames:
                if fn < len(self.dataset_frames):
                    split = Path(self.dataset_frames[fn][0]).parent.name
                    (train_frames if split == 'train' else val_frames).add(fn)
                else:
                    (train_frames if (fn % 10) < 8 else val_frames).add(fn)
            return train_frames, val_frames
        else:
            shuffled = all_frames.copy()
            random.shuffle(shuffled)
            split_idx = int(0.8 * len(shuffled))
            return set(shuffled[:split_idx]), set(shuffled[split_idx:])

    def _detect_max_existing_index(self, images_dir):
        """Detecta maior índice existente nos diretórios de imagens."""
        max_index = -1
        for split in ['train', 'val']:
            split_dir = images_dir / split
            if split_dir.exists():
                for img_file in split_dir.glob('frame_*.jpg'):
                    try:
                        num = int(img_file.stem.split('_')[1])
                        max_index = max(max_index, num)
                    except (IndexError, ValueError):
                        continue
        return max_index + 1

    def _export_segmentation_frames(self, all_frames, frames_dict, images_dir, labels_dir,
                                    class_to_id, is_dataset_mode, train_frames, val_frames, index_offset):
        """Exporta frames e labels de segmentação."""
        progress = QProgressDialog(self.texts["exporting_frames"], self.texts["cancel"], 0, len(all_frames), self)
        progress.setWindowTitle(self.texts["exporting_dataset"])
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        exported_count = 0
        for i, frame_num in enumerate(all_frames):
            progress.setValue(i)
            QApplication.processEvents()
            if progress.wasCanceled():
                break

            frame, img_name, split = self._get_segmentation_frame(
                frame_num, is_dataset_mode, train_frames, val_frames, index_offset, images_dir
            )
            if frame is None:
                continue

            img_subdir = "train" if split else "val"
            img_path = images_dir / img_subdir / img_name
            if not img_path.exists():
                cv2.imwrite(str(img_path), frame)

            # >>>  só cria label se houver anotações para este frame <<<
            if frame_num in frames_dict and frames_dict[frame_num]:
                label_name = Path(img_name).stem + ".txt"
                label_path = labels_dir / img_subdir / label_name
                existing_lines = []
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        existing_lines = [line.strip() for line in f if line.strip()]

                new_lines = []
                for ann in frames_dict[frame_num]:
                    class_id = class_to_id[ann["class"]]
                    coords = ann["polygon"]
                    line = f"{class_id} " + " ".join([f"{x:.6f} {y:.6f}" for x, y in coords])
                    new_lines.append(line)

                with open(label_path, 'w') as f:
                    for line in existing_lines + new_lines:
                        f.write(line + "\n")

            exported_count += 1

        progress.close()
        return exported_count

    def _get_segmentation_frame(self, frame_num, is_dataset_mode, train_frames, val_frames, index_offset, images_dir):
        """Obtém frame para exportação de segmentação."""
        if is_dataset_mode and frame_num < len(self.dataset_frames):
            dataset_path, _, _ = self.dataset_frames[frame_num]
            frame = cv2.imread(dataset_path)
            if frame is None:
                return None, None, None
            # >>> CORREÇÃO: usar nome original da imagem e manter split <<<
            img_name = Path(dataset_path).name
            orig_split = Path(dataset_path).parent.name
            is_train = orig_split == 'train' if orig_split in ('train', 'val') else True
            return frame, img_name, is_train
        elif self.cap and self.cap.isOpened():
            current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
            if ret:
                img_name = f"frame_{frame_num:06d}.jpg"
                is_train = frame_num in train_frames
                return frame, img_name, is_train
        return None, None, None

    def _show_segmentation_export_summary(self, images_dir, labels_dir, exported_count, class_to_id):
        """Exibe resumo da exportação de segmentação."""
        train_imgs = len(list((images_dir / "train").glob("*.jpg")))
        val_imgs = len(list((images_dir / "val").glob("*.jpg")))
        train_labels = len(list((labels_dir / "train").glob("*.txt")))
        val_labels = len(list((labels_dir / "val").glob("*.txt")))

        QMessageBox.information(self, self.texts.get("export_completed", "Export Completed"),
            f"Segmentation dataset exported!\n\n"
            f"Train: {train_imgs} images, {train_labels} labels\n"
            f"Val: {val_imgs} images, {val_labels} labels\n"
            f"Total annotations: {len(self.segmentation_annotations)}\n"
            f"Classes: {list(class_to_id.keys())}")


    # -------------------------------------------------------------------------
    # TREINAMENTO DE SEGMENTAÇÃO
    # -------------------------------------------------------------------------

    def train_segmentation_model(self):
        has_annotations = hasattr(self, 'segmentation_annotations') and len(self.segmentation_annotations) > 0

        if not has_annotations:
            reply = QMessageBox.question(
                self, "No Segmentations in Memory",
                "No segmentation annotations in current session.\nUse existing dataset folder?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return

            dataset_dir = QFileDialog.getExistingDirectory(self, self.texts.get("select_seg_dataset_folder", "Select segmentation dataset folder"))
            if not dataset_dir:
                return
            yaml_path = Path(dataset_dir) / "dataset.yaml"
            if not yaml_path.exists():
                QMessageBox.warning(self, self.texts.get("error_title", "Error"), self.texts.get("no_dataset_yaml", "No dataset.yaml found in selected folder."))
                return
            name, ok = QInputDialog.getText(self, self.texts.get("seg_model_name_title", "Model Name"), self.texts.get("enter_model_name", "Enter model name:"))
            if not ok or not name:
                return
        else:
            dataset_dir = QFileDialog.getExistingDirectory(self, self.texts.get("select_seg_save_folder", "Select folder to save segmentation dataset"))
            if not dataset_dir:
                return
            exported = self.export_yolo_segmentation_annotations(dataset_dir)
            if exported == 0:
                QMessageBox.warning(self, self.texts.get("error_title", "Error"), self.texts.get("seg_export_failed", "Failed to export dataset"))
                return
            name, ok = QInputDialog.getText(self, self.texts.get("seg_model_name_title", "Model Name"), self.texts.get("enter_model_name", "Enter model name:"))
            if not ok or not name:
                return

        yaml_path = Path(dataset_dir) / "dataset.yaml"
        train_config = {
            "data": str(yaml_path),
            "epochs": 50, "imgsz": 640, "batch": 8,
            "name": f"seg_model_{name}",
            "device": "0" if torch.cuda.is_available() else "cpu",
            "exist_ok": True, "patience": 20,
        }

        progress = QProgressDialog(self.texts.get("training_seg_model", "Training segmentation model..."), self.texts.get("cancel", "Cancel"),
                                     0, train_config["epochs"], self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        self.train_seg_thread = TrainSegmentationThread(train_config)
        self.train_seg_thread.epoch_progress.connect(progress.setValue)
        self.train_seg_thread.finished.connect(lambda: self.on_seg_training_finished(progress))
        self.train_seg_thread.start()

    def on_seg_training_finished(self, progress):
        progress.close()
        if self.train_seg_thread.success:
            QMessageBox.information(self, self.texts.get("training_complete_title", "Training Complete"),
                f"Segmentation model saved to:\n{self.train_seg_thread.model_path}")
        else:
            QMessageBox.critical(self, self.texts.get("training_failed_title", "Training Failed"), self.train_seg_thread.error)

    # -------------------------------------------------------------------------
    # CARREGAMENTO DE ANOTAÇÕES
    # -------------------------------------------------------------------------

    def load_annotations_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, self.texts["select_annotations_file"], "", CSV_EXTS)
        if file_path:
            self.load_annotations(file_path)

    def load_annotations(self, file_path):
        try:
            if not file_path.endswith('.csv'):
                raise ValueError(f"Unsupported format. Use .csv files only (got: {os.path.splitext(file_path)[1]})")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Annotation file not found: {file_path}")
            if os.path.getsize(file_path) == 0:
                raise ValueError("CSV file is empty")

            annotations_list = []
            with open(file_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                required_cols = ['Frame_Number', 'Taxon', 'x1', 'y1', 'x2', 'y2', 'Type']
                missing = [col for col in required_cols if col not in reader.fieldnames]
                if missing:
                    raise ValueError(f"Missing required CSV columns: {missing}")

                for row in reader:
                    annotation = {
                        'frame': int(row['Frame_Number']),
                        'timestamp': row.get('Timestamp', ''),
                        'class': row['Taxon'],
                        'confidence': float(row.get('Confidence', 1.0)),
                        'type': row['Type'],
                        'track_id': row.get('Track_ID', ''),
                        'bbox': [int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])],
                        'photo_path': row.get('Photo', '')
                    }
                    annotations_list.append(annotation)

            self.annotations = []
            self.video_label.manual_annotations = []
            self.detections_dock.all_detections = []
            self.detections_dock.detections_list.clear()

            for ann in annotations_list:
                if ann['type'] == 'manual':
                    self.video_label.manual_annotations.append(ann)
                else:
                    self.annotations.append(ann)
                self.detections_dock.add_detection(ann)

            self.custom_classes = list(set(a['class'] for a in annotations_list))
            self.set_status_message("annotations_loaded", os.path.basename(file_path))
            self.detections_dock.update_class_filter()

            if self.cap is not None and self.cap.isOpened():
                self.display_frame(self.video_label._pixmap)
            self.detections_dock.apply_filters()

        except Exception as e:
            self.show_error_message("error", "load_annotations_error", str(e))
            self.set_status_message("load_annotations_status_error", str(e))

    # -------------------------------------------------------------------------
    # MERGE DE ANOTAÇÕES COM GEOREFERENCIAMENTO
    # -------------------------------------------------------------------------

    def merge_annotations(self):
        self.show_info_message("warning", "select_annotations_file")
        file_path1, _ = QFileDialog.getOpenFileName(
            self, self.texts["select_annotations_file"], "", CSV_EXTS)
        if not file_path1:
            return

        self.show_info_message("warning", "select_georeferencing_file")
        file_path2, _ = QFileDialog.getOpenFileName(
            self, self.texts["select_georeferencing_file"], "", CSV_EXTS)
        if not file_path2:
            return

        try:
            df1 = self._robust_read_csv(file_path1)
            df2 = self._robust_read_csv(file_path2)

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

            df1["time_delta"] = pd.to_timedelta(df1[col_left])
            df2["time_delta"] = pd.to_timedelta(df2[col_right])
            df1 = df1.sort_values("time_delta")
            df2 = df2.sort_values("time_delta")

            merged = pd.merge_asof(
                df1, df2, left_on="time_delta", right_on="time_delta",
                direction="nearest", tolerance=pd.Timedelta("1s")
            )

            base_name = os.path.splitext(os.path.basename(file_path1))[0]
            save_path, _ = QFileDialog.getSaveFileName(
                self, self.texts["save_merged_annotations"], f"{base_name}_georreferenciado.csv", CSV_EXTS)
            if not save_path:
                return

            merged.to_csv(save_path, index=False, encoding="utf-8")
            self.show_info_message("success", "merge_completed", save_path)

        except Exception as e:
            self.show_error_message("error", "merge_error", str(e))

    def _robust_read_csv(self, file_path):
        try:
            return pd.read_csv(file_path, encoding='utf-8', sep=',')
        except Exception:
            return pd.read_csv(file_path, encoding='latin1', sep=',')

    @staticmethod
    def parse_time_string(t: str) -> timedelta:
        try:
            return datetime.strptime(t, "%H:%M:%S.%f") - datetime.strptime("00:00:00.000", "%H:%M:%S.%f")
        except ValueError:
            return datetime.strptime(t, "%H:%M:%S") - datetime.strptime("00:00:00", "%H:%M:%S")

    # -------------------------------------------------------------------------
    # ANOTAÇÃO MANUAL - HISTÓRICO
    # -------------------------------------------------------------------------

    def add_manual_annotation_to_history(self, annotation):
        if not hasattr(self, 'all_detections'):
            self.all_detections = []

        if "system_date" not in annotation:
            system_info = self.get_system_timestamp()
            annotation["system_date"] = system_info['system_date']
            annotation["system_time"] = system_info['system_time']

        annotation.setdefault("video_path", self.video_path or "Live")
        annotation.setdefault("frame_number", self.current_frame_num)
        annotation.setdefault("timestamp", self.get_video_timestamp(self.current_frame_num))
        if not annotation.get("type"):
            annotation["type"] = "manual"
        annotation.setdefault("confidence", 1.0)

        # Evita duplicados
        is_dup = any(
            existing.get("x1") == annotation.get("x1") and
            existing.get("y1") == annotation.get("y1") and
            existing.get("x2") == annotation.get("x2") and
            existing.get("y2") == annotation.get("y2") and
            existing.get("frame_number") == annotation.get("frame_number") and
            existing.get("class") == annotation.get("class")
            for existing in self.all_detections
        )

        if not is_dup:
            self.all_detections.append(annotation)

        if self.recording:
            self._save_manual_annotation_recording(annotation)

        self.detections_dock.add_detection(annotation)
        self.refresh_frame_display()

    def _save_manual_annotation_recording(self, annotation):
        """Salva anotação manual durante gravação."""
        manual_annotation = annotation.copy()
        if manual_annotation.get("track_id") is None:
            manual_annotation["track_id"] = f"manual_{len(self.recorded_detections)}_{self.current_frame_num}"

        if hasattr(self, 'current_frame') and self.current_frame is not None:
            video_name = Path(self.recording_filename).stem if hasattr(self, 'recording_filename') else "recording"
            frames_dir = Path(self.recording_filename).parent / f"{video_name}_frames" if hasattr(self, 'recording_filename') else Path("recording_frames")
            frames_dir.mkdir(exist_ok=True)

            timestamp_str = datetime.now().strftime("%H%M%S%f")
            frame_filename = f"{video_name}_manual_{manual_annotation['track_id']}_{self.current_frame_num:06d}_{timestamp_str}.jpg"
            frame_path = frames_dir / frame_filename
            cv2.imwrite(str(frame_path), self.current_frame)
            manual_annotation["frame_path"] = str(frame_path)
            manual_annotation["frame_saved_at"] = datetime.now().isoformat()

        self.recorded_detections.append(manual_annotation)

    def refresh_frame_display(self):
        self.video_label.current_frame_num = self.current_frame_num
        self.video_label.update_active_annotations()
        self.video_label.update()

    def remove_annotation_from_history(self, ann):
        """Remove uma anotação do histórico principal e do dock."""
        if hasattr(self, 'all_detections'):
            bbox_id = ann.get("bbox_id")
            frame_num = ann.get("frame_number")
            self.all_detections = [
                d for d in self.all_detections
                if not (d.get("bbox_id") == bbox_id and d.get("frame_number") == frame_num)
            ]
        if self.detections_dock:
            self.detections_dock.remove_detection(ann)

    # -------------------------------------------------------------------------
    # HELP E ABOUT
    # -------------------------------------------------------------------------

    def show_shortcuts(self):
        shortcuts = [
            f"{self.texts['space']}: {self.texts['pause']}",
            f"Setas: {self.texts['navigate_frames']}",
            f"D: {self.texts['detect_frame']}",
            f"T: {self.texts['toggle_detection']}",
            f"M: {self.texts['annotate_manual']}",
            f"A: {self.texts['segment_with_sam2']}",
            f"E: {self.texts['enrich_taxonomy']}",
            f"Ctrl+O: {self.texts['load_video']}",
            f"Ctrl+M: {self.texts['load_model']}",
            f"Ctrl+Y: {self.texts['load_yaml']}",
            f"Ctrl+W: {self.texts['live']}",
            f"Ctrl+S: {self.texts['save_annotations']}",
            f"Ctrl+R: {self.texts['start_recording']}",
            f"Ctrl+Shift+R: {self.texts['stop_recording']}",
            f"Ctrl+Q: {self.texts['exit']}",
            f"Ctrl+L: {self.texts['load_annotations']}",
            f"Ctrl+H: {self.texts['history_show']}",
            f"Ctrl+T: {self.texts['taxon_show']}",
            f"F1: {self.texts['shortcuts']}"
        ]
        QMessageBox.information(self, self.texts["shortcuts"], "\n".join(shortcuts))

    def show_about(self):
        QMessageBox.about(self, self.texts["about"], self.texts["about_text"])

    def show_manual(self):
        manual_path = resource_path(f"manual/manual_{self.language}.pdf")
        QDesktopServices.openUrl(QUrl.fromLocalFile(manual_path))

    def open_enrichment_dialog(self):
        dialog = TaxonomyEnrichmentDialog(self)
        dialog.exec()

    def update_training_menu_state(self):
        """Atualiza estado do menu de treinamento (placeholder)."""
        pass

    # -------------------------------------------------------------------------
    # FECHAMENTO
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # MODO HEADLESS (SEM RENDERIZAÇÃO)
    # -------------------------------------------------------------------------

    def toggle_headless_mode(self, enabled):
        """Ativa/desativa modo de processamento sem renderização de vídeo."""
        self.headless_mode = enabled
        if enabled:
            self.video_label.setVisible(False)
            self.progress_slider.setVisible(False)
            self.current_time_label.setVisible(False)
            self.total_time_label.setVisible(False)
            self.info_panel.setVisible(True)
            self._update_info_panel(
                self.texts.get("headless_active", "⚡ Modo rápido ativo — processando sem vídeo"),
                "#FF9800"
            )
            self.headless_stats = {
                "frames_processed": 0,
                "detections_found": 0,
                "start_time": datetime.now(),
                "output_path": None,
                "all_detections": []
            }
        else:
            self.video_label.setVisible(True)
            self.progress_slider.setVisible(True)
            self.current_time_label.setVisible(True)
            self.total_time_label.setVisible(True)
            self._update_info_panel("")
            if self.current_frame is not None:
                self.display_frame(self.current_frame)

    def run_headless_video_processing(self):
        """Processa vídeo inteiro sem renderização, salvando CSV ao final."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.texts.get("select_video_headless", "Selecionar vídeo para processamento rápido"),
            "",
            self.texts.get("video_files_filter", "Vídeos") + " (*.mp4 *.avi *.mov *.mkv *.m4v *.flv *.wmv);;" +
            self.texts.get("all_files", "Todos os arquivos") + " (*)"
        )
        if not file_path:
            return

        if self.model is None:
            self.show_warning_message("warning", "no_model_loaded")
            return

        default_csv = f"{Path(file_path).stem}_detections.csv"
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            self.texts.get("save_annotations_dialog", "Salvar detecções"),
            default_csv,
            "CSV Files (*.csv);;All Files (*)"
        )
        if not output_path:
            return

        self.start_video(file_path)
        self.continuous_detection = True
        self.headless_mode = True
        self.headless_stats = {
            "frames_processed": 0,
            "detections_found": 0,
            "start_time": datetime.now(),
            "output_path": output_path,
            "all_detections": []
        }

        self.video_label.setVisible(False)
        self.progress_slider.setVisible(False)
        self.current_time_label.setVisible(False)
        self.total_time_label.setVisible(False)
        self.info_panel.setVisible(True)
        self._update_info_panel(
            self.texts.get("headless_active", "⚡ Modo rápido ativo — processando sem vídeo"),
            "#FF9800"
        )

        if self.paused:
            self.toggle_play_pause()

    def _save_headless_results(self):
        """Salva resultados acumulados do modo headless em CSV."""
        if not hasattr(self, 'headless_stats') or not self.headless_stats.get("output_path"):
            return

        output_path = self.headless_stats["output_path"]
        detections = self.headless_stats.get("all_detections", [])

        if detections:
            self._save_detections_csv(detections, output_path, self.video_path or "headless")
            self.status_label.setText(
                self.texts.get("headless_results_saved", "✓ Resultados salvos em: {}").format(output_path)
            )
        else:
            self.status_label.setText(
                self.texts.get("no_annotations_to_export", "Nenhuma anotação disponível para exportar.")
            )

        # Limpa para evitar salvar duplicado
        self.headless_stats["output_path"] = None

    # INFERÊNCIA EM BATCH (IMAGENS)
    # -------------------------------------------------------------------------

    def load_image_folder_for_inference(self):
        """Carrega uma pasta de imagens e executa inferência em batch.

        Pergunta se deseja usar um dataset.yaml existente para preservar IDs de classe.
        """
        folder = QFileDialog.getExistingDirectory(
            self, self.texts.get("select_image_folder", "Select Image Folder"))
        if not folder:
            return

        folder_path = Path(folder)

        # Verificar se é uma estrutura YOLO existente (tem images/train ou images/val)
        has_train = (folder_path / "images" / "train").exists()
        has_val = (folder_path / "images" / "val").exists()

        if has_train or has_val:
            # Estrutura YOLO existente - carregar para visualização
            self._load_yolo_folder_for_visualization(folder_path)
            return

        # Pasta plana - executar inferência
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
        image_files = []
        for f in sorted(folder_path.iterdir()):
            if f.is_file() and f.suffix.lower() in image_exts:
                image_files.append(str(f))

        if not image_files:
            QMessageBox.warning(self, self.texts.get("warning", "Warning"),
                "Nenhuma imagem encontrada na pasta selecionada.")
            return

        if self.model is None:
            self.show_warning_message("warning", "no_model_loaded")
            return

        # Perguntar se deseja usar dataset.yaml existente para preservar IDs
        class_mapping = None
        yaml_path = folder_path / "dataset.yaml"

        if yaml_path.exists():
            reply = QMessageBox.question(
                self,
                self.texts.get("use_existing_dataset", "Use Existing Dataset"),
                self.texts.get("use_yaml_for_ids", "Dataset.yaml encontrado na pasta.\n\n"
                    "Deseja usar os IDs de classe do dataset.yaml existente?\n\n"
                    "Sim = preserva IDs originais do dataset\n"
                    "Não = usa IDs do modelo atual"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
            )

            if reply == QMessageBox.StandardButton.Cancel:
                return

            if reply == QMessageBox.StandardButton.Yes:
                try:
                    with open(yaml_path, 'r', encoding='utf-8') as f:
                        import yaml
                        data = yaml.safe_load(f)

                    if data and 'names' in data:
                        names = data['names']
                        if isinstance(names, dict):
                            class_mapping = {str(name): int(idx) for idx, name in names.items()}
                        elif isinstance(names, list):
                            class_mapping = {str(name): i for i, name in enumerate(names)}

                        self.status_label.setText(
                            f"IDs de classe carregados do dataset.yaml: {len(class_mapping)} classes")

                except Exception as e:
                    QMessageBox.warning(self, self.texts.get("warning", "Warning"),
                        f"Erro ao carregar dataset.yaml: {str(e)}\n\nUsando IDs do modelo.")

        self._run_batch_inference(image_files, folder, class_mapping)

    def _load_yolo_folder_for_visualization(self, folder_path: Path):
        """Carrega uma pasta com estrutura YOLO existente para visualização."""
        images_dir = folder_path / "images"
        labels_dir = folder_path / "labels"

        # Coletar imagens de train e val
        all_images = []
        splits = []
        for split in ["train", "val"]:
            split_dir = images_dir / split
            if split_dir.exists():
                for f in sorted(split_dir.iterdir()):
                    if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                        all_images.append(str(f))
                        splits.append(split)

        if not all_images:
            QMessageBox.warning(self, self.texts.get("warning", "Warning"),
                "Nenhuma imagem encontrada nas pastas train/val.")
            return

        # Carregar classes do dataset.yaml
        yaml_path = folder_path / "dataset.yaml"
        class_names = []
        class_to_id = {}
        if yaml_path.exists():
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    import yaml
                    data = yaml.safe_load(f)
                if data and 'names' in data:
                    names = data['names']
                    if isinstance(names, dict):
                        for idx, name in sorted(names.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0):
                            class_to_id[str(name)] = int(idx) if str(idx).isdigit() else idx
                            class_names.append(str(name))
                    elif isinstance(names, list):
                        class_names = [str(n) for n in names]
                        class_to_id = {name: i for i, name in enumerate(class_names)}
            except Exception as e:
                print(f"Erro ao carregar dataset.yaml: {e}")

        if not class_names:
            # Tentar inferir das labels
            all_labels = set()
            for split in ["train", "val"]:
                split_labels = labels_dir / split
                if split_labels.exists():
                    for label_file in split_labels.glob("*.txt"):
                        try:
                            with open(label_file, 'r') as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if parts and parts[0].isdigit():
                                        all_labels.add(int(parts[0]))
                        except:
                            pass
            if all_labels:
                max_id = max(all_labels)
                class_names = [f"class_{i}" for i in range(max_id + 1)]
                class_to_id = {name: i for i, name in enumerate(class_names)}

        # Construir dataset_frames e carregar anotações
        self.dataset_frames = []
        self.all_detections = []
        frame_annotations = {}

        for idx, img_path in enumerate(all_images):
            self.dataset_frames.append((img_path, idx, idx))

            # Carregar label correspondente
            img_path_obj = Path(img_path)
            split = splits[idx]
            label_file = labels_dir / split / f"{img_path_obj.stem}.txt"

            if label_file.exists():
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        h, w = img.shape[:2]
                        frame_dets = []

                        with open(label_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                parts = line.split()
                                if len(parts) < 5:
                                    continue

                                try:
                                    class_id = int(parts[0])
                                    coords = list(map(float, parts[1:]))
                                except ValueError:
                                    continue

                                # Verificar se é segmentação (mais de 4 valores) ou bbox (4 valores)
                                if len(coords) == 4:
                                    # BBOX
                                    x_center, y_center, bw, bh = coords
                                    x1 = int((x_center - bw / 2) * w)
                                    y1 = int((y_center - bh / 2) * h)
                                    x2 = int((x_center + bw / 2) * w)
                                    y2 = int((y_center + bh / 2) * h)

                                    x1, y1 = max(0, x1), max(0, y1)
                                    x2, y2 = min(w - 1, x2), min(h - 1, y2)

                                    if x2 <= x1 or y2 <= y1:
                                        continue

                                    class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"

                                    detection = {
                                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                        "class": class_name, "confidence": 1.0,
                                        "type": "manual", "frame_number": idx,
                                        "timestamp": "00:00:00",
                                        "video_path": img_path,
                                        "frame_source": (img_path, idx),
                                        "frame_dimensions": f"{w}x{h}",
                                        "bbox_id": str(uuid.uuid4())[:8]
                                    }
                                    frame_dets.append(detection)
                                    self.all_detections.append(detection)

                                elif len(coords) > 4 and len(coords) % 2 == 0:
                                    # SEGMENTAÇÃO (polígono)
                                    class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"

                                    # Converter para polygon absoluto
                                    polygon = []
                                    for i in range(0, len(coords), 2):
                                        px = int(coords[i] * w)
                                        py = int(coords[i + 1] * h)
                                        polygon.append((px, py))

                                    # Calcular bbox do polygon
                                    xs = [p[0] for p in polygon]
                                    ys = [p[1] for p in polygon]
                                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

                                    detection = {
                                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                        "class": class_name, "confidence": 1.0,
                                        "type": "segmentation", "frame_number": idx,
                                        "timestamp": "00:00:00",
                                        "video_path": img_path,
                                        "frame_source": (img_path, idx),
                                        "frame_dimensions": f"{w}x{h}",
                                        "bbox_id": str(uuid.uuid4())[:8],
                                        "polygon": [(y/h, x/w) for x, y in polygon],
                                    }
                                    frame_dets.append(detection)
                                    self.all_detections.append(detection)

                        if frame_dets:
                            frame_annotations[idx] = frame_dets

                except Exception as e:
                    print(f"Erro ao carregar label {label_file}: {e}")

        # Configurar modo dataset
        self._reset_dataset_state()
        self.dataset_mode = True
        self.dataset_index = 0

        # IMPORTANTE: repopular frame_annotations DEPOIS do reset
        self.video_label.frame_annotations = frame_annotations

        # Atualizar dock
        self.detections_dock.all_detections = self.all_detections.copy()
        self.detections_dock.apply_filters()

        # Atualizar taxon grid
        if class_names and self.taxon_grid:
            self.taxon_grid.clear()
            for cls in class_names:
                self.taxon_grid.add_taxon(cls)

        # Carregar primeiro frame
        self.load_dataset_frame(0)
        self.video_name_label.setText(
            self.texts.get("dataset_prefix", "[Dataset] {}").format(f"YOLO: {folder_path.name}"))

        total_train = len([s for s in splits if s == "train"])
        total_val = len([s for s in splits if s == "val"])

        QMessageBox.information(self,
            self.texts.get("dataset_ready_title", "Dataset Ready"),
            f"{len(self.dataset_frames)} imagens carregadas\n"
            f"Train: {total_train}, Val: {total_val}\n"
            f"{len(self.all_detections)} anotações\n"
            f"Classes: {', '.join(class_names) if class_names else 'N/A'}\n\n"
            f"Pasta: {folder_path}")

    def _run_batch_inference(self, image_files: list, source_folder: str, class_mapping: dict = None):
        """Executa inferência YOLO em lote sobre imagens e prepara modo dataset.

        Args:
            image_files: Lista de caminhos das imagens
            source_folder: Pasta de origem
            class_mapping: Mapeamento opcional {nome_classe: id} do dataset.yaml existente.
                          Se fornecido, usa esses IDs nas labels. Se não, usa IDs do modelo.

        Preserva os nomes originais das imagens nos arquivos de label gerados.
        """
        progress = QProgressDialog(
            self.texts.get("running_inference", "Running inference..."),
            self.texts.get("cancel", "Cancel"),
            0, len(image_files), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        # Preparar estrutura de saída temporária
        temp_dir = Path(source_folder) / "_batch_results"
        images_dir = temp_dir / "images" / "train"
        labels_dir = temp_dir / "labels" / "train"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        all_detections = []
        class_names = set()

        # Guardar frame_annotations temporariamente
        frame_annotations = {}

        for i, img_path in enumerate(image_files):
            progress.setValue(i)
            QApplication.processEvents()
            if progress.wasCanceled():
                break

            frame = cv2.imread(img_path)
            if frame is None:
                continue

            h, w = frame.shape[:2]

            # Usar nome original da imagem (sem extensão) para os arquivos de saída
            original_path = Path(img_path)
            original_stem = original_path.stem
            original_suffix = original_path.suffix

            # Inferência
            try:
                with torch.no_grad():
                    results = self.model(frame, conf=0.25, iou=0.45, verbose=False)
            except Exception as e:
                print(f"Erro na inferência de {img_path}: {e}")
                continue

            # Copiar imagem para pasta do dataset com nome original
            dst_img = images_dir / f"{original_stem}{original_suffix}"
            cv2.imwrite(str(dst_img), frame)

            # Gerar labels YOLO
            label_lines = []
            frame_dets = []

            if results and len(results) > 0 and results[0].boxes:
                for box in results[0].boxes:
                    model_cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = self.model.names[model_cls_id]
                    class_names.add(label)

                    # Determinar o class_id para a label
                    if class_mapping is not None and label in class_mapping:
                        # Usar ID do dataset.yaml existente
                        out_cls_id = class_mapping[label]
                    else:
                        # Usar ID do modelo
                        out_cls_id = model_cls_id

                    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())

                    # Normalizar para YOLO format
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h

                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    bw = max(0.0, min(1.0, bw))
                    bh = max(0.0, min(1.0, bh))

                    label_lines.append(f"{out_cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

                    detection = {
                        "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                        "class": label, "confidence": conf,
                        "type": "auto", "frame_number": i,
                        "timestamp": "00:00:00",
                        "video_path": str(dst_img),
                        "frame_source": (str(dst_img), i),
                        "frame_dimensions": f"{w}x{h}",
                        "bbox_id": str(uuid.uuid4())[:8]
                    }
                    frame_dets.append(detection)
                    all_detections.append(detection)

            # Salvar label file com nome original da imagem
            if label_lines:
                label_path = labels_dir / f"{original_stem}.txt"
                with open(label_path, 'w') as f:
                    for line in label_lines:
                        f.write(line + "\n")

            # Guardar anotações para repopular depois do reset
            if frame_dets:
                frame_annotations[i] = frame_dets

        progress.close()

        # Criar dataset.yaml
        if class_names:
            class_list = sorted(class_names)

            # Se temos class_mapping, usar a ordem do mapping original
            if class_mapping is not None:
                # Criar names a partir do class_mapping, mantendo IDs originais
                class_map = {}
                for name, cid in sorted(class_mapping.items(), key=lambda x: x[1]):
                    if name in class_names:
                        class_map[cid] = name
                # Adicionar classes novas que não estavam no mapping
                next_id = max(class_map.keys()) + 1 if class_map else 0
                for name in class_names:
                    if name not in class_mapping:
                        class_map[next_id] = name
                        next_id += 1
            else:
                class_map = {i: name for i, name in enumerate(class_list)}

            yaml_path = temp_dir / "dataset.yaml"
            with open(yaml_path, 'w') as f:
                f.write(f"# YOLO Dataset - Batch Inference Results\n")
                f.write(f"path: {str(temp_dir.absolute()).replace('\\', '/')}\n")
                f.write(f"train: images/train\n")
                f.write(f"val: images/train\n")
                f.write(f"nc: {len(class_map)}\n")
                f.write(f"names:\n")
                for idx, name in sorted(class_map.items()):
                    f.write(f"  {idx}: {name}\n")

        # Configurar modo dataset com os resultados
        self._setup_batch_dataset_mode(temp_dir, image_files, all_detections, frame_annotations)

        self.status_label.setText(
            f"Inferência concluída: {len(all_detections)} detecções em {len(image_files)} imagens")

    def _setup_batch_dataset_mode(self, temp_dir: Path, image_files: list, all_detections: list, frame_annotations: dict):
        """Configura o modo dataset para visualizar resultados do batch."""
        # Construir dataset_frames a partir das imagens processadas (mantendo nomes originais)
        images_dir = temp_dir / "images" / "train"
        self.dataset_frames = []

        # Ordenar pelos nomes originais para consistência
        img_files = sorted(images_dir.glob("*"))
        for idx, img_path in enumerate(img_files):
            self.dataset_frames.append((str(img_path), idx, idx))

        if not self.dataset_frames:
            QMessageBox.warning(self, self.texts.get("warning", "Warning"),
                "Nenhum frame processado.")
            return

        # Resetar estado anterior (limpa frame_annotations)
        self._reset_dataset_state()

        # IMPORTANTE: repopular frame_annotations DEPOIS do reset
        self.video_label.frame_annotations = frame_annotations

        # Configurar estado dataset
        self.dataset_mode = True
        self.dataset_index = 0
        self.all_detections = all_detections.copy()

        # Atualizar dock
        self.detections_dock.all_detections = self.all_detections.copy()
        self.detections_dock.apply_filters()

        # Atualizar taxon grid com classes detectadas
        detected_classes = sorted(set(d["class"] for d in all_detections))
        if self.taxon_grid:
            self.taxon_grid.clear()
            for cls in detected_classes:
                self.taxon_grid.add_taxon(cls)

        # Carregar primeiro frame
        self.load_dataset_frame(0)
        self.video_name_label.setText(
            self.texts.get("dataset_prefix", "[Dataset] {}").format(f"Batch: {len(image_files)} images"))

        QMessageBox.information(self,
            self.texts.get("dataset_ready_title", "Dataset Ready"),
            f"{len(self.dataset_frames)} imagens processadas\n"
            f"{len(all_detections)} detecções\n"
            f"Classes: {', '.join(detected_classes)}\n\n"
            f"Dataset salvo em: {temp_dir}")

    def export_batch_results(self):
        """Exporta resultados do batch para pasta permanente."""
        if not self.dataset_mode or not self.dataset_frames:
            self.show_warning_message("warning", "no_images_loaded_warning")
            return

        output_dir = QFileDialog.getExistingDirectory(
            self, self.texts.get("select_output_dir", "Select Output Folder"))
        if not output_dir:
            return

        output_path = Path(output_dir)
        images_dir = output_path / "images" / "train"
        labels_dir = output_path / "labels" / "train"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Copiar imagens e labels do temp_dir
        temp_dir = None
        for path, _, _ in self.dataset_frames:
            p = Path(path)
            if temp_dir is None:
                temp_dir = p.parent.parent.parent

        if temp_dir and temp_dir.exists():
            src_images = temp_dir / "images" / "train"
            src_labels = temp_dir / "labels" / "train"
            if src_images.exists():
                for f in src_images.glob("*"):
                    shutil.copy2(str(f), str(images_dir / f.name))
            if src_labels.exists():
                for f in src_labels.glob("*.txt"):
                    shutil.copy2(str(f), str(labels_dir / f.name))
            yaml_src = temp_dir / "dataset.yaml"
            if yaml_src.exists():
                shutil.copy2(str(yaml_src), str(output_path / "dataset.yaml"))

        self.show_info_message("success_title", "batch_results_exported", str(output_path))

    def closeEvent(self, event):
        if self.sam2_thread is not None:
            self.sam2_thread.stop()
            self.sam2_thread.wait(2000)

        if self.detection_thread is not None:
            self.detection_thread.stop()
            self.detection_thread.wait(2000)

        if hasattr(self, 'train_thread') and self.train_thread is not None:
            if self.train_thread.isRunning():
                self.train_thread.terminate()
                self.train_thread.wait(2000)

        if self.seafloor_thread:
            self.seafloor_thread.stop()
            self.seafloor_thread.wait(1000)

        if self.current_seafloor_class:
            self._save_seafloor_annotation_segment()


        # Salvar resultados headless se houver
        if hasattr(self, 'headless_stats') and self.headless_stats.get("output_path"):
            self._save_headless_results()
        event.accept()