import os
import cv2
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QComboBox, QDoubleSpinBox, QPushButton, QMessageBox, QVBoxLayout,
                            QListWidget, QGroupBox, QCompleter, QListWidgetItem, QDialog)
from PyQt6.QtGui import QIcon, QColor, QPixmap, QPainter, QPen, QImage
from PyQt6.QtCore import Qt
from .translations import TEXTS


class DetectionsDockWidget(QDockWidget):
    """Class for detection history with integrated filters"""
    def __init__(self, main_window):
        super().__init__(main_window.texts["history"], main_window)
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | 
                        Qt.DockWidgetArea.RightDockWidgetArea)
        self.language = "pt" 
        self.texts = TEXTS[self.language] 
        self.main = main_window

        # main widget 
        main_widget = QWidget()
        main_widget.setAutoFillBackground(True)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Group of filters
        self.filter_group = QGroupBox("Filtrar Detecções")
        filter_layout = QHBoxLayout(self.filter_group)
        
        # Labels with text that can be updated
        self.taxon_label = QLabel(self.texts["taxon"])
        self.confidence_label = QLabel(self.texts["confidence"])
        
        self.class_filter = QComboBox()
        self.class_filter.setEditable(True)
        self.class_filter.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.class_filter.addItem("")
        
        # Completer configuration
        completer = QCompleter(self.class_filter)
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.class_filter.setCompleter(completer)
        self.class_filter.completer().setModel(self.class_filter.model())
        
        # Confidence filter
        self.confidence_input = QDoubleSpinBox()
        self.confidence_input.setRange(0.0, 1.0)
        self.confidence_input.setValue(0.5)
        self.confidence_input.setSingleStep(0.1)
        self.confidence_input.setDecimals(2)
        
        # Filter button
        self.filter_button = QPushButton("Filtrar")
        self.filter_button.clicked.connect(self.apply_filters)
        
        # Add widgets to the layout using instance variables
        filter_layout.addWidget(self.taxon_label)
        filter_layout.addWidget(self.class_filter)
        filter_layout.addWidget(self.confidence_label)
        filter_layout.addWidget(self.confidence_input)
        filter_layout.addWidget(self.filter_button)
        
        # List of detections
        self.detections_list = QListWidget()
        self.detections_list.itemClicked.connect(self.show_detection_frame)
        self.detections_list.setStyleSheet("""
            QListWidget {
                font-size: 12px;
                background-color: #f0f0f0;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #ddd;
            }
            QListWidget::item:hover {
                background-color: #e0e0e0;
            }
            QListWidget::item:selected {
                background-color: #5c9eff;
                color: white;
            }
        """)
        
        # Main Layout
        main_layout.addWidget(self.filter_group)
        main_layout.addWidget(self.detections_list, 1)
        
        self.setWidget(main_widget)
        
        # Stores all detections for filtering
        self.all_detections = []
        self.max_visible = 16
        
        self.setStyleSheet("""
            QGroupBox {
                font-size: 12px;
                border: 1px solid #ccc;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
            QComboBox {
                padding: 3px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            QComboBox:editable {
                background: white;
            }
            QDoubleSpinBox {
                padding: 3px;
                border: 1px solid #ccc;
                border-radius: 4px;
                min-width: 60px;
            }
            QPushButton {
                padding: 3px 8px;
                background: #5c9eff;
                color: white;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #4a8ad4;
            }
        """)
 

    def update_class_filter(self):
        current_text = self.class_filter.currentText()
        self.class_filter.blockSignals(True)
        self.class_filter.clear()
        self.class_filter.addItem("")
        
        # Gets classes from the model if available
        if hasattr(self.parent(), 'model') and self.parent().model is not None:
            try:
                model_classes = sorted(list(self.parent().model.names.values()))
                for class_name in model_classes:
                    self.class_filter.addItem(class_name)
            except Exception as e:
                print(f"Erro ao obter classes do modelo: {str(e)}")
        
        # Adds custom classes
        if hasattr(self.parent(), 'custom_classes'):
            for custom_class in self.parent().custom_classes:
                if self.class_filter.findText(custom_class) == -1:  # Avoids duplicates
                    self.class_filter.addItem(custom_class)
        
        # Restores previous selection
        if current_text and current_text != "":
            index = self.class_filter.findText(current_text)
            if index >= 0:
                self.class_filter.setCurrentIndex(index)
        
        self.class_filter.blockSignals(False)

    def update_language(self, lang):

        self.language = lang
        self.texts = TEXTS[lang]
        
        self.setWindowTitle(self.texts["history"])
        self.filter_button.setText(self.texts["filter"])
        self.filter_group.setTitle(self.texts["filter_detection"])
        
        self.taxon_label.setText(self.texts["taxon"])
        self.confidence_label.setText(self.texts["confidence"])
    

    def add_detection(self, detection):
        if not isinstance(detection, dict):
            print("⚠️ Detecção inválida, ignorada")
            return
        
        if detection.get("type") == "manual":
            detection.setdefault("track_id", f"manual_{len(self.all_detections)}")
            detection.setdefault("confidence", 1.0)
            current_video_path = self.main.video_path if self.main.video_path else "Live"
            detection["video_path"] = current_video_path
        
        self.all_detections.append(detection)
        self.apply_filters()

    def filter_all(self, class_name="", min_conf=0.0):
        selected_class = class_name if class_name else self.class_filter.currentText()
        min_conf = min_conf if min_conf else self.confidence_input.value()

        best_detections = {}
        for d in self.all_detections:
            if selected_class and d.get("class") != selected_class:
                continue
            if d.get("type") == "auto" and d.get("confidence", 0) < min_conf:
                continue

            tid = d.get("track_id")
            if tid is not None:
                if tid not in best_detections or d.get("confidence", 0) > best_detections[tid].get("confidence", 0):
                    best_detections[tid] = d
            else:  # manual or without ID
                best_detections[f"manual_{id(d)}"] = d

        # returns list, without [-16:]
        return list(best_detections.values())

    def apply_filters(self):
        """Apply filters showing only the best detection by ID and sort by timestamp"""
        
        scroll_position = self.detections_list.verticalScrollBar().value()
        current_row = self.detections_list.currentRow()
        
        selected_class = self.class_filter.currentText()
        min_confidence = self.confidence_input.value()
        
        self.detections_list.clear()
        
        best_detections = {}
        filtered_detections = []
        
        for detection in self.all_detections:
            if selected_class != "" and detection.get("class") != selected_class:
                continue
                
            if detection.get("type") == "auto" and detection.get("confidence", 0) < min_confidence:
                continue
                
            track_id = detection.get("track_id")
            
            if track_id is not None:
                if detection.get("type") == "manual":
                    filtered_detections.append(detection)
                else:
                    if track_id not in best_detections or \
                    detection.get("confidence", 0) > best_detections[track_id].get("confidence", 0):
                        best_detections[track_id] = detection
        
        filtered_detections.extend(best_detections.values())
        
        def parse_timestamp(timestamp_str):
            if not timestamp_str:
                return 0
                    
            if ':' in timestamp_str and '-' not in timestamp_str:
                parts = timestamp_str.split(':')
                if len(parts) == 3:
                    return int(parts[0])*3600 + int(parts[1])*60 + float(parts[2])
                
            elif ' ' in timestamp_str:
                date_part, time_part = timestamp_str.split(' ')
                time_parts = time_part.split(':')
                if len(time_parts) >= 3:
                    return int(time_parts[0])*3600 + int(time_parts[1])*60 + float(time_parts[2])
                
            return 0
        
        try:
            filtered_detections.sort(key=lambda x: parse_timestamp(x.get("timestamp", "")))
        except Exception as e:
            print(f"Erro ao ordenar detecções: {e}")
            filtered_detections.sort(key=lambda x: x.get("timestamp", ""))

        current_video = os.path.basename(str(self.main.video_path or "Live"))
        # current video detections
        current_video_dets = [d for d in filtered_detections
                            if os.path.basename(str(d.get("video_path") or "Live")) == current_video]

        if current_video_dets:
            visible = current_video_dets[-self.max_visible:]
        else:
            visible = filtered_detections[-self.max_visible:]

        self.detections_list.clear()
        for detection in visible:
            self.add_detection_to_list(detection)

        self.detections_list.verticalScrollBar().setValue(scroll_position)
        if 0 <= current_row < self.detections_list.count():
            self.detections_list.setCurrentRow(current_row)

        return visible

    def add_detection_to_list(self, detection):

        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, detection)

        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(5, 2, 10, 5)
        layout.setSpacing(5)

        # Detection text
        class_name = detection.get("class", "Desconhecido")
        confidence = detection.get("confidence", 0)
        timestamp = detection.get("timestamp", "")
        track_id = detection.get("track_id")

        # Fix video_path = None issue in live mode
        current_video_path = self.main.video_path if self.main.video_path else "Live"
        detection_video_path = detection.get("video_path", "Desconhecido")
        
        if detection_video_path is None:
            detection_video_path = "Live"
        
        current_video = os.path.basename(str(current_video_path))
        video_name = os.path.basename(str(detection_video_path))

        # Assemble text
        text = f"{timestamp} - {class_name}" if timestamp else class_name
        if detection.get("type") == "manual":
            text += " (manual)"
        else:
            text += f" ({confidence:.2f})" if confidence else ""
            if track_id is not None:
                text += f" (ID: {track_id})"
                
        label = QLabel(text)
        label.setStyleSheet("font-size: 12px;")


        delete_btn = QPushButton()
        delete_btn.setToolTip(self.texts["delete_detection"])
        delete_btn.setIcon(QIcon("icons/delete_icon.png"))
        delete_btn.setFixedSize(30, 30)
        delete_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                padding: 4px;
            }
            QPushButton:hover {
                background: transparent;
            }
        """)
        delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)

        delete_btn.clicked.connect(lambda _=None, btn=delete_btn: self.delete_single_detection(btn))

        layout.addWidget(label, 1)  
        layout.addWidget(delete_btn, 0, Qt.AlignmentFlag.AlignRight)

        widget.setLayout(layout)
        widget.adjustSize()
        item.setSizeHint(widget.sizeHint())
        widget.setMinimumHeight(40)
        
        self.detections_list.addItem(item)
        self.detections_list.setItemWidget(item, widget)


    def delete_single_detection(self, button):
 
        for i in range(self.detections_list.count()):
            item = self.detections_list.item(i)
            widget = self.detections_list.itemWidget(item)
            if widget and widget.findChild(QPushButton) is button:
                detection = item.data(Qt.ItemDataRole.UserRole)

                reply = QMessageBox.question(
                    self, self.texts["confirm_deletion"],
                    self.texts["deletion_question"],
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return

                # Removes from visual list
                self.detections_list.takeItem(i)

                # Removes from internal list 
                if detection in self.all_detections:
                    self.all_detections.remove(detection)

                break
            
    def remove_detection(self, detection):
        if detection in self.all_detections:
            self.all_detections.remove(detection)
            self.apply_filters() 
            
    def set_dark_mode(self, enable=True):
        if enable:
            self.setStyleSheet("""
                QDockWidget {
                    background-color: #2d2d2d;
                    color: #ffffff;
                    border: 1px solid #444;
                }

                QGroupBox {
                    font-size: 12px;
                    color: #ffffff;
                    background-color: #353535;
                    border: 1px solid #555;
                    border-radius: 4px;
                    margin-top: 10px;
                    padding-top: 15px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 3px;
                    color: #ffffff;
                }
                
                QListWidget {
                    background-color: #2d2d2d;
                    color: #ffffff;
                    border: 1px solid #555;
                    border-radius: 4px;
                    font-size: 12px;
                    alternate-background-color: #353535;
                }
                QListWidget::item {
                    background-color: #353535;
                    color: #ffffff;
                    padding: 5px;
                    border-bottom: 1px solid #444;
                }
                QListWidget::item:hover {
                    background-color: #454545;
                }
                QListWidget::item:selected {
                    background-color: #2a82da;
                    color: white;
                }
                
                QComboBox {
                    background-color: #353535;
                    color: #ffffff;
                    border: 1px solid #555;
                    border-radius: 4px;
                    padding: 3px;
                    min-width: 80px;
                }
                QComboBox QAbstractItemView {
                    background-color: #353535;
                    color: #ffffff;
                    selection-background-color: #2a82da;
                }
                QComboBox:editable {
                    background: #353535;
                }
                
                QDoubleSpinBox {
                    background-color: #353535;
                    color: #ffffff;
                    border: 1px solid #555;
                    border-radius: 4px;
                    padding: 3px;
                    min-width: 60px;
                }
                
                QPushButton {
                    background-color: #5c9eff;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 3px 8px;
                }
                QPushButton:hover {
                    background-color: #4a8ad4;
                }
                
                QLabel {
                    color: #ffffff;
                    background-color: transparent;
                }
            """)
            
            self.detections_list.setStyleSheet("""
                QListWidget {
                    background-color: #2d2d2d;
                    color: #ffffff;
                    border: 1px solid #555;
                    border-radius: 4px;
                    font-size: 12px;
                }
                QListWidget::item {
                    background-color: #353535;
                    color: #ffffff;
                    padding: 5px;
                    border-bottom: 1px solid #444;
                }
                QListWidget::item:hover {
                    background-color: #454545;
                }
                QListWidget::item:selected {
                    background-color: #2a82da;
                    color: white;
                }
            """)
        else:
            # light mode
            self.setStyleSheet("""
                QDockWidget {
                    background-color: #f0f0f0;
                    color: #000000;
                    border: 1px solid #ccc;
                }
                
                QGroupBox {
                    font-size: 12px;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    margin-top: 10px;
                    padding-top: 15px;
                    background-color: white;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 3px;
                    color: #000000;
                }
                
                QListWidget {
                    background-color: #f0f0f0;
                    color: #000000;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    font-size: 12px;
                }
                QListWidget::item {
                    background-color: white;
                    color: #000000;
                    padding: 5px;
                    border-bottom: 1px solid #ddd;
                }
                QListWidget::item:hover {
                    background-color: #e0e0e0;
                }
                QListWidget::item:selected {
                    background-color: #5c9eff;
                    color: white;
                }
                
                QComboBox {
                    background-color: white;
                    color: #000000;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    padding: 3px;
                }
                QDoubleSpinBox {
                    background-color: white;
                    color: #000000;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    padding: 3px;
                }
                QPushButton {
                    background-color: #5c9eff;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 3px 8px;
                }
                QPushButton:hover {
                    background-color: #4a8ad4;
                }
                QLabel {
                    color: #000000;
                    background-color: transparent;
                }
            """)
            
            self.detections_list.setStyleSheet("""
                QListWidget {
                    font-size: 12px;
                    background-color: #f0f0f0;
                }
                QListWidget::item {
                    padding: 5px;
                    border-bottom: 1px solid #ddd;
                }
                QListWidget::item:hover {
                    background-color: #e0e0e0;
                }
                QListWidget::item:selected {
                    background-color: #5c9eff;
                    color: white;
                }
            """)
        
    def show_detection_frame(self, item):
        detection = item.data(Qt.ItemDataRole.UserRole)
        if not detection or "frame_source" not in detection:
            QMessageBox.information(self, "Sem imagem", "Frame não disponível para esta detecção.")
            return

        video_path, frame_num = detection["frame_source"]
        if not os.path.exists(video_path):
            QMessageBox.information(self, "Sem imagem", f"Vídeo não encontrado:\n{video_path}")
            return

        # loads frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            QMessageBox.information(self, "Sem imagem", f"Erro ao ler o frame {frame_num}.")
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # draws bbox
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if all(key in detection for key in ("x1", "y1", "x2", "y2")):
            x1 = int(detection["x1"])
            y1 = int(detection["y1"])
            x2 = int(detection["x2"])
            y2 = int(detection["y2"])

            # scale if necessary
            if "frame_dimensions" in detection:
                orig_w, orig_h = map(int, detection["frame_dimensions"].split("x"))
                if orig_w != w or orig_h != h:
                    scale_x, scale_y = w / orig_w, h / orig_h
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

            color = QColor(255, 0, 0)
            pen = QPen(color, 3, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            rect = QRect(x1, y1, x2 - x1, y2 - y1)
            painter.drawRect(rect)

            # text
            font = painter.font()
            font.setPixelSize(16)
            font.setBold(True)
            painter.setFont(font)
            label_text = detection.get("class", "Desconhecido")
            if "confidence" in detection:
                label_text += f" ({detection['confidence']:.2f})"
            if "track_id" in detection and detection["track_id"] is not None:
                label_text += f" ID:{detection['track_id']}"

            text_width = painter.fontMetrics().horizontalAdvance(label_text) + 10
            text_rect = QRect(x1, max(0, y1 - 20), text_width, 20)
            painter.fillRect(text_rect, color)
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.drawText(x1 + 5, max(5, y1 - 5), label_text)

        painter.end()

        # preview window
        title = f"{detection.get('class', 'Desconhecido')}"
        if "track_id" in detection and detection["track_id"] is not None:
            title += f"  ID:{detection['track_id']}"

        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(min(800, pixmap.width()), min(600, pixmap.height()))
        dialog.setWindowFlags(Qt.WindowType.Window)

        label = QLabel()
        label.setPixmap(pixmap)
        label.setScaledContents(True)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(label)
        dialog.exec()