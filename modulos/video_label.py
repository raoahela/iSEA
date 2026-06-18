import cv2
import uuid
import numpy as np
from PyQt6.QtCore import Qt, QPointF, QRect, QTimer, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QColor, QImage, QPolygonF
from PyQt6.QtWidgets import QLabel, QSizePolicy

class VideoLabel(QLabel):
    # Sinais para comunicação com o main window
    hover_point = pyqtSignal(int, int, int)  # x, y, frame_num (coordenadas do frame original)
    hover_cleared = pyqtSignal()
    mask_confirmed = pyqtSignal(dict)  # Máscara confirmada pelo clique

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.drawing_enabled = False
        self.hover_segmentation_enabled = False  # Modo hover-to-segment (SAM)
        self.current_class = None
        self.drawing_color = QColor(Qt.GlobalColor.green)
        self.active_annotations = []
        self.drawing = False
        self.delete_box_size = 16
        self.delete_box_offset = 4
        self.original_width = None
        self.original_height = None

        # Sistema de anotações por frame
        self.frame_annotations = {}
        self.current_frame_num = 0

        self.annotation_display_ms = 1000
        self.annotation_timers = {}

        # Mouse state
        self.mouse_state = "normal"
        self.start_point = QPointF()
        self.end_point = QPointF()

        # Visual configs
        self.handle_size = 8
        self.selection_color = QColor(255, 255, 0, 128)
        self.hover_color = QColor(255, 255, 255, 64)

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(1, 1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._aspect_ratio = None
        self.video_rect = None
        self._pixmap = None

        # Hover detection
        self.hovered_annotation = None

        # Preview de máscara do SAM (hover-to-segment)
        self.preview_mask = None
        self.preview_mask_color = QColor(0, 200, 255, 120)
        self.preview_mask_border = QColor(0, 200, 255, 220)
        self.preview_active = False
        self.last_hover_pos = None
        self.hover_debounce_timer = QTimer(self)
        self.hover_debounce_timer.setSingleShot(True)
        self.hover_debounce_timer.timeout.connect(self._process_hover)
        self.hover_delay_ms = 25  # MUDANÇA: era 80ms, agora 25ms

        # Máscaras confirmadas (clicadas)
        self.confirmed_masks = {}

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.update_display()

    def update_display(self):
        if self._pixmap is None:
            return

        scaled_pixmap = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.video_x = (self.width() - scaled_pixmap.width()) // 2
        self.video_y = (self.height() - scaled_pixmap.height()) // 2
        self.video_rect = QRect(self.video_x, self.video_y,
                              scaled_pixmap.width(), scaled_pixmap.height())

        super().setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        self.update_display()
        super().resizeEvent(event)

    def set_video_rect(self, rect):
        self.video_rect = rect

    def get_video_rect(self):
        return self.video_rect if self.video_rect else self.rect()

    def reset_annotations(self):
        self.active_annotations = []
        self.update()

    def _get_original_dimensions(self):
        main_window = self.window()
        original_width, original_height = 1920, 1080

        if hasattr(main_window, 'video_width') and hasattr(main_window, 'video_height'):
            original_width = main_window.video_width
            original_height = main_window.video_height
        elif hasattr(main_window, 'cap') and main_window.cap is not None:
            try:
                original_width = int(main_window.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                original_height = int(main_window.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            except:
                pass
        elif hasattr(main_window, 'current_frame') and main_window.current_frame is not None:
            original_height, original_width = main_window.current_frame.shape[:2]

        return original_width, original_height

    def _widget_to_frame_coords(self, widget_pos):
        video_rect = self.get_video_rect()
        if not video_rect or video_rect.width() == 0 or video_rect.height() == 0:
            return None

        rel_pos = widget_pos - video_rect.topLeft()
        original_width, original_height = self._get_original_dimensions()

        frame_x = int(rel_pos.x() * (original_width / video_rect.width()))
        frame_y = int(rel_pos.y() * (original_height / video_rect.height()))

        frame_x = max(0, min(frame_x, original_width - 1))
        frame_y = max(0, min(frame_y, original_height - 1))

        return frame_x, frame_y

    def mousePressEvent(self, event):
        if not self.drawing_enabled and not self.hover_segmentation_enabled:
            return

        video_rect = self.get_video_rect()
        pos = event.position().toPoint()
        if not video_rect.contains(pos):
            return

        if event.button() == Qt.MouseButton.LeftButton:
            # Check delete button
            for ann in self.active_annotations:
                if ann.get("delete_rect") and ann["delete_rect"].contains(pos):
                    self.remove_annotation(ann)
                    return

            # HOVER MODE: clique confirma a máscara preview
            if self.hover_segmentation_enabled and self.preview_active and self.preview_mask is not None:
                self._confirm_preview_mask()
                return

            # MANUAL DRAWING MODE
            if self.drawing_enabled:
                adjusted_pos = pos - video_rect.topLeft()
                video_size = video_rect.size()
                self.start_point = QPointF(adjusted_pos.x() / video_size.width(),
                                        adjusted_pos.y() / video_size.height())
                self.end_point = self.start_point
                self.drawing = True
                self.update()
                return

        if event.button() == Qt.MouseButton.RightButton:
            if self.hover_segmentation_enabled and self.preview_active:
                self._clear_preview()
                return
            self.delete_annotation_at(pos, video_rect)

    def _confirm_preview_mask(self):
        main_window = self.window()
        original_width, original_height = self._get_original_dimensions()

        # FILTRO: Rejeitar máscaras que cobrem mais de 80% do frame (provavelmente fundo)
        mask_area = np.sum(self.preview_mask > 0)
        total_area = self.preview_mask.shape[0] * self.preview_mask.shape[1]
        coverage = mask_area / total_area

        if coverage > 0.80:
            self._clear_preview()
            main_window.status_label.setText("SAM: Nenhum objeto detectado no ponto")
            return

        if coverage < 0.001:
            self._clear_preview()
            return

        from skimage import measure
        contours = measure.find_contours(self.preview_mask, 0.5)
        if not contours:
            return

        main_contour = max(contours, key=lambda c: len(c))

        h, w = self.preview_mask.shape[:2]
        normalized_coords = [(float(y)/h, float(x)/w) for x, y in main_contour]

        # Calcular bbox a partir da máscara para compatibilidade com o dock
        ys, xs = np.where(self.preview_mask > 0)
        if len(xs) == 0:
            self._clear_preview()
            return

        x1_px = int(np.min(xs))
        y1_px = int(np.min(ys))
        x2_px = int(np.max(xs))
        y2_px = int(np.max(ys))

        seg_annotation = {
            "type": "segmentation",
            "class": self.current_class or "unknown",
            "confidence": 1.0,
            "frame_number": main_window.current_frame_num,
            "timestamp": main_window.get_video_timestamp(main_window.current_frame_num),
            "video_path": main_window.video_path or "Live",
            "mask": self.preview_mask.copy(),
            "polygon": normalized_coords,
            "frame_source": (main_window.video_path, main_window.current_frame_num),
            "frame_dimensions": f"{original_width}x{original_height}",
            "bbox_id": str(uuid.uuid4())[:8],
            "source": "hover_sam",
            # Campos bbox para compatibilidade com dock/histórico
            "x1": x1_px,
            "y1": y1_px,
            "x2": x2_px,
            "y2": y2_px,
            "label": self.current_class or "unknown",
        }

        frame = main_window.current_frame_num
        if frame not in self.confirmed_masks:
            self.confirmed_masks[frame] = []
        self.confirmed_masks[frame].append(seg_annotation)

        # NÃO adicionar a frame_annotations para evitar duplicação no paintEvent
        # As segmentações são desenhadas separadamente via confirmed_masks
        # Apenas adicionar ao histórico via sinal

        self.mask_confirmed.emit(seg_annotation)

        self._clear_preview()

        if hasattr(main_window, 'add_manual_annotation_to_history'):
            main_window.add_manual_annotation_to_history(seg_annotation)

        self.update()

    def _clear_preview(self):
        self.preview_mask = None
        self.preview_active = False
        self.last_hover_pos = None
        self.hover_cleared.emit()
        self.update()

    def mouseMoveEvent(self, event):
        # HOVER MODE - sempre processar primeiro quando ativo
        if self.hover_segmentation_enabled:
            video_rect = self.get_video_rect()
            pos = event.position().toPoint()

            if video_rect and video_rect.contains(pos):
                frame_coords = self._widget_to_frame_coords(pos)
                if frame_coords:
                    self.last_hover_pos = frame_coords
                    self.hover_debounce_timer.stop()
                    self.hover_debounce_timer.start(self.hover_delay_ms)
            else:
                if self.preview_active:
                    self._clear_preview()
            return  # Importante: return aqui para não processar drawing

        # MANUAL DRAWING - só processa se hover não está ativo
        if self.drawing and self.drawing_enabled:
            video_rect = self.get_video_rect()
            if not video_rect:
                return

            current_pos = event.position().toPoint() - video_rect.topLeft()
            video_size = video_rect.size()

            x = max(0, min(current_pos.x() / video_size.width(), 1.0))
            y = max(0, min(current_pos.y() / video_size.height(), 1.0))
            self.end_point = QPointF(x, y)
            self.update()

    def _process_hover(self):
        if self.last_hover_pos and self.hover_segmentation_enabled:
            x, y = self.last_hover_pos
            # Usar o frame_num atual do VideoLabel (que deve estar sincronizado com VideoAnnotator)
            self.hover_point.emit(x, y, self.current_frame_num)

    def set_preview_mask(self, mask_data):
        if mask_data and "segmentation" in mask_data:
            mask = mask_data["segmentation"]

            # FILTRO: Não mostrar preview se máscara cobrir >80% do frame (fundo)
            mask_area = np.sum(mask > 0) if hasattr(mask, '__gt__') else np.sum(mask)
            total_area = mask.shape[0] * mask.shape[1]
            coverage = mask_area / total_area

            if coverage > 0.80:
                self.preview_mask = None
                self.preview_active = False
                self.last_hover_pos = None
                return

            self.preview_mask = mask
            self.preview_active = True
            self.update()
        else:
            pass

    def mouseReleaseEvent(self, event):
        main_window = self.window()

        if self.drawing and self.drawing_enabled and self.current_class:
            video_rect = self.get_video_rect()
            if not video_rect:
                return

            current_pos = event.position().toPoint() - video_rect.topLeft()
            video_size = video_rect.size()

            x = max(0, min(current_pos.x() / video_size.width(), 1.0))
            y = max(0, min(current_pos.y() / video_size.height(), 1.0))
            self.end_point = QPointF(x, y)
            self.drawing = False

            x1_norm = min(self.start_point.x(), self.end_point.x())
            y1_norm = min(self.start_point.y(), self.end_point.y())
            x2_norm = max(self.start_point.x(), self.end_point.x())
            y2_norm = max(self.start_point.y(), self.end_point.y())

            original_width, original_height = self._get_original_dimensions()
            self.original_width = original_width
            self.original_height = original_height

            min_size = 0.01
            if (x2_norm - x1_norm) > min_size and (y2_norm - y1_norm) > min_size:
                annotation = {
                    "x1": int(x1_norm * original_width),
                    "y1": int(y1_norm * original_height),
                    "x2": int(x2_norm * original_width),
                    "y2": int(y2_norm * original_height),
                    "class": self.current_class,
                    "timestamp": main_window.get_video_timestamp(main_window.current_frame_num),
                    "confidence": 1.0,
                    "color": self.drawing_color.name(),
                    "frame_number": main_window.current_frame_num,
                    "coordinates_type": "pixels",
                    "frame_dimensions": f"{original_width}x{original_height}",
                    "video_path": main_window.video_path or "Live",
                    "frame_source": (main_window.video_path or "Live", main_window.current_frame_num),
                    "bbox_id": str(uuid.uuid4())[:8]
                }

                if main_window.training_wizard is not None:
                    annotation["type"] = "training"
                else:
                    annotation["type"] = "manual"

                if main_window.current_frame_num not in self.frame_annotations:
                    self.frame_annotations[main_window.current_frame_num] = []
                self.frame_annotations[main_window.current_frame_num].append(annotation)

                self.active_annotations.append(annotation)

                if not main_window.paused:
                    bbox_id = annotation['bbox_id']
                    QTimer.singleShot(self.annotation_display_ms,
                                lambda: self.remove_annotation_completely(bbox_id))

                if hasattr(main_window, 'add_manual_annotation_to_history'):
                    main_window.add_manual_annotation_to_history(annotation)

            self.update()

    def update_active_annotations(self):
        # Carregar apenas anotações NÃO-segmentação do frame atual
        all_frame_anns = self.frame_annotations.get(self.current_frame_num, [])
        self.active_annotations = [
            ann for ann in all_frame_anns 
            if ann.get("type") != "segmentation"
        ].copy()
        self.update()

    def remove_annotation_completely(self, bbox_id: str):
        self.active_annotations = [
            a for a in self.active_annotations
            if a.get('bbox_id') != bbox_id
        ]

        frame = self.current_frame_num
        if frame in self.frame_annotations:
            self.frame_annotations[frame] = [
                a for a in self.frame_annotations[frame]
                if a.get('bbox_id') != bbox_id
            ]
            if not self.frame_annotations[frame]:
                del self.frame_annotations[frame]

        # Também remover de confirmed_masks se for segmentação
        if frame in self.confirmed_masks:
            self.confirmed_masks[frame] = [
                m for m in self.confirmed_masks[frame]
                if m.get('bbox_id') != bbox_id
            ]
            if not self.confirmed_masks[frame]:
                del self.confirmed_masks[frame]

        self.update()

    def remove_annotation(self, ann):
        main_window = self.window()
        frame = ann["frame_number"]
        bbox_id = ann.get("bbox_id")

        # Remover de active_annotations
        for i, a in enumerate(self.active_annotations):
            if a is ann:
                self.active_annotations.pop(i)
                break

        # Remover de frame_annotations
        if frame in self.frame_annotations:
            frame_anns = self.frame_annotations[frame]
            for i, a in enumerate(frame_anns):
                if a is ann:
                    frame_anns.pop(i)
                    break
            if not frame_anns:
                del self.frame_annotations[frame]

        # Remover de confirmed_masks (para segmentações)
        if frame in self.confirmed_masks:
            masks = self.confirmed_masks[frame]
            for i, m in enumerate(masks):
                if m.get("bbox_id") == bbox_id:
                    masks.pop(i)
                    break
            if not masks:
                del self.confirmed_masks[frame]

        # Remover do histórico principal
        if hasattr(main_window, 'segmentation_annotations'):
            indices_to_remove = []
            for i, seg in enumerate(main_window.segmentation_annotations):
                if seg.get("bbox_id") == bbox_id and seg.get("frame_number") == frame:
                    indices_to_remove.append(i)

            for i in reversed(indices_to_remove):
                seg = main_window.segmentation_annotations.pop(i)
                if hasattr(main_window.detections_dock, 'remove_detection'):
                    main_window.detections_dock.remove_detection(seg)

        if hasattr(main_window, 'remove_annotation_from_history'):
            main_window.remove_annotation_from_history(ann)
        elif hasattr(main_window.detections_dock, 'remove_detection'):
            main_window.detections_dock.remove_detection(ann)

        self.update()

    def delete_annotation_at(self, pos, video_rect):
        x = (pos.x() - video_rect.left()) / video_rect.width()
        y = (pos.y() - video_rect.top())  / video_rect.height()

        # Verificar anotações bbox primeiro
        for ann in reversed(self.active_annotations):
            if ann["x1"] <= x <= ann["x2"] and ann["y1"] <= y <= ann["y2"]:
                self.remove_annotation(ann)
                return

        # Verificar segmentações confirmadas
        frame = self.current_frame_num
        if frame in self.confirmed_masks:
            for mask_data in reversed(self.confirmed_masks[frame]):
                mask = mask_data.get("mask")
                if mask is not None:
                    # Converter posição do widget para coordenadas da máscara
                    original_width, original_height = self._get_original_dimensions()
                    scale_x = original_width / video_rect.width()
                    scale_y = original_height / video_rect.height()
                    mask_x = int(x * scale_x)
                    mask_y = int(y * scale_y)

                    h, w = mask.shape[:2]
                    if 0 <= mask_x < w and 0 <= mask_y < h:
                        if mask[mask_y, mask_x] > 0:
                            self.remove_annotation(mask_data)
                            return

    def remove_annotation_by_id(self, bbox_id: str):
        self.active_annotations = [
            a for a in self.active_annotations
            if a.get('bbox_id') != bbox_id
        ]

        frame = self.current_frame_num
        if frame in self.frame_annotations:
            self.frame_annotations[frame] = [
                a for a in self.frame_annotations[frame]
                if a.get('bbox_id') != bbox_id
            ]

        # Também limpar confirmed_masks
        if frame in self.confirmed_masks:
            self.confirmed_masks[frame] = [
                m for m in self.confirmed_masks[frame]
                if m.get('bbox_id') != bbox_id
            ]

        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        video_rect = self.get_video_rect()
        if not video_rect:
            return

        painter.setClipRect(video_rect)

        # Desenhar preview da máscara SAM
        if self.preview_active and self.preview_mask is not None:
            self._draw_preview_mask(painter, video_rect)

        # Desenhar máscaras confirmadas (segmentações)
        if hasattr(self, 'confirmed_masks') and self.confirmed_masks:
            for frame, masks in self.confirmed_masks.items():
                if frame == self.current_frame_num:
                    for mask_data in masks:
                        self._draw_confirmed_mask(painter, video_rect, mask_data)

        # Desenhar anotações ativas (bbox manuais) - EXCLUIR segmentações
        for ann in self.active_annotations:
            # Pular anotações de segmentação (são desenhadas separadamente via confirmed_masks)
            if ann.get("type") == "segmentation":
                continue

            # Verificar se tem coordenadas de bbox
            if not all(k in ann for k in ("x1", "y1", "x2", "y2")):
                continue

            color = QColor(ann.get("color", self.drawing_color.name()))
            painter.setPen(QPen(color, 4, Qt.PenStyle.SolidLine))

            original_width, original_height = self._get_original_dimensions()

            scale_x = video_rect.width() / original_width
            scale_y = video_rect.height() / original_height

            x1 = int(ann["x1"] * scale_x)
            y1 = int(ann["y1"] * scale_y)
            x2 = int(ann["x2"] * scale_x)
            y2 = int(ann["y2"] * scale_y)

            rect = QRect(
                video_rect.left() + x1,
                video_rect.top() + y1,
                x2 - x1,
                y2 - y1
            )
            painter.drawRect(rect)

            font = painter.font()
            font.setPixelSize(20)
            painter.setFont(font)
            text_x = video_rect.left() + x1
            text_y = video_rect.top() + max(video_rect.top(), y1 - 5)
            text_width = painter.fontMetrics().horizontalAdvance(ann["class"]) + 10
            text_height = painter.fontMetrics().height()
            text_rect = QRect(text_x - 5, max(0, text_y - 20),
                            text_width, text_height)
            painter.fillRect(text_rect, color)

            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.drawText(text_x, text_y, ann["class"])

            delete_rect = QRect(
                rect.right() - self.delete_box_size - self.delete_box_offset,
                rect.top() + self.delete_box_offset,
                self.delete_box_size,
                self.delete_box_size
            )

            painter.fillRect(delete_rect, QColor(255, 80, 80, 200))
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            x1_del, y1_del = delete_rect.left() + 4, delete_rect.top() + 4
            x2_del, y2_del = delete_rect.right() - 4, delete_rect.bottom() - 4
            painter.drawLine(x1_del, y1_del, x2_del, y2_del)
            painter.drawLine(x1_del, y2_del, x2_del, y1_del)

            ann["delete_rect"] = delete_rect

        # Desenhar caixa temporária
        if self.drawing and self.drawing_enabled:
            painter.setPen(QPen(self.drawing_color, 4, Qt.PenStyle.DashLine))

            x1 = int(self.start_point.x() * video_rect.width())
            y1 = int(self.start_point.y() * video_rect.height())
            x2 = int(self.end_point.x() * video_rect.width())
            y2 = int(self.end_point.y() * video_rect.height())

            rect = QRect(
                video_rect.left() + min(x1, x2),
                video_rect.top() + min(y1, y2),
                abs(x2 - x1),
                abs(y2 - y1)
            )
            painter.drawRect(rect)

        painter.end()

    def _draw_preview_mask(self, painter, video_rect):
        if self.preview_mask is None:
            return

        original_width, original_height = self._get_original_dimensions()

        # SAM retorna probabilidades (float 0-1), aplicar threshold para binarizar
        mask_binary = (self.preview_mask > 0.5).astype(np.uint8)

        mask_resized = cv2.resize(
            mask_binary,
            (video_rect.width(), video_rect.height()),
            interpolation=cv2.INTER_NEAREST
        )

        overlay = QImage(video_rect.width(), video_rect.height(), QImage.Format.Format_ARGB32)
        overlay.fill(Qt.GlobalColor.transparent)

        color = self.preview_mask_color
        border_color = self.preview_mask_border

        for y in range(mask_resized.shape[0]):
            for x in range(mask_resized.shape[1]):
                if mask_resized[y, x] > 0:
                    overlay.setPixelColor(x, y, color)

        painter.drawImage(video_rect.topLeft(), overlay)

        try:
            contours, _ = cv2.findContours(
                mask_resized,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            painter.setPen(QPen(border_color, 2, Qt.PenStyle.SolidLine))
            for contour in contours:
                points = [QPointF(
                    video_rect.left() + int(pt[0][0]),
                    video_rect.top() + int(pt[0][1])
                ) for pt in contour]
                if len(points) > 2:
                    painter.drawPolygon(QPolygonF(points))
        except:
            pass

        painter.setPen(QPen(border_color, 1))
        font = painter.font()
        font.setPixelSize(14)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(
            video_rect.left() + 10,
            video_rect.top() + 25,
            "Clique para confirmar | Botão direito para cancelar"
        )

    def _draw_confirmed_mask(self, painter, video_rect, mask_data):
        mask = mask_data.get("mask")
        if mask is None:
            return

        class_name = mask_data.get("class", "unknown")
        hue = hash(class_name) % 360
        import colorsys
        rgb = colorsys.hsv_to_rgb(hue/360, 0.8, 0.9)
        color = QColor(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255), 100)
        border_color = QColor(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255), 220)

        # Converter para uint8 se necessário (cv2.resize não aceita bool)
        if mask.dtype == np.float32 or mask.dtype == np.float64:
            mask = (mask > 0.5).astype(np.uint8)
        elif mask.dtype == bool:
            mask = mask.astype(np.uint8)

        mask_resized = cv2.resize(
            mask,
            (video_rect.width(), video_rect.height()),
            interpolation=cv2.INTER_NEAREST
        )

        overlay = QImage(video_rect.width(), video_rect.height(), QImage.Format.Format_ARGB32)
        overlay.fill(Qt.GlobalColor.transparent)

        for y in range(mask_resized.shape[0]):
            for x in range(mask_resized.shape[1]):
                if mask_resized[y, x] > 0:
                    overlay.setPixelColor(x, y, color)

        painter.drawImage(video_rect.topLeft(), overlay)

        try:
            contours, _ = cv2.findContours(
                mask_resized,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            painter.setPen(QPen(border_color, 2))
            for contour in contours:
                points = [QPointF(
                    video_rect.left() + int(pt[0][0]),
                    video_rect.top() + int(pt[0][1])
                ) for pt in contour]
                if len(points) > 2:
                    painter.drawPolygon(QPolygonF(points))
        except:
            pass

        label = f"{class_name} (SAM)"
        painter.setPen(QPen(border_color, 1))
        font = painter.font()
        font.setPixelSize(12)
        painter.setFont(font)

        ys, xs = np.where(mask_resized > 0)
        if len(xs) > 0:
            cx = int(np.mean(xs)) + video_rect.left()
            cy = int(np.mean(ys)) + video_rect.top()
            painter.drawText(cx - 30, cy, label)

    def get_frame_coordinates(self, pos):
        video_rect = self.get_video_rect()
        if not video_rect or video_rect.width() == 0 or video_rect.height() == 0:
            return None

        rel_pos = pos - video_rect.topLeft()
        original_width, original_height = self._get_original_dimensions()
        frame_x = int(rel_pos.x() * (original_width / video_rect.width()))
        frame_y = int(rel_pos.y() * (original_height / video_rect.height()))
        return frame_x, frame_y

    def remove_annotation_from_display(self, bbox_id: str):
        self.active_annotations = [
            a for a in self.active_annotations
            if a.get('bbox_id') != bbox_id
        ]
        self.update()

    def remove_annotation_by_id(self, bbox_id: str, remove_from_history=True):
        self.active_annotations = [
            a for a in self.active_annotations
            if a.get('bbox_id') != bbox_id
        ]

        if remove_from_history:
            frame = self.current_frame_num
            if frame in self.frame_annotations:
                self.frame_annotations[frame] = [
                    a for a in self.frame_annotations[frame]
                    if a.get('bbox_id') != bbox_id
                ]
                if not self.frame_annotations[frame]:
                    del self.frame_annotations[frame]

        # Também limpar confirmed_masks
        if hasattr(self, 'confirmed_masks'):
            frame = self.current_frame_num
            if frame in self.confirmed_masks:
                self.confirmed_masks[frame] = [
                    m for m in self.confirmed_masks[frame]
                    if m.get('bbox_id') != bbox_id
                ]

        self.update()