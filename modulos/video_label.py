import cv2
from PyQt6.QtCore import Qt, QPointF, QRect
from PyQt6.QtGui import QPainter, QPen, QColor
from PyQt6.QtWidgets import QLabel, QSizePolicy

class VideoLabel(QLabel):
    """Class for manual annotations"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.drawing_enabled = False
        self.current_class = None
        self.drawing_color = QColor(Qt.GlobalColor.green)
        self.active_annotations = []  
        self.drawing = False 
        self.delete_box_size = 16          
        self.delete_box_offset = 4   
        self.original_width  = None
        self.original_height = None      
        
        #system of annotations per frame
        self.frame_annotations = {}  
        self.current_frame_num = 0
        
        # mouse state
        self.mouse_state = "normal"  # normal, drawing, dragging, resizing
        self.start_point = QPointF()
        self.end_point = QPointF()
        
        # visual configurations 
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

    def setPixmap(self, pixmap):
        """Stores the original pixmap and updates the display"""
        self._pixmap = pixmap
        self.update_display()

    def update_display(self):
        """Updates the display of the resized video"""
        if self._pixmap is None:
            return
            
        # Resize while maintaining aspect ratio
        scaled_pixmap = self._pixmap.scaled(
            self.size(), 
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Calculates the position to center
        self.video_x = (self.width() - scaled_pixmap.width()) // 2
        self.video_y = (self.height() - scaled_pixmap.height()) // 2
        self.video_rect = QRect(self.video_x, self.video_y, 
                              scaled_pixmap.width(), scaled_pixmap.height())
        
        super().setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        """Resizes the video when the widget changes size"""
        self.update_display()
        super().resizeEvent(event)

    def set_video_rect(self, rect):
        """Defines the area where the video is being displayed"""
        self.video_rect = rect

    def get_video_rect(self):
        """Returns the video area or the entire rectangle if not set"""
        return self.video_rect if self.video_rect else self.rect()

    def reset_annotations(self):
        """Clear only visible annotations"""
        self.active_annotations = []
        self.update()

    def mousePressEvent(self, event):
        if not self.drawing_enabled:
            return

        video_rect = self.get_video_rect()
        pos = event.position().toPoint()
        if not video_rect.contains(pos):
            return

        if event.button() == Qt.MouseButton.LeftButton:
            main_win = self.window()
            
            # SAM 2 mode active
            if hasattr(main_win, 'sam2_refinement_mode') and main_win.sam2_refinement_mode:
                frame_coords = self.get_frame_coordinates(pos)
                if frame_coords:
                    x, y = frame_coords
                    
                    # Check if click is inside the SAM refinement bbox
                    if hasattr(main_win, 'current_bb_for_refinement') and main_win.current_bb_for_refinement:
                        bb = main_win.current_bb_for_refinement
                        if bb["x1"] <= x <= bb["x2"] and bb["y1"] <= y <= bb["y2"]:
                            # Click INSIDE → process SAM 2
                            if hasattr(main_win, 'sam2_thread'):
                                prompts = [(x, y, 1)]
                                frame = self._get_frame_for_sam(main_win)
                                if frame is not None:
                                    main_win.sam2_thread.set_frame_and_prompts(
                                        frame, main_win.current_frame_num, prompts
                                    )
                                    return  # Block drawing new bbox
                        else:
                            # Click OUTSIDE → auto-disable SAM 2
                            main_win.toggle_sam2_refinement()
                            # Continue to draw new bbox below
                    else:
                        # No bbox stored → auto-disable SAM 2
                        main_win.toggle_sam2_refinement()
                        # Continue to draw new bbox
            
            if event.button() == Qt.MouseButton.LeftButton:
                for ann in self.active_annotations:
                    if ann.get("delete_rect") and ann["delete_rect"].contains(pos):
                        self.remove_annotation(ann)
                        return       

            if event.button() == Qt.MouseButton.LeftButton:
                adjusted_pos = pos - video_rect.topLeft()
                video_size   = video_rect.size()
                self.start_point = QPointF(adjusted_pos.x() / video_size.width(),
                                        adjusted_pos.y() / video_size.height())
                self.end_point = self.start_point
                self.drawing   = True
                self.update()
                return
            
            if event.button() == Qt.MouseButton.RightButton:
                self.delete_annotation_at(pos, video_rect) 

    def _get_frame_for_sam(self, main_win):
        """Helper to get current frame for SAM"""
        if hasattr(main_win, 'current_frame') and main_win.current_frame is not None:
            return main_win.current_frame
        elif hasattr(main_win, 'cap') and main_win.cap is not None:
            current_pos = main_win.cap.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = main_win.cap.read()
            main_win.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
            if ret:
                return frame
        return None   

    def mouseMoveEvent(self, event):
        if self.drawing and self.drawing_enabled:
            video_rect = self.get_video_rect()
            if not video_rect:
                return
                
            current_pos = event.position().toPoint() - video_rect.topLeft()
            video_size = video_rect.size()
            
            # Converts to normalised coordinates (only for temporary visualization)
            x = max(0, min(current_pos.x() / video_size.width(), 1.0))
            y = max(0, min(current_pos.y() / video_size.height(), 1.0))
            self.end_point = QPointF(x, y)
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drawing and self.drawing_enabled and self.current_class:
            video_rect = self.get_video_rect()
            if not video_rect:
                return
                
            current_pos = event.position().toPoint() - video_rect.topLeft()
            video_size = video_rect.size()
            
            # Normalize coordinates
            x = max(0, min(current_pos.x() / video_size.width(), 1.0))
            y = max(0, min(current_pos.y() / video_size.height(), 1.0))
            self.end_point = QPointF(x, y)
            self.drawing = False
            
            # Calculate normalized coordinates
            x1_norm = min(self.start_point.x(), self.end_point.x())
            y1_norm = min(self.start_point.y(), self.end_point.y())
            x2_norm = max(self.start_point.x(), self.end_point.x())
            y2_norm = max(self.start_point.y(), self.end_point.y())
            
            # Get original dimensions
            main_window = self.window()
            original_width, original_height = 1920, 1080  # fallback

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

            self.original_width = original_width
            self.original_height = original_height

            # Only create annotation if box has significant size
            min_size = 0.01  # 1% of frame size
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
                    "frame_source": (main_window.video_path or "Live", main_window.current_frame_num)
                }
                
                if main_window.training_wizard is not None:
                    annotation["type"] = "training"
                else:
                    annotation["type"] = "manual"
                
                # Store in current frame's annotations
                if main_window.current_frame_num not in self.frame_annotations:
                    self.frame_annotations[main_window.current_frame_num] = []
                self.frame_annotations[main_window.current_frame_num].append(annotation)
                
                # Add to active annotations for display
                self.active_annotations.append(annotation)
                
                # Show SAM button after first bbox is drawn
                if hasattr(main_window, 'sam2_refinement_btn') and main_window.sam2_refinement_btn:
                    main_window.sam2_refinement_btn.setVisible(True)
                
                # Add to history
                if hasattr(main_window, 'add_manual_annotation_to_history'):
                    main_window.add_manual_annotation_to_history(annotation)
            
            self.update()

    def update_active_annotations(self):
        self.active_annotations = self.frame_annotations.get(self.current_frame_num, [])
        self.update()
        
    def remove_annotation(self, ann):
        frame = ann["frame_number"]

        if ann in self.active_annotations:
            self.active_annotations.remove(ann)

        if frame in self.frame_annotations:
            try:
                self.frame_annotations[frame].remove(ann)
            except ValueError:
                pass
            if not self.frame_annotations[frame]:   
                del self.frame_annotations[frame]

        main_window = self.window()
        if hasattr(main_window, 'remove_annotation_from_history'):
            main_window.remove_annotation_from_history(ann)
        elif hasattr(main_window, 'detections_dock') and \
             hasattr(main_window.detections_dock, 'remove_detection'):
            main_window.detections_dock.remove_detection(ann)

        self.update()

    def delete_annotation_at(self, pos, video_rect):
        """Remove the annotation under the cursor (right click)."""
        x = (pos.x() - video_rect.left()) / video_rect.width()
        y = (pos.y() - video_rect.top())  / video_rect.height()

        # search from newest to oldest 
        for ann in reversed(self.active_annotations):
            if ann["x1"] <= x <= ann["x2"] and ann["y1"] <= y <= ann["y2"]:
                self.remove_annotation(ann)
                break

    def paintEvent(self, event):
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        video_rect = self.get_video_rect()
        if not video_rect:
            return
            
        painter.setClipRect(video_rect)

        # Draws active annotations 
        for ann in self.active_annotations:
            color = QColor(ann.get("color", self.drawing_color.name()))
            painter.setPen(QPen(color, 4, Qt.PenStyle.SolidLine))
            

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

            # Calculate scale factors 
            scale_x = video_rect.width() / original_width
            scale_y = video_rect.height() / original_height
                
            # Converts absolute pixels to widget coordinates
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

            # translucent red background
            painter.fillRect(delete_rect, QColor(255, 80, 80, 200))

            # white delete 'x'
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            x1_del, y1_del = delete_rect.left() + 4, delete_rect.top() + 4
            x2_del, y2_del = delete_rect.right() - 4, delete_rect.bottom() - 4
            painter.drawLine(x1_del, y1_del, x2_del, y2_del)
            painter.drawLine(x1_del, y2_del, x2_del, y1_del)

            ann["delete_rect"] = delete_rect

        # Draws temporary box (normalised coordinates)
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

    def get_frame_coordinates(self, pos):
        """Convert widget coordinates to original frame coordinates"""
        video_rect = self.get_video_rect()
        if not video_rect or video_rect.width() == 0 or video_rect.height() == 0:
            return None
            
        rel_pos = pos - video_rect.topLeft()
        frame_x = int(rel_pos.x() * (self.original_width / video_rect.width()))
        frame_y = int(rel_pos.y() * (self.original_height / video_rect.height()))
        return frame_x, frame_y