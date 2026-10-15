import cv2
import numpy as np
from PyQt5.QtCore import QTimer, QRectF, Qt, pyqtSignal, QLineF
from PyQt5.QtGui import (
    QColor,
    QFont,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PyQt5.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QMessageBox,
)
from network.image_geometry import crop_effective_roi


class ImageCanvas(QGraphicsView):

    lymphnode_selected = pyqtSignal(int)
    image_point_selected = pyqtSignal(float, float)
    calibration_line_finished = pyqtSignal(float, float, float, float)

    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(520, 420)
        self.pixmap_item = None
        self.ai_items = []
        self.lymphnode_items = []
        self.lymphnode_instances = []
        self.selected_lymphnode_index = -1
        # True = 多实例淋巴结显示模式，此时禁止旧 bbox 逻辑覆盖
        self._lymphnode_mode = False
        self._video_timer = QTimer(self)
        self._video_timer.timeout.connect(self._read_video_frame)
        self._video_capture = None
        self._video_interval = 33
        self._is_playing_video = False
        self._roi_info = None
        self._display_mode = "overlay"
        self._calibration_active = False
        self._last_mask = None
        self._calibration_start = None
        self._calibration_line = None
        self._show_empty_state()

    def _show_empty_state(self):
        self.scene.clear()
        self.scene.setSceneRect(0, 0, 520, 420)
        label = self.scene.addText("请上传超声图像")
        label.setDefaultTextColor(QColor("#64748b"))
        label.setFont(QFont("Microsoft YaHei", 16))
        label.setPos(180, 190)

    def load_image(self, image_path):
        self.stop_video()
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            QMessageBox.warning(self, "图像错误", "无法读取所选图像。")
            self._show_empty_state()
            return False

        try:
            _, roi_info = crop_effective_roi(image_bgr)
        except ValueError as exc:
            QMessageBox.warning(self, "图像错误", str(exc))
            self._show_empty_state()
            return False

        return self.load_image_array(image_bgr, roi_info=roi_info)

    def load_image_array(self, image_bgr, roi_info=None):
        self.stop_video()
        self.scene.clear()
        self.ai_items = []
        self.lymphnode_items = []
        self.lymphnode_instances = []
        self.selected_lymphnode_index = -1
        self._lymphnode_mode = False
        self._last_mask = None
        self._roi_info = roi_info
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        image = QImage(
            image_rgb.data,
            width,
            height,
            width * 3,
            QImage.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(image)
        if pixmap.isNull():
            QMessageBox.warning(self, "图像错误", "无法读取所选图像。")
            self._show_empty_state()
            return False

        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
        # Leave a small scene margin so thick selected contours at the image
        # edge are not clipped by the viewport boundary.
        self.scene.setSceneRect(QRectF(pixmap.rect()).adjusted(-8, -8, 8, 8))
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        return True

    def _mask_for_display(self, mask, width, height):
        mask = np.asarray(mask, dtype=np.uint8)
        if mask.ndim == 3:
            mask = mask[0]

        roi = self._roi_info
        if roi and mask.shape == (roi["height"], roi["width"]):
            full_mask = np.zeros((height, width), dtype=np.uint8)
            full_mask[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]] = mask
            return full_mask

        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return mask

    def play_video(self, video_path):
        self.stop_video()
        self.scene.clear()
        self.ai_items = []
        self.lymphnode_items = []
        self.lymphnode_instances = []
        self.selected_lymphnode_index = -1
        self._lymphnode_mode = False
        self.pixmap_item = None

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            QMessageBox.warning(self, "视频错误", "无法打开视频文件。")
            return False

        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 30.0
        self._video_interval = int(1000.0 / fps)
        self._video_capture = capture
        self._is_playing_video = True
        self._video_timer.start(self._video_interval)
        return True

    def stop_video(self):
        if self._video_timer.isActive():
            self._video_timer.stop()
        if self._video_capture is not None:
            self._video_capture.release()
            self._video_capture = None
        self._is_playing_video = False

    def draw_ai_result(self, mask=None):
        # 多实例淋巴结模式下，旧 bbox / 旧 overlay 一律不允许覆盖
        if self._lymphnode_mode:
            return

        if not self.pixmap_item:
            return

        for item in self.ai_items:
            self.scene.removeItem(item)
        self.ai_items = []

        if mask is None:
            return
        self._last_mask = mask

        rect = self.pixmap_item.boundingRect()
        width = int(rect.width())
        height = int(rect.height())
        if mask.ndim == 3:
            mask = mask[0]

        mask = self._mask_for_display(mask, width, height)

        overlay_image = QImage(width, height, QImage.Format_ARGB32)
        overlay_image.fill(Qt.transparent)
        for y, x in zip(*np.nonzero(mask > 0)):
            overlay_image.setPixelColor(x, y, QColor(59, 130, 246, 90))

        overlay_pixmap = QGraphicsPixmapItem(QPixmap.fromImage(overlay_image))
        overlay_pixmap.setZValue(1)
        overlay_pixmap.setToolTip("AI分割结果")
        self.scene.addItem(overlay_pixmap)
        self.ai_items.append(overlay_pixmap)

        coords = np.column_stack(np.nonzero(mask > 0))
        if coords.size > 0:
            top, left = coords.min(axis=0)
            bottom, right = coords.max(axis=0)
            box = QGraphicsRectItem(left, top, right - left + 1, bottom - top + 1)
            box.setPen(QPen(QColor("#ef4444"), 4))
            box.setToolTip("AI检测框：疑似淋巴结病灶")
            box.setZValue(2)
            self.scene.addItem(box)
            self.ai_items.append(box)

    def draw_lymphnode_instances(self, instances):
        # Clear stale ai_items overlays to avoid mixing with multi-instance view
        for item in self.ai_items:
            self.scene.removeItem(item)
        self.ai_items = []

        # Clear previous lymphnode overlay items
        for item in self.lymphnode_items:
            self.scene.removeItem(item)
        self.lymphnode_items = []

        self.lymphnode_instances = instances or []
        self.selected_lymphnode_index = 0 if self.lymphnode_instances else -1

        if not self.lymphnode_instances:
            # 无实例：退回旧逻辑，允许 draw_ai_result 正常绘制
            self._lymphnode_mode = False
            return

        # 进入多实例模式：只显示 per-instance mask / 红蓝轮廓 / 测量线
        self._lymphnode_mode = True
        self._redraw_lymphnode_overlays()

    def _redraw_lymphnode_overlays(self):
        # Clear existing lymphnode overlays (used on selection change too)
        for item in self.lymphnode_items:
            self.scene.removeItem(item)
        self.lymphnode_items = []

        # 防御性清除：任何残留的旧 bbox / overlay 都不允许出现在多实例视图中
        for item in self.ai_items:
            self.scene.removeItem(item)
        self.ai_items = []

        if not self.pixmap_item or not self.lymphnode_instances:
            return

        rect = self.pixmap_item.boundingRect()
        pix_w = int(rect.width())
        pix_h = int(rect.height())

        for i, inst in enumerate(self.lymphnode_instances):
            mask = inst.get("segmentation", {}).get("mask")
            if mask is None:
                continue

            mask_disp = self._mask_for_display(mask, pix_w, pix_h)

            is_selected = (i == self.selected_lymphnode_index)

            # --- Semi-transparent blue fill (per instance) ---
            fill_image = QImage(pix_w, pix_h, QImage.Format_ARGB32)
            fill_image.fill(Qt.transparent)
            ys, xs = np.nonzero(mask_disp > 0)
            fill_color = QColor(59, 130, 246, 90)
            for yy, xx in zip(ys, xs):
                fill_image.setPixelColor(int(xx), int(yy), fill_color)

            fill_item = QGraphicsPixmapItem(QPixmap.fromImage(fill_image))
            fill_item.setZValue(1)
            fill_item.setToolTip("分割区域")
            self.scene.addItem(fill_item)
            self.lymphnode_items.append(fill_item)

            # --- Contour (red + thick if selected) ---
            contours, _ = cv2.findContours(
                mask_disp,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            pts = contour.reshape(-1, 2).astype(float)

            path = QPainterPath()
            for j, (px, py) in enumerate(pts):
                if j == 0:
                    path.moveTo(px, py)
                else:
                    path.lineTo(px, py)
            path.closeSubpath()

            contour_item = QGraphicsPathItem(path)
            if is_selected:
                contour_item.setPen(QPen(QColor("#ef4444"), 5))
                contour_item.setZValue(3)
            else:
                contour_item.setPen(QPen(QColor("#3b82f6"), 2))
                contour_item.setZValue(2)
            self.scene.addItem(contour_item)
            self.lymphnode_items.append(contour_item)

    def mousePressEvent(self, event):
        if self._calibration_active and self.pixmap_item is not None:
            point = self._image_point_from_event(event)
            if point is not None:
                self._calibration_start = point
                self._draw_calibration_line(point, point)
                event.accept()
                return
        if self.lymphnode_instances and self.pixmap_item is not None:
            scene_pos = self.mapToScene(event.pos())
            rect = self.pixmap_item.boundingRect()

            if rect.contains(scene_pos):
                px = scene_pos.x() - rect.left()
                py = scene_pos.y() - rect.top()

                pix_w = rect.width()
                pix_h = rect.height()

                for i, inst in enumerate(self.lymphnode_instances):
                    mask = inst.get("segmentation", {}).get("mask")

                    if mask is None:
                        continue

                    mask = self._mask_for_display(
                        mask,
                        int(rect.width()),
                        int(rect.height()),
                    )
                    mask_h, mask_w = mask.shape[:2]

                    x = int(px * mask_w / pix_w)
                    y = int(py * mask_h / pix_h)

                    x = max(0, min(x, mask_w - 1))
                    y = max(0, min(y, mask_h - 1))

                    if mask[y, x] > 0:
                        self.selected_lymphnode_index = i
                        self._redraw_lymphnode_overlays()
                        self.lymphnode_selected.emit(i)
                        event.accept()
                        return

        # 点击空白区域：保持当前选择，不取消高亮
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._calibration_active and self._calibration_start is not None:
            point = self._image_point_from_event(event)
            if point is not None:
                self._draw_calibration_line(self._calibration_start, point)
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._calibration_active and self._calibration_start is not None:
            point = self._image_point_from_event(event)
            start = self._calibration_start
            self._calibration_start = None
            self._remove_calibration_line()
            if point is not None:
                self.calibration_line_finished.emit(start[0], start[1], point[0], point[1])
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self._calibration_active:
            self.set_calibration_active(False)
            event.accept()
            return
        super().keyPressEvent(event)

    def _image_point_from_event(self, event):
        if self.pixmap_item is None:
            return None
        scene_pos = self.mapToScene(event.pos())
        rect = self.pixmap_item.boundingRect()
        if not rect.contains(scene_pos):
            return None
        width = self.pixmap_item.pixmap().width()
        height = self.pixmap_item.pixmap().height()
        return ((scene_pos.x() - rect.left()) * width / rect.width(), (scene_pos.y() - rect.top()) * height / rect.height())

    def _draw_calibration_line(self, start, end):
        self._remove_calibration_line()
        from PyQt5.QtWidgets import QGraphicsLineItem
        self._calibration_line = QGraphicsLineItem(QLineF(start[0], start[1], end[0], end[1]))
        self._calibration_line.setPen(QPen(QColor("#f97316"), 2))
        self._calibration_line.setZValue(10)
        self.scene.addItem(self._calibration_line)

    def _remove_calibration_line(self):
        if self._calibration_line is not None:
            self.scene.removeItem(self._calibration_line)
            self._calibration_line = None

    def set_calibration_active(self, active):
        self._calibration_active = bool(active)
        if not active:
            self._calibration_start = None
            self._remove_calibration_line()
        self.setCursor(Qt.CrossCursor if active else Qt.ArrowCursor)

    def set_display_mode(self, mode):
        self._display_mode = mode
        if mode == "original":
            for item in self.ai_items + self.lymphnode_items:
                self.scene.removeItem(item)
            self.ai_items = []
            self.lymphnode_items = []
            self._lymphnode_mode = False
        elif mode == "mask":
            self._lymphnode_mode = False
            for item in self.lymphnode_items:
                self.scene.removeItem(item)
            self.lymphnode_items = []
            self.draw_ai_result(self._last_mask)
        else:
            if self.lymphnode_instances:
                self.draw_lymphnode_instances(self.lymphnode_instances)
            elif self._last_mask is not None:
                self.draw_ai_result(self._last_mask)

    def _read_video_frame(self):
        if self._video_capture is None:
            return

        ret, frame = self._video_capture.read()
        if not ret:
            self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._video_capture.read()
            if not ret:
                self.stop_video()
                return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytes_per_line = channel * width
        image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(image)
        if self.pixmap_item is None:
            self.pixmap_item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(self.pixmap_item)
        else:
            self.pixmap_item.setPixmap(pixmap)

        self.scene.setSceneRect(QRectF(pixmap.rect()).adjusted(-8, -8, 8, 8))
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def export_snapshot(self, output_path, width=760):
        if not self.pixmap_item:
            return ""

        rect = self.scene.sceneRect()
        scale = width / rect.width() if rect.width() else 1
        height = max(1, int(rect.height() * scale))
        image = QImage(width, height, QImage.Format_ARGB32)
        image.fill(QColor("#ffffff"))

        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        self.scene.render(painter, QRectF(0, 0, width, height), rect)
        painter.end()

        image.save(str(output_path))
        return str(output_path)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap_item:
            self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
