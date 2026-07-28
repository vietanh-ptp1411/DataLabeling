"""Interactive labeling canvas. Coordinates: the QGraphicsScene is in
image-pixel space, so all hit tests and stored shapes use image pixels."""
from enum import Enum, auto

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from app.models.bounding_box import BoundingBox
from app.models.polygon import PolygonAnnotation
from app.services import geometry as geo

MIN_BOX = 5.0
HANDLE_PX = 10.0          # hit threshold, screen pixels
ROT_OFFSET_PX = 25.0      # rotation handle distance above box, screen pixels
CLOSE_POLY_PX = 12.0
MIN_SCALE, MAX_SCALE = 0.1, 10.0


class DrawMode(Enum):
    BOX = auto()
    POLYGON = auto()


class Tool(Enum):
    POINTER = auto()
    PAN = auto()


class LabelCanvas(QGraphicsView):
    annotation_changed = Signal()
    selection_changed = Signal()
    status_message = Signal(str)
    mouse_moved = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._gscene = QGraphicsScene(self)
        self.setScene(self._gscene)
        self._pix_item = QGraphicsPixmapItem()
        self._gscene.addItem(self._pix_item)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QColor("#2b2b2b"))

        self.boxes = []
        self.polygons = []
        self.selected_box = None
        self.selected_polygon = None
        self.class_colors = {}
        self.current_class = ""
        self.labeling_enabled = True
        self.draw_mode = DrawMode.BOX
        self.tool = Tool.POINTER

        self._drag = None                 # dict describing active interaction
        self._pending_poly = []
        self._clipboard = None
        self._img_w = 0
        self._img_h = 0

    # ---------- image ----------

    def set_image(self, path):
        pix = QPixmap(path)
        if pix.isNull():
            return False
        self._pix_item.setPixmap(pix)
        self._img_w, self._img_h = pix.width(), pix.height()
        self._gscene.setSceneRect(0, 0, self._img_w, self._img_h)
        self.selected_box = None
        self.selected_polygon = None
        self._pending_poly = []
        self._drag = None
        self.fit_image()
        return True

    def clear_image(self):
        self._pix_item.setPixmap(QPixmap())
        self._img_w = self._img_h = 0
        self.boxes = []
        self.polygons = []
        self.viewport().update()

    def image_size(self):
        return (self._img_w, self._img_h)

    def fit_image(self):
        if self._img_w:
            self.fitInView(self._pix_item, Qt.KeepAspectRatio)
            self._clamp_scale()

    def _scale(self):
        return self.transform().m11()

    def _clamp_scale(self):
        s = self._scale()
        if s < MIN_SCALE:
            self.scale(MIN_SCALE / s, MIN_SCALE / s)
        elif s > MAX_SCALE:
            self.scale(MAX_SCALE / s, MAX_SCALE / s)

    # ---------- zoom / pan ----------

    def wheelEvent(self, event):
        if not self._img_w:
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        if not (MIN_SCALE <= self._scale() * factor <= MAX_SCALE):
            return
        pos = event.position().toPoint()
        before = self.mapToScene(pos)
        self.scale(factor, factor)
        delta = self.mapToScene(pos) - before
        self.translate(delta.x(), delta.y())

    def zoom_in(self):
        self._zoom_center(1.25)

    def zoom_out(self):
        self._zoom_center(0.8)

    def _zoom_center(self, factor):
        if self._img_w and MIN_SCALE <= self._scale() * factor <= MAX_SCALE:
            center = self.viewport().rect().center()
            before = self.mapToScene(center)
            self.scale(factor, factor)
            delta = self.mapToScene(center) - before
            self.translate(delta.x(), delta.y())

    def _start_pan(self, event):
        self._drag = {"kind": "pan", "last": event.position()}
        self.setCursor(Qt.ClosedHandCursor)

    def _do_pan(self, event):
        pos = event.position()
        last = self._drag["last"]
        delta = self.mapToScene(pos.toPoint()) - self.mapToScene(last.toPoint())
        self.translate(delta.x(), delta.y())
        self._drag["last"] = pos

    # ---------- events (extended in later tasks) ----------

    def mousePressEvent(self, event):
        if not self._img_w:
            return
        if event.button() == Qt.MiddleButton or (
                event.button() == Qt.LeftButton and self.tool == Tool.PAN):
            self._start_pan(event)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._img_w:
            sp = self.mapToScene(event.position().toPoint())
            self.mouse_moved.emit(sp.x(), sp.y())
        if self._drag and self._drag["kind"] == "pan":
            self._do_pan(event)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drag and self._drag["kind"] == "pan":
            self._drag = None
            self.setCursor(Qt.ArrowCursor)
            return
        super().mouseReleaseEvent(event)


if __name__ == "__main__":  # demo harness: python -m app.ui.label_canvas <image>
    import sys

    from PySide6.QtWidgets import QApplication

    qapp = QApplication(sys.argv)
    canvas = LabelCanvas()
    canvas.resize(1000, 700)
    canvas.set_image(sys.argv[1])
    canvas.show()
    sys.exit(qapp.exec())
