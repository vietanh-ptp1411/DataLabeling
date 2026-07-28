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

    # ---------- clipboard / edit API ----------

    def copy_selected(self):
        if self.selected_box:
            self._clipboard = self.selected_box.clone()
            self.status_message.emit("Đã copy ROI")

    def paste(self):
        if not self._clipboard or not self._img_w:
            return
        c = self._clipboard
        nx, ny = c.x + 20, c.y + 20
        if nx + c.width > self._img_w:
            nx = 0
        if ny + c.height > self._img_h:
            ny = 0
        nb = BoundingBox(c.class_name, nx, ny, c.width, c.height, c.angle)
        self.boxes.append(nb)
        self.selected_box = nb
        self.selected_polygon = None
        self._clipboard = nb.clone()   # repeated paste keeps cascading
        self.annotation_changed.emit()
        self.selection_changed.emit()
        self.status_message.emit("Đã paste ROI")
        self.viewport().update()

    def delete_selected(self):
        if self.selected_polygon is not None:
            self.polygons.remove(self.selected_polygon)
            self.selected_polygon = None
        elif self.selected_box is not None:
            self.boxes.remove(self.selected_box)
            self.selected_box = None
        else:
            return
        self.annotation_changed.emit()
        self.selection_changed.emit()
        self.viewport().update()

    # ---------- hit tests ----------

    def _tol(self):
        return HANDLE_PX / self._scale()

    def _resize_dir(self, box, sx, sy):
        """8-direction resize hit test in the box's local (unrotated) frame."""
        cx, cy = box.center
        lx, ly = geo.rotate_point(sx, sy, cx, cy, -box.angle)
        t = self._tol()
        if not (box.x - t <= lx <= box.x + box.width + t
                and box.y - t <= ly <= box.y + box.height + t):
            return None
        near_l = abs(lx - box.x) <= t
        near_r = abs(lx - (box.x + box.width)) <= t
        near_t = abs(ly - box.y) <= t
        near_b = abs(ly - (box.y + box.height)) <= t
        d = ("n" if near_t else "s" if near_b else "") + \
            ("w" if near_l else "e" if near_r else "")
        return d or None

    def _hit_rotation_handle(self, box, sx, sy):
        cx, cy = box.center
        off = ROT_OFFSET_PX / self._scale()
        hx, hy = geo.rotate_point(cx, box.y - off, cx, cy, box.angle)
        r = 10.0 / self._scale()
        return (sx - hx) ** 2 + (sy - hy) ** 2 <= r * r

    # ---------- polygons ----------

    def cancel_pending_polygon(self):
        self._pending_poly = []
        self.viewport().update()

    def _polygon_click(self, sx, sy):
        if not self.current_class:
            self.status_message.emit("Chọn class trước khi vẽ")
            return
        close_tol = CLOSE_POLY_PX / self._scale()
        if len(self._pending_poly) >= 3:
            fx, fy = self._pending_poly[0]
            if (sx - fx) ** 2 + (sy - fy) ** 2 <= close_tol ** 2:
                poly = PolygonAnnotation(self.current_class, list(self._pending_poly))
                self.polygons.append(poly)
                self.selected_polygon = poly
                self.selected_box = None
                self._pending_poly = []
                self.annotation_changed.emit()
                self.selection_changed.emit()
                self.viewport().update()
                return
        # vertex drag on an existing selected polygon takes priority over adding
        if not self._pending_poly and self.selected_polygon:
            idx = self._vertex_index_at(self.selected_polygon, sx, sy)
            if idx is not None:
                self._drag = {"kind": "drag_vertex",
                              "poly": self.selected_polygon, "index": idx}
                return
        if not self._pending_poly:
            hit = self._topmost_polygon_at(sx, sy)
            if hit and hit is not self.selected_polygon:
                self.selected_polygon = hit
                self.selected_box = None
                self.selection_changed.emit()
                self.viewport().update()
                return
        self._pending_poly.append((sx, sy))
        self.viewport().update()

    def _vertex_index_at(self, poly, sx, sy):
        t = self._tol()
        for i, (px, py) in enumerate(poly.points):
            if (sx - px) ** 2 + (sy - py) ** 2 <= t * t:
                return i
        return None

    def _draw_polygon(self, painter, poly, selected):
        color = self.class_colors.get(poly.class_name, "#FF0000")
        painter.setPen(self._pen(color, selected))
        painter.setBrush(Qt.NoBrush)
        qpoly = QPolygonF([QPointF(x, y) for x, y in poly.points])
        painter.drawPolygon(qpoly)
        font = QFont()
        font.setPointSizeF(max(6.0, 11.0 / self._scale()))
        painter.setFont(font)
        top = min(poly.points, key=lambda p: p[1])
        painter.drawText(QPointF(top[0], top[1] - 4 / self._scale()), poly.class_name)
        if selected:
            s = 6.0 / self._scale()
            painter.setBrush(QColor("#FFFF00"))
            for px, py in poly.points:
                painter.drawRect(QRectF(px - s / 2, py - s / 2, s, s))

    def _draw_pending_polygon(self, painter):
        pen = QPen(QColor("#00FF00"))
        pen.setWidthF(1.5 / self._scale())
        painter.setPen(pen)
        pts = [QPointF(x, y) for x, y in self._pending_poly]
        for a, b in zip(pts, pts[1:]):
            painter.drawLine(a, b)
        painter.setBrush(QColor("#00FF00"))
        r = 4.0 / self._scale()
        for p in pts:
            painter.drawEllipse(p, r, r)

    def _topmost_box_at(self, sx, sy):
        for box in reversed(self.boxes):
            if geo.point_in_box(box, sx, sy):
                return box
        return None

    def _topmost_polygon_at(self, sx, sy):
        for poly in reversed(self.polygons):
            if geo.point_in_polygon(poly.points, sx, sy):
                return poly
        return None

    # ---------- mouse events ----------

    def mousePressEvent(self, event):
        if not self._img_w:
            return
        if event.button() == Qt.MiddleButton or (
                event.button() == Qt.LeftButton and self.tool == Tool.PAN):
            self._start_pan(event)
            return
        if event.button() != Qt.LeftButton:
            return
        sp = self.mapToScene(event.position().toPoint())
        sx, sy = sp.x(), sp.y()

        if self.draw_mode == DrawMode.POLYGON and self.labeling_enabled:
            self._polygon_click(sx, sy)
            return

        # selected box first: rotation handle, then resize edges
        if self.selected_box:
            if self._hit_rotation_handle(self.selected_box, sx, sy):
                self._drag = {"kind": "rotate", "box": self.selected_box}
                return
            d = self._resize_dir(self.selected_box, sx, sy)
            if d:
                self._drag = {"kind": "resize", "box": self.selected_box, "dir": d}
                return
        hit = self._topmost_box_at(sx, sy)
        if hit:
            self.selected_box = hit
            self.selected_polygon = None
            self.selection_changed.emit()
            self._drag = {"kind": "move", "box": hit, "last": (sx, sy)}
            self.viewport().update()
            return
        poly = self._topmost_polygon_at(sx, sy)
        if poly:
            self.selected_polygon = poly
            self.selected_box = None
            self.selection_changed.emit()
            self._drag = {"kind": "move_poly", "poly": poly, "last": (sx, sy)}
            self.viewport().update()
            return
        # empty area: deselect + maybe start drawing
        self.selected_box = None
        self.selected_polygon = None
        self.selection_changed.emit()
        if self.labeling_enabled and self.current_class:
            self._drag = {"kind": "draw", "start": (sx, sy), "cur": (sx, sy)}
        self.viewport().update()

    def mouseMoveEvent(self, event):
        sp = self.mapToScene(event.position().toPoint())
        sx, sy = sp.x(), sp.y()
        if self._img_w:
            self.mouse_moved.emit(sx, sy)
        if not self._drag:
            self._update_hover_cursor(sx, sy)
            if self._pending_poly:
                self.viewport().update()
            return
        kind = self._drag["kind"]
        if kind == "pan":
            self._do_pan(event)
        elif kind == "draw":
            self._drag["cur"] = (sx, sy)
        elif kind == "move":
            box = self._drag["box"]
            lx, ly = self._drag["last"]
            box.x += sx - lx
            box.y += sy - ly
            self._drag["last"] = (sx, sy)
        elif kind == "move_poly":
            poly = self._drag["poly"]
            lx, ly = self._drag["last"]
            poly.points = [(px + sx - lx, py + sy - ly) for px, py in poly.points]
            self._drag["last"] = (sx, sy)
        elif kind == "resize":
            self._apply_resize(self._drag["box"], self._drag["dir"], sx, sy)
        elif kind == "rotate":
            box = self._drag["box"]
            cx, cy = box.center
            box.angle = geo.angle_from_center(cx, cy, sx, sy)
        elif kind == "drag_vertex":
            self._drag["poly"].points[self._drag["index"]] = (sx, sy)
        self.viewport().update()

    def mouseReleaseEvent(self, event):
        if not self._drag:
            return
        kind = self._drag["kind"]
        if kind == "pan":
            self.setCursor(Qt.ArrowCursor)
        elif kind == "draw":
            x0, y0 = self._drag["start"]
            x1, y1 = self._drag["cur"]
            x, y = min(x0, x1), min(y0, y1)
            w, h = abs(x1 - x0), abs(y1 - y0)
            if w >= MIN_BOX and h >= MIN_BOX:
                nb = BoundingBox(self.current_class, x, y, w, h)
                self.boxes.append(nb)
                self.selected_box = nb
                self.annotation_changed.emit()
                self.selection_changed.emit()
        elif kind in ("move", "move_poly", "resize", "rotate", "drag_vertex"):
            self.annotation_changed.emit()
        self._drag = None
        self.viewport().update()

    def _apply_resize(self, box, d, sx, sy):
        cx, cy = box.center
        lx, ly = geo.rotate_point(sx, sy, cx, cy, -box.angle)
        x2, y2 = box.x + box.width, box.y + box.height
        if "w" in d:
            box.x = min(lx, x2 - MIN_BOX)
            box.width = x2 - box.x
        if "e" in d:
            box.width = max(MIN_BOX, lx - box.x)
        if "n" in d:
            box.y = min(ly, y2 - MIN_BOX)
            box.height = y2 - box.y
        if "s" in d:
            box.height = max(MIN_BOX, ly - box.y)

    _CURSORS = {"n": Qt.SizeVerCursor, "s": Qt.SizeVerCursor,
                "e": Qt.SizeHorCursor, "w": Qt.SizeHorCursor,
                "ne": Qt.SizeBDiagCursor, "sw": Qt.SizeBDiagCursor,
                "nw": Qt.SizeFDiagCursor, "se": Qt.SizeFDiagCursor}

    def _update_hover_cursor(self, sx, sy):
        if self.selected_box:
            if self._hit_rotation_handle(self.selected_box, sx, sy):
                self.setCursor(Qt.PointingHandCursor)
                return
            d = self._resize_dir(self.selected_box, sx, sy)
            if d:
                self.setCursor(self._CURSORS[d])
                return
        if self._topmost_box_at(sx, sy) or self._topmost_polygon_at(sx, sy):
            self.setCursor(Qt.SizeAllCursor)
        else:
            self.setCursor(Qt.CrossCursor if self.labeling_enabled else Qt.ArrowCursor)

    # ---------- rendering ----------

    def _pen(self, color, selected, width=2.0):
        pen = QPen(QColor("#FFFF00") if selected else QColor(color))
        pen.setWidthF((3.0 if selected else width) / self._scale())
        pen.setCosmetic(False)
        return pen

    def drawForeground(self, painter, rect):
        if not self._img_w:
            return
        for box in self.boxes:
            self._draw_box(painter, box, box is self.selected_box)
        for poly in self.polygons:
            self._draw_polygon(painter, poly, poly is self.selected_polygon)
        if self._drag and self._drag["kind"] == "draw":
            x0, y0 = self._drag["start"]
            x1, y1 = self._drag["cur"]
            pen = QPen(QColor("#00FF00"))
            pen.setWidthF(1.5 / self._scale())
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(QRectF(min(x0, x1), min(y0, y1),
                                    abs(x1 - x0), abs(y1 - y0)))
        if self._pending_poly:
            self._draw_pending_polygon(painter)

    def _draw_box(self, painter, box, selected):
        color = self.class_colors.get(box.class_name, "#FF0000")
        painter.save()
        cx, cy = box.center
        painter.translate(cx, cy)
        painter.rotate(box.angle)
        w, h = box.width, box.height
        painter.setPen(self._pen(color, selected))
        painter.drawRect(QRectF(-w / 2, -h / 2, w, h))
        # class badge
        font = QFont()
        font.setPointSizeF(max(6.0, 11.0 / self._scale()))
        painter.setFont(font)
        painter.drawText(QPointF(-w / 2, -h / 2 - 4 / self._scale()), box.class_name)
        if selected:
            self._draw_handles(painter, w, h)
        painter.restore()

    def _draw_handles(self, painter, w, h):
        s = 6.0 / self._scale()
        painter.setPen(QPen(QColor("#FFFF00"), 1.0 / self._scale()))
        painter.setBrush(QColor("#FFFF00"))
        for hx in (-w / 2, 0, w / 2):
            for hy in (-h / 2, 0, h / 2):
                if hx == 0 and hy == 0:
                    continue
                painter.drawRect(QRectF(hx - s / 2, hy - s / 2, s, s))
        # rotation handle: circle above top edge, connected by a line
        off = ROT_OFFSET_PX / self._scale()
        painter.drawLine(QPointF(0, -h / 2), QPointF(0, -h / 2 - off))
        painter.setBrush(QColor("#00BFFF"))
        r = 7.0 / self._scale()
        painter.drawEllipse(QPointF(0, -h / 2 - off), r, r)


if __name__ == "__main__":  # demo harness: python -m app.ui.label_canvas <image>
    import sys

    from PySide6.QtWidgets import QApplication

    qapp = QApplication(sys.argv)
    canvas = LabelCanvas()
    canvas.resize(1000, 700)
    canvas.set_image(sys.argv[1])
    canvas.show()
    sys.exit(qapp.exec())
