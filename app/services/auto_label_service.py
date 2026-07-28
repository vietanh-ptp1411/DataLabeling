"""Ultralytics-based auto labeling. `ultralytics` is imported lazily so the
module (and its pure parser) stays importable in test environments."""
import math

from app.models.bounding_box import BoundingBox
from app.services.export_service import detect_line


def boxes_from_result(result, class_names):
    """Convert one ultralytics Results object to BoundingBox list (pixels)."""
    out = []

    def name_of(cls_idx):
        i = int(cls_idx)
        return class_names[i] if 0 <= i < len(class_names) else str(i)

    obb = getattr(result, "obb", None)
    if obb is not None:
        for row, cls in zip(obb.xywhr.tolist(), obb.cls.tolist()):
            cx, cy, w, h, r = row
            out.append(BoundingBox(name_of(cls), cx - w / 2, cy - h / 2,
                                   w, h, math.degrees(r) % 360))
        return out
    boxes = getattr(result, "boxes", None)
    if boxes is not None:
        for row, cls in zip(boxes.xyxy.tolist(), boxes.cls.tolist()):
            x1, y1, x2, y2 = row
            out.append(BoundingBox(name_of(cls), x1, y1, x2 - x1, y2 - y1))
    return out


def save_yolo_txt(boxes, txt_path, img_w, img_h, class_names):
    ids = {n: i for i, n in enumerate(class_names)}
    lines = [detect_line(b, ids[b.class_name], img_w, img_h)
             for b in boxes if b.class_name in ids]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


class AutoLabelService:
    def __init__(self):
        self.model = None
        self.class_names = []

    def load_model(self, path):
        from ultralytics import YOLO
        self.model = YOLO(path)
        names = self.model.names  # dict {id: name}
        self.class_names = [names[i] for i in sorted(names)] if names else []
        return self.class_names

    def predict(self, image_path, conf=0.5, iou=0.45):
        result = self.model.predict(image_path, conf=conf, iou=iou,
                                    verbose=False)[0]
        return boxes_from_result(result, self.class_names)
