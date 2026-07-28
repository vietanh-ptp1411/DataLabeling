import datetime
import json
import os

from app.models.image_annotation import ImageAnnotation
from app.models.label_class import LabelClass

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def scan_images(folder):
    names = [n for n in os.listdir(folder)
             if os.path.splitext(n)[1].lower() in IMAGE_EXTS]
    return [os.path.join(folder, n) for n in sorted(names, key=str.lower)]


def labels_dir(folder):
    return os.path.join(folder, "labels")


def _label_path(folder, image_path):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(labels_dir(folder), stem + ".json")


def save_annotation(folder, ann):
    os.makedirs(labels_dir(folder), exist_ok=True)
    path = _label_path(folder, ann.image_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ann.to_dict(), f, indent=2)
    return path


def load_annotation(folder, image_path):
    path = _label_path(folder, image_path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        ann = ImageAnnotation.from_dict(data)
        ann.image_path = image_path  # trust current location, not stored path
        return ann
    except (json.JSONDecodeError, KeyError, TypeError, OSError):
        return None


def save_all_annotations(folder, anns):
    out_dir = os.path.join(folder, "annotations")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "annotations.json")
    data = {"ExportDate": datetime.datetime.now().isoformat(),
            "Images": [a.to_dict() for a in anns]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def _classes_path(folder):
    return os.path.join(folder, "classes.json")


def save_classes(folder, classes):
    with open(_classes_path(folder), "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in classes], f, indent=2)


def load_classes(folder):
    path = _classes_path(folder)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return [LabelClass.from_dict(d) for d in json.load(f)]
    except (json.JSONDecodeError, KeyError, TypeError, OSError):
        return []
