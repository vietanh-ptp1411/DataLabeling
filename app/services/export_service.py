"""Export in-memory annotations to an ultralytics-ready YOLO dataset."""
import os
import random
import shutil

import yaml
from PIL import Image

from app.services.geometry import aabb_of, box_corners, clamp01

TASKS = ("detect", "obb", "segment")


def detect_line(box, cls_id, img_w, img_h):
    x, y, w, h = (box.x, box.y, box.width, box.height)
    if box.angle:
        x, y, w, h = aabb_of(box_corners(box))
    cx = clamp01((x + w / 2) / img_w)
    cy = clamp01((y + h / 2) / img_h)
    return f"{cls_id} {cx:.6f} {cy:.6f} {clamp01(w / img_w):.6f} {clamp01(h / img_h):.6f}"


def obb_line(box, cls_id, img_w, img_h):
    coords = []
    for px, py in box_corners(box):
        coords += [clamp01(px / img_w), clamp01(py / img_h)]
    return f"{cls_id} " + " ".join(f"{v:.6f}" for v in coords)


def seg_line(poly, cls_id, img_w, img_h):
    coords = []
    for px, py in poly.points:
        coords += [clamp01(px / img_w), clamp01(py / img_h)]
    return f"{cls_id} " + " ".join(f"{v:.6f}" for v in coords)


def split_names(names, train_ratio, seed=42):
    shuffled = list(names)
    random.Random(seed).shuffle(shuffled)
    n_train = int(len(shuffled) * train_ratio)
    return shuffled[:n_train], shuffled[n_train:]


def _relevant(ann, task):
    return ann.polygons if task == "segment" else ann.boxes


def _lines_for(ann, task, cls_ids, img_w, img_h):
    lines = []
    if task == "segment":
        for p in ann.polygons:
            if p.class_name in cls_ids and len(p.points) >= 3:
                lines.append(seg_line(p, cls_ids[p.class_name], img_w, img_h))
    else:
        fn = obb_line if task == "obb" else detect_line
        for b in ann.boxes:
            if b.class_name in cls_ids:
                lines.append(fn(b, cls_ids[b.class_name], img_w, img_h))
    return lines


def export_dataset(annotations, class_names, out_dir, task, train_ratio=0.8,
                   copy_images=True, write_yaml=True, seed=42, progress_cb=None):
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    cls_ids = {name: i for i, name in enumerate(class_names)}
    usable = [a for a in annotations if _relevant(a, task)]
    train_anns, val_anns = split_names(usable, train_ratio, seed)

    for split in ("train", "val"):
        os.makedirs(os.path.join(out_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, split, "labels"), exist_ok=True)

    total = len(usable) or 1
    done = 0
    for split, anns in (("train", train_anns), ("val", val_anns)):
        for ann in anns:
            with Image.open(ann.image_path) as im:
                img_w, img_h = im.size
            stem = os.path.splitext(os.path.basename(ann.image_path))[0]
            label_path = os.path.join(out_dir, split, "labels", stem + ".txt")
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(_lines_for(ann, task, cls_ids, img_w, img_h)))
            if copy_images:
                shutil.copy2(ann.image_path,
                             os.path.join(out_dir, split, "images",
                                          os.path.basename(ann.image_path)))
            done += 1
            if progress_cb:
                progress_cb(done, total)

    yaml_path = None
    if write_yaml:
        yaml_path = os.path.join(out_dir, "data.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump({"path": os.path.abspath(out_dir),
                            "train": "train/images", "val": "val/images",
                            "nc": len(class_names),
                            "names": list(class_names)}, f, sort_keys=False)
    return {"train": len(train_anns), "val": len(val_anns), "yaml": yaml_path}
