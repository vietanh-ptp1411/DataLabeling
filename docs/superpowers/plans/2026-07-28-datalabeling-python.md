# DataLabeling_Python Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the C# WPF image-labeling app to Python (PySide6) with integrated YOLO training (ultralytics), at `E:\DataLabeling_Python`.

**Architecture:** A PySide6 desktop app. All annotation state lives in plain dataclasses (`app/models`); pure-Python services (`app/services`) handle JSON persistence, geometry math, YOLO export, auto-label inference, and video frame extraction — all unit-testable without Qt. The UI layer (`app/ui`) has one custom `QGraphicsView` canvas (`LabelCanvas`) that draws shapes in `drawForeground` and hit-tests in image-pixel coordinates via the geometry service; windows/dialogs wire services to widgets. Long work (inference, video, training) runs on `QThread` workers with signals.

**Tech Stack:** Python 3.10+, PySide6, ultralytics (YOLO), opencv-python, PyYAML, Pillow, pytest.

## Global Constraints

- Target dir: `E:\DataLabeling_Python` (git repo already initialized; spec committed).
- Annotation JSON must stay byte-compatible in structure with the C# app: per-image `labels/<imageName>.json` with keys `ImagePath, ImageFileName, BoundingBoxes[{Id, ClassName, X, Y, Width, Height, Angle}], Polygons[{Id, ClassName, Points[{X, Y}]}]` — absolute pixel coords, angle in **degrees**. Extra keys in old files (e.g. `Tag`) must be ignored on load, not crash.
- Class colors: `md5(class_name)` → 16-color palette (NOT Python `hash()`, which is per-process randomized).
- Export split: `random.Random(seed)` shuffle, seed **42** default; train ratio default 80/20; only images with ≥1 annotation relevant to the chosen task.
- Zoom clamp 0.1×–10×; min box size 5×5 px; paste offset +20/+20 with wrap-to-0 when out of image bounds.
- Angle convention: 0° = up-edge pointing up, clockwise positive, stored 0–360.
- No heavy imports at module import time in services (`ultralytics` imported inside functions) so tests stay fast.
- Windows paths: always build with `os.path.join` / `pathlib`; never hardcode `/` or `\\`.
- All commits in `E:\DataLabeling_Python`; commit message style `feat:`/`test:`/`docs:` prefixes.
- Run commands from `E:\DataLabeling_Python` (pytest picks up `pytest.ini` with `pythonpath = .`).

---

### Task 1: Project scaffold

**Files:**
- Create: `requirements.txt`, `pytest.ini`, `.gitignore`, `main.py`, `README.md`
- Create: `app/__init__.py`, `app/models/__init__.py`, `app/services/__init__.py`, `app/training/__init__.py`, `app/ui/__init__.py`, `tests/__init__.py`
- Test: `tests/test_scaffold.py`

**Interfaces:**
- Produces: importable `app` package; `pytest` runs from repo root.

- [ ] **Step 1: Create the files**

`requirements.txt`:
```
PySide6>=6.6
ultralytics>=8.3
opencv-python>=4.9
pyyaml>=6.0
pillow>=10.0
pytest>=8.0
```

`pytest.ini`:
```ini
[pytest]
pythonpath = .
testpaths = tests
```

`.gitignore`:
```
__pycache__/
*.pyc
.pytest_cache/
runs/
*.pt
*.onnx
.venv/
```

`main.py`:
```python
import sys

from PySide6.QtWidgets import QApplication


def main():
    from app.ui.main_window import MainWindow

    qapp = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(qapp.exec())


if __name__ == "__main__":
    main()
```

`README.md`:
```markdown
# DataLabeling Python

Image labeling tool (boxes, rotated boxes, polygons) with integrated YOLO training.
Python port of the C# WPF ImageLableing app.

## Setup

    pip install -r requirements.txt
    python main.py
```

All `__init__.py` files are empty.

`tests/test_scaffold.py`:
```python
def test_app_package_imports():
    import app  # noqa: F401
```

- [ ] **Step 2: Install deps and run test**

Run: `pip install -r requirements.txt` then `pytest -v`
Expected: `test_app_package_imports PASSED`

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat: project scaffold (PySide6 + ultralytics app skeleton)"
```

---

### Task 2: Data models (C#-compatible JSON round-trip)

**Files:**
- Create: `app/models/label_class.py`, `app/models/bounding_box.py`, `app/models/polygon.py`, `app/models/image_annotation.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces:
  - `LabelClass(name: str, color: str = "")` — auto color from `stable_color(name)`; `to_dict()/from_dict()` with keys `Name, Color`.
  - `stable_color(name: str) -> str` (hex like `"#FFD700"`).
  - `BoundingBox(class_name, x, y, width, height, angle=0.0, id=auto-uuid)`; property `center -> tuple[float, float]`; `to_dict()/from_dict()` with C# keys.
  - `PolygonAnnotation(class_name, points: list[tuple[float, float]], id=auto-uuid)`; `to_dict()/from_dict()`.
  - `ImageAnnotation(image_path: str, boxes: list[BoundingBox], polygons: list[PolygonAnnotation])`; `to_dict()/from_dict()`.

- [ ] **Step 1: Write the failing tests**

`tests/test_models.py`:
```python
from app.models.bounding_box import BoundingBox
from app.models.image_annotation import ImageAnnotation
from app.models.label_class import LabelClass, stable_color
from app.models.polygon import PolygonAnnotation


def test_stable_color_is_deterministic_hex():
    c1 = stable_color("Car")
    c2 = stable_color("Car")
    assert c1 == c2
    assert c1.startswith("#") and len(c1) == 7


def test_label_class_auto_color_and_dict():
    lc = LabelClass("Person")
    assert lc.color == stable_color("Person")
    d = lc.to_dict()
    assert d == {"Name": "Person", "Color": lc.color}
    assert LabelClass.from_dict(d) == lc


def test_bounding_box_center_and_csharp_dict_roundtrip():
    b = BoundingBox("Car", 10, 20, 100, 50, angle=30.0)
    assert b.center == (60.0, 45.0)
    d = b.to_dict()
    assert set(d) == {"Id", "ClassName", "X", "Y", "Width", "Height", "Angle"}
    b2 = BoundingBox.from_dict(d)
    assert (b2.x, b2.y, b2.width, b2.height, b2.angle, b2.class_name, b2.id) == (
        10, 20, 100, 50, 30.0, "Car", b.id)


def test_bounding_box_from_dict_defaults_angle_zero():
    b = BoundingBox.from_dict(
        {"Id": "i1", "ClassName": "Car", "X": 1, "Y": 2, "Width": 3, "Height": 4})
    assert b.angle == 0.0


def test_polygon_roundtrip_ignores_extra_keys():
    d = {"Id": "p1", "ClassName": "Person", "Tag": "legacy-guid",
         "Points": [{"X": 1, "Y": 2}, {"X": 3, "Y": 4}, {"X": 5, "Y": 6}]}
    p = PolygonAnnotation.from_dict(d)
    assert p.points == [(1, 2), (3, 4), (5, 6)]
    assert p.to_dict()["Points"][0] == {"X": 1, "Y": 2}


def test_image_annotation_roundtrip_and_filename():
    ann = ImageAnnotation(
        image_path=r"C:\data\img1.jpg",
        boxes=[BoundingBox("Car", 0, 0, 10, 10)],
        polygons=[PolygonAnnotation("Person", [(0, 0), (1, 0), (1, 1)])])
    d = ann.to_dict()
    assert d["ImageFileName"] == "img1.jpg"
    ann2 = ImageAnnotation.from_dict(d)
    assert len(ann2.boxes) == 1 and len(ann2.polygons) == 1
    assert ann2.boxes[0].class_name == "Car"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.models.label_class'`

- [ ] **Step 3: Implement the models**

`app/models/label_class.py`:
```python
import hashlib
from dataclasses import dataclass

PALETTE = [
    "#FFD700", "#FF6347", "#32CD32", "#1E90FF", "#FF69B4", "#8A2BE2",
    "#00CED1", "#FFA500", "#ADFF2F", "#DC143C", "#00FA9A", "#4169E1",
    "#FF4500", "#9370DB", "#20B2AA", "#F08080",
]


def stable_color(name: str) -> str:
    digest = int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16)
    return PALETTE[digest % len(PALETTE)]


@dataclass
class LabelClass:
    name: str
    color: str = ""

    def __post_init__(self):
        if not self.color:
            self.color = stable_color(self.name)

    def to_dict(self) -> dict:
        return {"Name": self.name, "Color": self.color}

    @classmethod
    def from_dict(cls, d: dict) -> "LabelClass":
        return cls(d["Name"], d.get("Color", ""))
```

`app/models/bounding_box.py`:
```python
import uuid
from dataclasses import dataclass, field


@dataclass
class BoundingBox:
    class_name: str
    x: float
    y: float
    width: float
    height: float
    angle: float = 0.0  # degrees, 0 = upright, clockwise
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    def clone(self) -> "BoundingBox":
        return BoundingBox(self.class_name, self.x, self.y,
                           self.width, self.height, self.angle)

    def to_dict(self) -> dict:
        return {"Id": self.id, "ClassName": self.class_name, "X": self.x,
                "Y": self.y, "Width": self.width, "Height": self.height,
                "Angle": self.angle}

    @classmethod
    def from_dict(cls, d: dict) -> "BoundingBox":
        return cls(d["ClassName"], d["X"], d["Y"], d["Width"], d["Height"],
                   d.get("Angle", 0.0), d.get("Id") or str(uuid.uuid4()))
```

`app/models/polygon.py`:
```python
import uuid
from dataclasses import dataclass, field


@dataclass
class PolygonAnnotation:
    class_name: str
    points: list  # list[tuple[float, float]]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        return {"Id": self.id, "ClassName": self.class_name,
                "Points": [{"X": x, "Y": y} for x, y in self.points]}

    @classmethod
    def from_dict(cls, d: dict) -> "PolygonAnnotation":
        pts = [(p["X"], p["Y"]) for p in d.get("Points", [])]
        return cls(d["ClassName"], pts, d.get("Id") or str(uuid.uuid4()))
```

`app/models/image_annotation.py`:
```python
import os
from dataclasses import dataclass, field

from app.models.bounding_box import BoundingBox
from app.models.polygon import PolygonAnnotation


@dataclass
class ImageAnnotation:
    image_path: str
    boxes: list = field(default_factory=list)
    polygons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"ImagePath": self.image_path,
                "ImageFileName": os.path.basename(self.image_path),
                "BoundingBoxes": [b.to_dict() for b in self.boxes],
                "Polygons": [p.to_dict() for p in self.polygons]}

    @classmethod
    def from_dict(cls, d: dict) -> "ImageAnnotation":
        return cls(d.get("ImagePath", ""),
                   [BoundingBox.from_dict(b) for b in d.get("BoundingBoxes", [])],
                   [PolygonAnnotation.from_dict(p) for p in d.get("Polygons", [])])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add app/models tests/test_models.py && git commit -m "feat: annotation dataclasses with C#-compatible JSON serialization"
```

---

### Task 3: Geometry service

**Files:**
- Create: `app/services/geometry.py`
- Test: `tests/test_geometry.py`

**Interfaces:**
- Produces:
  - `rotate_point(px, py, cx, cy, angle_deg) -> tuple[float, float]` — rotate point around center, clockwise-positive in screen coords (y down).
  - `box_corners(box: BoundingBox) -> list[tuple]` — 4 corners clockwise from top-left, rotated by `box.angle` around box center.
  - `point_in_box(box, x, y) -> bool` — rotation-aware containment.
  - `point_in_polygon(points: list[tuple], x, y) -> bool` — ray casting.
  - `aabb_of(points) -> tuple[x, y, w, h]` — axis-aligned bounds of a point list.
  - `angle_from_center(cx, cy, x, y) -> float` — degrees 0–360, 0 = straight up from center, clockwise.
  - `clamp01(v) -> float`

- [ ] **Step 1: Write the failing tests**

`tests/test_geometry.py`:
```python
import math

from app.models.bounding_box import BoundingBox
from app.services import geometry as geo


def test_rotate_point_90_clockwise_screen_coords():
    # y grows downward: rotating the "up" vector (0,-1) by +90° gives "right" (1,0)
    x, y = geo.rotate_point(0, -1, 0, 0, 90)
    assert math.isclose(x, 1, abs_tol=1e-9) and math.isclose(y, 0, abs_tol=1e-9)


def test_box_corners_unrotated():
    b = BoundingBox("c", 100, 100, 50, 20)
    assert geo.box_corners(b) == [(100, 100), (150, 100), (150, 120), (100, 120)]


def test_box_corners_rotated_90():
    b = BoundingBox("c", 0, 0, 40, 20, angle=90)
    corners = geo.box_corners(b)
    expected = [(30, -10), (30, 30), (10, 30), (10, -10)]
    for (cx, cy), (ex, ey) in zip(corners, expected):
        assert math.isclose(cx, ex, abs_tol=1e-9)
        assert math.isclose(cy, ey, abs_tol=1e-9)


def test_point_in_box_respects_rotation():
    b = BoundingBox("c", 0, 0, 40, 20, angle=90)
    assert geo.point_in_box(b, 25, 25)          # inside only when rotated
    assert not geo.point_in_box(BoundingBox("c", 0, 0, 40, 20), 25, 25)


def test_point_in_polygon():
    tri = [(0, 0), (10, 0), (0, 10)]
    assert geo.point_in_polygon(tri, 2, 2)
    assert not geo.point_in_polygon(tri, 8, 8)


def test_aabb_of():
    assert geo.aabb_of([(30, -10), (30, 30), (10, 30), (10, -10)]) == (10, -10, 20, 40)


def test_angle_from_center_zero_is_up_clockwise():
    assert math.isclose(geo.angle_from_center(0, 0, 0, -5), 0.0, abs_tol=1e-9)
    assert math.isclose(geo.angle_from_center(0, 0, 5, 0), 90.0, abs_tol=1e-9)
    assert math.isclose(geo.angle_from_center(0, 0, 0, 5), 180.0, abs_tol=1e-9)


def test_clamp01():
    assert geo.clamp01(-0.2) == 0.0 and geo.clamp01(1.7) == 1.0 and geo.clamp01(0.5) == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_geometry.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

`app/services/geometry.py`:
```python
"""Pure geometry helpers. Screen coordinate convention: y grows downward,
positive angles rotate clockwise, angle 0 = pointing up."""
import math


def rotate_point(px, py, cx, cy, angle_deg):
    a = math.radians(angle_deg)
    dx, dy = px - cx, py - cy
    return (cx + dx * math.cos(a) - dy * math.sin(a),
            cy + dx * math.sin(a) + dy * math.cos(a))


def box_corners(box):
    cx, cy = box.center
    pts = [(box.x, box.y), (box.x + box.width, box.y),
           (box.x + box.width, box.y + box.height), (box.x, box.y + box.height)]
    if not box.angle:
        return pts
    return [rotate_point(px, py, cx, cy, box.angle) for px, py in pts]


def point_in_box(box, x, y):
    cx, cy = box.center
    lx, ly = rotate_point(x, y, cx, cy, -box.angle)
    return box.x <= lx <= box.x + box.width and box.y <= ly <= box.y + box.height


def point_in_polygon(points, x, y):
    inside = False
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            t = (y - y1) / (y2 - y1)
            if x < x1 + t * (x2 - x1):
                inside = not inside
    return inside


def aabb_of(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def angle_from_center(cx, cy, x, y):
    return math.degrees(math.atan2(x - cx, -(y - cy))) % 360


def clamp01(v):
    return max(0.0, min(1.0, v))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_geometry.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/geometry.py tests/test_geometry.py && git commit -m "feat: rotation-aware geometry service"
```

---

### Task 4: File service (JSON persistence, folder scan, classes)

**Files:**
- Create: `app/services/file_service.py`
- Test: `tests/test_file_service.py`

**Interfaces:**
- Consumes: `ImageAnnotation`, `LabelClass` from Task 2.
- Produces:
  - `IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}`
  - `scan_images(folder: str) -> list[str]` — absolute paths, name-sorted.
  - `labels_dir(folder) -> str` — `<folder>/labels`, created on demand by save.
  - `save_annotation(folder, ann: ImageAnnotation) -> str` — writes `labels/<stem>.json`, returns path.
  - `load_annotation(folder, image_path) -> ImageAnnotation | None` — None if missing or corrupt (never raises).
  - `save_all_annotations(folder, anns: list[ImageAnnotation]) -> str` — `annotations/annotations.json` with `ExportDate, Images`.
  - `save_classes(folder, classes: list[LabelClass])`, `load_classes(folder) -> list[LabelClass]` — `<folder>/classes.json`.

- [ ] **Step 1: Write the failing tests**

`tests/test_file_service.py`:
```python
import json
import os

from app.models.bounding_box import BoundingBox
from app.models.image_annotation import ImageAnnotation
from app.models.label_class import LabelClass
from app.services import file_service as fs


def _touch(p):
    open(p, "wb").close()


def test_scan_images_filters_and_sorts(tmp_path):
    for name in ["b.jpg", "a.PNG", "c.txt", "d.gif"]:
        _touch(tmp_path / name)
    result = fs.scan_images(str(tmp_path))
    assert [os.path.basename(p) for p in result] == ["a.PNG", "b.jpg", "d.gif"]


def test_save_and_load_annotation_roundtrip(tmp_path):
    img = str(tmp_path / "img1.jpg")
    _touch(img)
    ann = ImageAnnotation(img, boxes=[BoundingBox("Car", 1, 2, 3, 4, angle=15)])
    path = fs.save_annotation(str(tmp_path), ann)
    assert path == os.path.join(str(tmp_path), "labels", "img1.json")
    loaded = fs.load_annotation(str(tmp_path), img)
    assert loaded.boxes[0].angle == 15
    assert loaded.boxes[0].class_name == "Car"


def test_load_annotation_missing_returns_none(tmp_path):
    assert fs.load_annotation(str(tmp_path), str(tmp_path / "nope.jpg")) is None


def test_load_annotation_corrupt_returns_none(tmp_path):
    os.makedirs(tmp_path / "labels")
    (tmp_path / "labels" / "img1.json").write_text("{not json", encoding="utf-8")
    assert fs.load_annotation(str(tmp_path), str(tmp_path / "img1.jpg")) is None


def test_save_all_annotations(tmp_path):
    anns = [ImageAnnotation(str(tmp_path / "a.jpg"), boxes=[BoundingBox("Car", 0, 0, 5, 5)])]
    path = fs.save_all_annotations(str(tmp_path), anns)
    data = json.loads(open(path, encoding="utf-8").read())
    assert "ExportDate" in data and len(data["Images"]) == 1


def test_classes_roundtrip(tmp_path):
    fs.save_classes(str(tmp_path), [LabelClass("Car"), LabelClass("Person")])
    loaded = fs.load_classes(str(tmp_path))
    assert [c.name for c in loaded] == ["Car", "Person"]


def test_load_classes_missing_returns_empty(tmp_path):
    assert fs.load_classes(str(tmp_path)) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_file_service.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

`app/services/file_service.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_file_service.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/file_service.py tests/test_file_service.py && git commit -m "feat: JSON annotation persistence compatible with C# app"
```

---

### Task 5: Export service (YOLO detect / OBB / segmentation)

**Files:**
- Create: `app/services/export_service.py`
- Test: `tests/test_export_service.py`

**Interfaces:**
- Consumes: models (Task 2), `geometry.box_corners/aabb_of/clamp01` (Task 3).
- Produces:
  - `detect_line(box, cls_id, img_w, img_h) -> str` — `"{id} {cx} {cy} {w} {h}"`, 6 decimals; rotated boxes use the AABB of their rotated corners.
  - `obb_line(box, cls_id, img_w, img_h) -> str` — 8 normalized clamped corner coords, clockwise from top-left.
  - `seg_line(poly, cls_id, img_w, img_h) -> str` — normalized clamped vertex list.
  - `split_names(names: list[str], train_ratio: float, seed: int = 42) -> tuple[list, list]` — deterministic shuffle-split.
  - `export_dataset(annotations, class_names, out_dir, task, train_ratio=0.8, copy_images=True, write_yaml=True, seed=42, progress_cb=None) -> dict` — returns `{"train": int, "val": int, "yaml": str | None}`. `task` in `{"detect", "obb", "segment"}`. Creates `train/images`, `train/labels`, `val/images`, `val/labels`, `data.yaml`. Only images with ≥1 relevant annotation (boxes for detect/obb, polygons for segment). Image size read with Pillow.

- [ ] **Step 1: Write the failing tests**

`tests/test_export_service.py`:
```python
import os

import yaml
from PIL import Image

from app.models.bounding_box import BoundingBox
from app.models.image_annotation import ImageAnnotation
from app.models.polygon import PolygonAnnotation
from app.services import export_service as ex


def test_detect_line_unrotated():
    b = BoundingBox("c", 50, 50, 100, 50)
    assert ex.detect_line(b, 0, 200, 100) == "0 0.500000 0.750000 0.500000 0.500000"


def test_detect_line_rotated_uses_aabb():
    b = BoundingBox("c", 0, 0, 40, 20, angle=90)  # AABB: (10,-10,20,40)
    line = ex.detect_line(b, 1, 100, 100)
    parts = line.split()
    assert parts[0] == "1"
    # cy = (-10 + 40/2)/100 = 0.10; h clamped inside [0,1]: 40/100
    assert parts[2] == "0.100000" and parts[4] == "0.400000"


def test_obb_line_corners_normalized():
    b = BoundingBox("c", 100, 100, 50, 20)
    line = ex.obb_line(b, 2, 200, 200)
    assert line == ("2 0.500000 0.500000 0.750000 0.500000 "
                    "0.750000 0.600000 0.500000 0.600000")


def test_obb_line_clamps_out_of_bounds():
    b = BoundingBox("c", 0, 0, 40, 20, angle=90)  # corner y=-10 → clamp 0
    values = [float(v) for v in ex.obb_line(b, 0, 100, 100).split()[1:]]
    assert all(0.0 <= v <= 1.0 for v in values)


def test_seg_line():
    p = PolygonAnnotation("c", [(0, 0), (50, 0), (50, 100)])
    assert ex.seg_line(p, 3, 100, 100) == "3 0.000000 0.000000 0.500000 0.000000 0.500000 1.000000"


def test_split_names_deterministic_and_complete():
    names = [f"img{i}" for i in range(10)]
    t1, v1 = ex.split_names(names, 0.8, seed=42)
    t2, v2 = ex.split_names(names, 0.8, seed=42)
    assert t1 == t2 and v1 == v2
    assert len(t1) == 8 and len(v1) == 2
    assert sorted(t1 + v1) == sorted(names)


def _make_dataset(tmp_path, n=5):
    anns = []
    for i in range(n):
        img = str(tmp_path / f"img{i}.jpg")
        Image.new("RGB", (100, 80)).save(img)
        anns.append(ImageAnnotation(
            img,
            boxes=[BoundingBox("Car", 10, 10, 30, 20)],
            polygons=[PolygonAnnotation("Car", [(0, 0), (10, 0), (10, 10)])]))
    return anns


def test_export_dataset_detect(tmp_path):
    anns = _make_dataset(tmp_path)
    out = str(tmp_path / "out")
    result = ex.export_dataset(anns, ["Car"], out, "detect", train_ratio=0.8)
    assert result["train"] == 4 and result["val"] == 1
    assert len(os.listdir(os.path.join(out, "train", "images"))) == 4
    assert len(os.listdir(os.path.join(out, "train", "labels"))) == 4
    data = yaml.safe_load(open(result["yaml"], encoding="utf-8"))
    assert data["nc"] == 1 and data["names"] == ["Car"]
    assert data["train"] == "train/images" and data["val"] == "val/images"


def test_export_dataset_segment_skips_images_without_polygons(tmp_path):
    img = str(tmp_path / "only_box.jpg")
    Image.new("RGB", (50, 50)).save(img)
    anns = [ImageAnnotation(img, boxes=[BoundingBox("Car", 0, 0, 5, 5)])]
    result = ex.export_dataset(anns, ["Car"], str(tmp_path / "o"), "segment")
    assert result["train"] == 0 and result["val"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_export_service.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

`app/services/export_service.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_export_service.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/export_service.py tests/test_export_service.py && git commit -m "feat: YOLO detect/OBB/segmentation export with deterministic split"
```

---

### Task 6: LabelCanvas core (image display, zoom at cursor, pan, fit)

**Files:**
- Create: `app/ui/label_canvas.py`
- Manual test via demo entry: `python -m app.ui.label_canvas <path-to-image>`

**Interfaces:**
- Consumes: geometry (Task 3), models (Task 2).
- Produces (used by every later UI task):
  - `class DrawMode(Enum): BOX, POLYGON`; `class Tool(Enum): POINTER, PAN`
  - `LabelCanvas(QGraphicsView)` with attributes `boxes: list[BoundingBox]`, `polygons: list[PolygonAnnotation]`, `selected_box`, `selected_polygon`, `class_colors: dict[str, str]`, `current_class: str`, `labeling_enabled: bool`, `draw_mode`, `tool`.
  - Methods this task delivers: `set_image(path: str) -> bool`, `clear_image()`, `fit_image()`, `image_size() -> tuple[int, int]`.
  - Signals: `annotation_changed = Signal()`, `selection_changed = Signal()`, `status_message = Signal(str)`, `mouse_moved = Signal(float, float)` (image-pixel coords).
  - Zoom: wheel zooms at cursor, total scale clamped 0.1–10. Pan: middle-drag always; left-drag when `tool == Tool.PAN`. `fit_image()` fits and centers.

- [ ] **Step 1: Implement the canvas core**

`app/ui/label_canvas.py`:
```python
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
```

- [ ] **Step 2: Manual verify**

Run: `python -m app.ui.label_canvas <any .jpg on disk>`
Checklist:
- Image shows fitted and centered.
- Mouse wheel zooms in/out **toward the cursor position** (point under cursor stays put).
- Zoom stops at ~10× in and ~0.1× out.
- Middle-mouse drag pans; release restores arrow cursor.

- [ ] **Step 3: Commit**

```bash
git add app/ui/label_canvas.py && git commit -m "feat: canvas core - image display, cursor zoom, pan, fit"
```

---

### Task 7: LabelCanvas boxes (draw, select, move, resize, delete, copy/paste)

**Files:**
- Modify: `app/ui/label_canvas.py`

**Interfaces:**
- Produces (consumed by MainWindow in Task 10):
  - `copy_selected()`, `paste()`, `delete_selected()` methods.
  - Drawing: left-drag in empty area creates a `BoundingBox(current_class, ...)` (min 5×5) and emits `annotation_changed`.
  - Click on box selects it (topmost wins) and emits `selection_changed`; drag moves; drag near edge/corner (10 px screen threshold) resizes with 8 directions; `Escape`-free — releasing mouse commits.
  - All shapes render with class color from `class_colors`, selected shape yellow/thicker, class name badge above.

- [ ] **Step 1: Add hit-testing and interaction to `LabelCanvas`**

Replace `mousePressEvent`, `mouseMoveEvent`, `mouseReleaseEvent` and add the following methods to `app/ui/label_canvas.py`:

```python
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
        return False  # stub — replaced with the real hit test in Task 8

    def _polygon_click(self, sx, sy):
        pass  # stub — implemented in Task 9

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
```

Also add rendering — `drawForeground` plus helpers (rotation-handle drawing is used from Task 8 but harmless now):

```python
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
```

- [ ] **Step 2: Manual verify**

Add temporary demo state to the harness (`__main__` block) before `canvas.show()`:
```python
    canvas.class_colors = {"Car": "#FFD700"}
    canvas.current_class = "Car"
```
Run: `python -m app.ui.label_canvas <image>`
Checklist:
- Drag on empty area draws a dashed green rect; release creates a gold box with "Car" badge; boxes <5×5 are discarded.
- Click a box → turns yellow with 8 square handles + blue circle above.
- Drag body moves; drag each edge/corner resizes in the right direction with matching cursor; box never inverts (min 5 px).
- Click empty space deselects.

- [ ] **Step 3: Commit**

```bash
git add app/ui/label_canvas.py && git commit -m "feat: canvas box draw/select/move/resize + copy-paste API"
```

---

### Task 8: LabelCanvas rotation

**Files:**
- Modify: `app/ui/label_canvas.py`

**Interfaces:**
- Produces: dragging the blue circle handle rotates the selected box around its center (`angle_from_center`, 0° = up, clockwise, 0–360); hit tests on rotated boxes work (already wired via `point_in_box` / local-frame `_resize_dir`).

- [ ] **Step 1: Replace the Task-7 stub with the real rotation-handle hit test**

In `LabelCanvas`, replace the `_hit_rotation_handle` stub (`return False`) with:
```python
    def _hit_rotation_handle(self, box, sx, sy):
        cx, cy = box.center
        off = ROT_OFFSET_PX / self._scale()
        hx, hy = geo.rotate_point(cx, box.y - off, cx, cy, box.angle)
        r = 10.0 / self._scale()
        return (sx - hx) ** 2 + (sy - hy) ** 2 <= r * r
```
(The `rotate`/`resize` drag kinds and handle drawing were already added in Task 7 — this method completes the wiring.)

- [ ] **Step 2: Manual verify**

Run: `python -m app.ui.label_canvas <image>`
Checklist:
- Draw a box, select it, drag the blue circle → box rotates smoothly around its center; straight up = 0°.
- After rotating ~45°, clicking inside the rotated box selects it; clicking where the unrotated box used to be (now empty) does not.
- Resizing a rotated box pulls the correct (rotated) edge; moving works.

- [ ] **Step 3: Commit**

```bash
git add app/ui/label_canvas.py && git commit -m "feat: rotated bounding boxes with drag handle"
```

---

### Task 9: LabelCanvas polygons

**Files:**
- Modify: `app/ui/label_canvas.py`

**Interfaces:**
- Produces: in `DrawMode.POLYGON`, left-clicks append vertices; clicking within 12 screen px of the first vertex with ≥3 points closes the polygon (creates `PolygonAnnotation(current_class, points)`, emits `annotation_changed`). Selected polygons can be moved (body drag) and have vertices dragged. `cancel_pending_polygon()` clears an unfinished polygon.

- [ ] **Step 1: Add polygon methods (replace the Task-7 `_polygon_click` stub)**

Add to `LabelCanvas`:
```python
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
```

Note: in the polygon branch, also allow moving a selected polygon by body-drag — the generic `_topmost_polygon_at` + `move_poly` path in `mousePressEvent` already covers this in BOX mode; in POLYGON mode selection happens via `_polygon_click` above.

- [ ] **Step 2: Manual verify**

In the demo `__main__` block, add `canvas.draw_mode = DrawMode.POLYGON`.
Run: `python -m app.ui.label_canvas <image>`
Checklist:
- Clicks drop green dots connected by lines.
- With ≥3 points, clicking near the first point closes → colored polygon with class badge, selected (yellow) with vertex squares.
- Dragging a vertex square reshapes it. Delete works via `delete_selected` (verify in Task 10 with the shortcut).
- Switch back to `DrawMode.BOX` in the demo: box interactions unaffected.

- [ ] **Step 3: Commit**

```bash
git add app/ui/label_canvas.py && git commit -m "feat: polygon drawing and vertex editing"
```

---

### Task 10: MainWindow + ManageClassesDialog + shortcuts

**Files:**
- Create: `app/ui/main_window.py`, `app/ui/manage_classes_dialog.py`

**Interfaces:**
- Consumes: `LabelCanvas` (Tasks 6–9), `file_service` (Task 4), models.
- Produces:
  - `MainWindow(QMainWindow)` with attributes later dialogs read: `classes: list[LabelClass]`, `image_paths: list[str]`, `folder: str | None`, `store: dict[str, ImageAnnotation]` (in-memory annotations keyed by image path), `last_export_yaml: str | None`.
  - Methods: `open_folder()`, `show_image(index: int)`, `next_image()`, `prev_image()`, `save_current()`, `save_all()`, `current_annotations() -> list[ImageAnnotation]` (flushes canvas into store first).
  - `ManageClassesDialog(classes, store, parent)` — add/delete classes; delete warns with usage count and cascades removal from every `ImageAnnotation` in `store`.

- [ ] **Step 1: Implement `ManageClassesDialog`**

`app/ui/manage_classes_dialog.py`:
```python
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QInputDialog, QListWidget,
                               QMessageBox, QPushButton, QVBoxLayout)

from app.models.label_class import LabelClass


class ManageClassesDialog(QDialog):
    """Add/remove label classes; removal cascades across all annotations."""

    def __init__(self, classes, store, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Quản lý Classes")
        self.classes = classes
        self.store = store
        layout = QVBoxLayout(self)
        self.listw = QListWidget()
        layout.addWidget(self.listw)
        row = QHBoxLayout()
        add_btn = QPushButton("Thêm...")
        del_btn = QPushButton("Xóa")
        close_btn = QPushButton("Đóng")
        row.addWidget(add_btn)
        row.addWidget(del_btn)
        row.addStretch()
        row.addWidget(close_btn)
        layout.addLayout(row)
        add_btn.clicked.connect(self.add_class)
        del_btn.clicked.connect(self.delete_class)
        close_btn.clicked.connect(self.accept)
        self.refresh()

    def refresh(self):
        self.listw.clear()
        for c in self.classes:
            self.listw.addItem(f"{c.name}  ({c.color})")

    def add_class(self):
        name, ok = QInputDialog.getText(self, "Thêm class", "Tên class:")
        name = name.strip()
        if not ok or not name:
            return
        if any(c.name == name for c in self.classes):
            QMessageBox.warning(self, "Trùng tên", f"Class '{name}' đã tồn tại.")
            return
        self.classes.append(LabelClass(name))
        self.refresh()

    def delete_class(self):
        row = self.listw.currentRow()
        if row < 0:
            return
        cls = self.classes[row]
        used = sum(
            sum(1 for b in ann.boxes if b.class_name == cls.name)
            + sum(1 for p in ann.polygons if p.class_name == cls.name)
            for ann in self.store.values())
        msg = f"Xóa class '{cls.name}'?"
        if used:
            msg += f"\nClass đang được dùng bởi {used} nhãn — xóa sẽ gỡ khỏi TẤT CẢ ảnh."
        if QMessageBox.question(self, "Xác nhận", msg) != QMessageBox.Yes:
            return
        for ann in self.store.values():
            ann.boxes = [b for b in ann.boxes if b.class_name != cls.name]
            ann.polygons = [p for p in ann.polygons if p.class_name != cls.name]
        del self.classes[row]
        self.refresh()
```

- [ ] **Step 2: Implement `MainWindow`**

`app/ui/main_window.py`:
```python
import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (QComboBox, QFileDialog, QLabel, QListWidget,
                               QMainWindow, QMessageBox, QSplitter, QToolBar,
                               QWidget)

from app.models.image_annotation import ImageAnnotation
from app.models.label_class import LabelClass
from app.services import file_service as fs
from app.ui.label_canvas import DrawMode, LabelCanvas, Tool
from app.ui.manage_classes_dialog import ManageClassesDialog

DEFAULT_CLASSES = ["Car", "Person", "Motorcycle"]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DataLabeling Python")
        self.resize(1400, 900)

        self.folder = None
        self.image_paths = []
        self.index = -1
        self.store = {}          # image_path -> ImageAnnotation
        self.classes = [LabelClass(n) for n in DEFAULT_CLASSES]
        self.last_export_yaml = None

        self.canvas = LabelCanvas()
        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self._on_list_row)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.image_list)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([220, 1000])
        self.setCentralWidget(splitter)

        self._build_toolbar()
        self._build_statusbar()
        self._build_shortcuts()
        self._sync_class_combo()

        self.canvas.annotation_changed.connect(self._flush_canvas_to_store)
        self.canvas.mouse_moved.connect(
            lambda x, y: self.coord_label.setText(f"x={x:.0f}, y={y:.0f}"))
        self.canvas.status_message.connect(self.statusBar().showMessage)

    # ---------- UI construction ----------

    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)
        tb.addAction(QAction("Mở thư mục...", self, triggered=self.open_folder))
        tb.addAction(QAction("Lưu (Ctrl+S)", self, triggered=self.save_current))
        tb.addAction(QAction("Lưu tất cả", self, triggered=self.save_all))
        tb.addSeparator()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Box", "Polygon"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        tb.addWidget(QLabel(" Chế độ: "))
        tb.addWidget(self.mode_combo)
        self.tool_combo = QComboBox()
        self.tool_combo.addItems(["Pointer (Vẽ ROI)", "Touch (Kéo ảnh)"])
        self.tool_combo.currentIndexChanged.connect(
            lambda i: setattr(self.canvas, "tool", Tool.PAN if i else Tool.POINTER))
        tb.addWidget(QLabel(" Công cụ: "))
        tb.addWidget(self.tool_combo)
        self.class_combo = QComboBox()
        self.class_combo.currentTextChanged.connect(
            lambda name: setattr(self.canvas, "current_class", name))
        tb.addWidget(QLabel(" Class: "))
        tb.addWidget(self.class_combo)
        tb.addAction(QAction("Quản lý classes...", self, triggered=self.manage_classes))
        tb.addSeparator()
        tb.addAction(QAction("Fit ảnh (0)", self, triggered=self.canvas.fit_image))
        tb.addSeparator()
        tb.addAction(QAction("Auto Label...", self, triggered=self.open_auto_label))
        tb.addAction(QAction("Export...", self, triggered=self.open_export))
        tb.addAction(QAction("Train...", self, triggered=self.open_train))

    def _build_statusbar(self):
        self.coord_label = QLabel("x=–, y=–")
        self.counter_label = QLabel("Ảnh 0 / 0")
        self.statusBar().addPermanentWidget(self.coord_label)
        self.statusBar().addPermanentWidget(self.counter_label)

    def _build_shortcuts(self):
        def sc(key, fn):
            # Default WindowShortcut context on purpose: plain-letter shortcuts
            # (A/D/1-9) must NOT steal keystrokes from QLineEdits in child
            # dialogs (Export/Train/AutoLabel are separate windows, unaffected).
            s = QShortcut(QKeySequence(key), self)
            s.activated.connect(fn)
        sc("Ctrl+S", self.save_current)
        sc("Ctrl+C", self.canvas.copy_selected)
        sc("Ctrl+V", self.canvas.paste)
        sc("Delete", self.canvas.delete_selected)
        sc("D", self.next_image)
        sc("Right", self.next_image)
        sc("A", self.prev_image)
        sc("Left", self.prev_image)
        sc("0", self.canvas.fit_image)
        sc("Escape", self.canvas.cancel_pending_polygon)
        for i in range(1, 10):
            sc(str(i), lambda i=i: self._select_class_index(i - 1))

    # ---------- classes ----------

    def _sync_class_combo(self):
        current = self.class_combo.currentText()
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        self.class_combo.addItems([c.name for c in self.classes])
        if current in [c.name for c in self.classes]:
            self.class_combo.setCurrentText(current)
        self.class_combo.blockSignals(False)
        self.canvas.class_colors = {c.name: c.color for c in self.classes}
        self.canvas.current_class = self.class_combo.currentText()
        if self.folder:
            fs.save_classes(self.folder, self.classes)

    def _select_class_index(self, i):
        if 0 <= i < self.class_combo.count():
            self.class_combo.setCurrentIndex(i)

    def manage_classes(self):
        dlg = ManageClassesDialog(self.classes, self.store, self)
        dlg.exec()
        self._sync_class_combo()
        self._load_canvas_from_store()

    # ---------- folder / navigation ----------

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục ảnh")
        if not folder:
            return
        self.folder = folder
        self.image_paths = fs.scan_images(folder)
        stored = fs.load_classes(folder)
        if stored:
            self.classes = stored
        self._sync_class_combo()
        self.store = {}
        for p in self.image_paths:          # eager-load existing labels
            ann = fs.load_annotation(folder, p)
            self.store[p] = ann or ImageAnnotation(p)
        self.image_list.clear()
        self.image_list.addItems([os.path.basename(p) for p in self.image_paths])
        if self.image_paths:
            self.image_list.setCurrentRow(0)
        else:
            self.canvas.clear_image()
            QMessageBox.information(self, "Trống", "Thư mục không có ảnh.")
        self._update_counter()

    def _on_list_row(self, row):
        if 0 <= row < len(self.image_paths):
            self.show_image(row)

    def show_image(self, index):
        self._flush_canvas_to_store()
        self.index = index
        path = self.image_paths[index]
        self.canvas.set_image(path)
        ann = self.store[path]
        self.canvas.boxes = ann.boxes
        self.canvas.polygons = ann.polygons
        self.canvas.viewport().update()
        self._update_counter()
        if self.image_list.currentRow() != index:
            self.image_list.setCurrentRow(index)

    def next_image(self):
        if self.index < len(self.image_paths) - 1:
            self.show_image(self.index + 1)

    def prev_image(self):
        if self.index > 0:
            self.show_image(self.index - 1)

    def _update_counter(self):
        self.counter_label.setText(
            f"Ảnh {self.index + 1} / {len(self.image_paths)}")

    # ---------- persistence ----------

    def _flush_canvas_to_store(self):
        if 0 <= self.index < len(self.image_paths):
            path = self.image_paths[self.index]
            self.store[path].boxes = self.canvas.boxes
            self.store[path].polygons = self.canvas.polygons

    def _load_canvas_from_store(self):
        if 0 <= self.index < len(self.image_paths):
            ann = self.store[self.image_paths[self.index]]
            self.canvas.boxes = ann.boxes
            self.canvas.polygons = ann.polygons
            self.canvas.selected_box = None
            self.canvas.selected_polygon = None
            self.canvas.viewport().update()

    def current_annotations(self):
        self._flush_canvas_to_store()
        return list(self.store.values())

    def save_current(self):
        if not self.folder or self.index < 0:
            return
        self._flush_canvas_to_store()
        fs.save_annotation(self.folder, self.store[self.image_paths[self.index]])
        self.statusBar().showMessage("Đã lưu nhãn ảnh hiện tại", 3000)

    def save_all(self):
        if not self.folder:
            return
        self._flush_canvas_to_store()
        for ann in self.store.values():
            if ann.boxes or ann.polygons:
                fs.save_annotation(self.folder, ann)
        fs.save_all_annotations(self.folder, self.current_annotations())
        fs.save_classes(self.folder, self.classes)
        self.statusBar().showMessage("Đã lưu tất cả nhãn", 3000)

    # ---------- mode / dialogs (Export/AutoLabel/Train wired in later tasks) ----------

    def _on_mode_changed(self, text):
        self.canvas.draw_mode = DrawMode.POLYGON if text == "Polygon" else DrawMode.BOX
        self.canvas.cancel_pending_polygon()

    def open_export(self):
        from app.ui.export_dialog import ExportDialog
        ExportDialog(self).exec()

    def open_auto_label(self):
        from app.ui.auto_label_window import AutoLabelWindow
        win = AutoLabelWindow(self)
        win.show()

    def open_train(self):
        from app.ui.train_dialog import TrainDialog
        TrainDialog(self).exec()
```

Note: `open_export` / `open_auto_label` / `open_train` reference dialogs created in Tasks 11–14; until then those buttons raise ImportError — acceptable during development, they are the last three toolbar items. If you want a clean intermediate state, guard each with `try/except ImportError: QMessageBox.information(self, "Chưa có", "Tính năng đang được xây dựng")` and remove the guard in the task that adds the dialog.

- [ ] **Step 3: Manual verify**

Run: `python main.py`
Checklist:
- Open a folder of images → list fills, first image shows, counter "Ảnh 1 / N".
- Draw boxes/polygons; `A`/`D` and `←`/`→` switch images; annotations persist per image when navigating back.
- `Ctrl+S` writes `labels/<stem>.json`; restart app, reopen folder → annotations reload (fix over C# confirmed).
- `Ctrl+C`/`Ctrl+V` copies/pastes a box (+20/+20, wraps at edges); `Delete` deletes; `1`/`2`/`3` switches class combo; `0` fits.
- Quản lý classes: add a class (gets stable color), delete a used class → warning with count, shapes disappear from all images; `classes.json` written on save.
- Mode combo switches Box/Polygon; Escape cancels a half-drawn polygon; Touch tool pans with left-drag.

- [ ] **Step 4: Commit**

```bash
git add app/ui/main_window.py app/ui/manage_classes_dialog.py && git commit -m "feat: main window with navigation, persistence, shortcuts, class management"
```

---

### Task 11: Export dialog

**Files:**
- Create: `app/ui/export_dialog.py`

**Interfaces:**
- Consumes: `export_service.export_dataset` (Task 5), `MainWindow.current_annotations()`, `MainWindow.classes`, sets `MainWindow.last_export_yaml`.

- [ ] **Step 1: Implement**

`app/ui/export_dialog.py`:
```python
import os

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QProgressBar, QPushButton,
                               QSpinBox, QVBoxLayout)

from app.services.export_service import export_dataset


class _ExportWorker(QThread):
    progress = Signal(int, int)
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs

    def run(self):
        try:
            result = export_dataset(
                progress_cb=lambda d, t: self.progress.emit(d, t), **self.kwargs)
            self.done.emit(result)
        except Exception as e:  # surfaced to a message box
            self.failed.emit(str(e))


class ExportDialog(QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main = main_window
        self.worker = None
        self.setWindowTitle("Export YOLO Dataset")
        self.resize(460, 300)

        form = QFormLayout()
        self.task_combo = QComboBox()
        self.task_combo.addItems(["detect", "obb", "segment"])
        form.addRow("Task:", self.task_combo)

        out_row = QHBoxLayout()
        self.out_edit = QLineEdit()
        browse = QPushButton("...")
        browse.clicked.connect(self._browse)
        out_row.addWidget(self.out_edit)
        out_row.addWidget(browse)
        form.addRow("Thư mục xuất:", out_row)

        self.train_spin = QSpinBox()
        self.train_spin.setRange(1, 99)
        self.train_spin.setValue(80)
        self.val_label = QLabel("Val: 20%")
        self.train_spin.valueChanged.connect(
            lambda v: self.val_label.setText(f"Val: {100 - v}%"))
        ratio_row = QHBoxLayout()
        ratio_row.addWidget(self.train_spin)
        ratio_row.addWidget(self.val_label)
        form.addRow("Train %:", ratio_row)

        self.copy_check = QCheckBox("Copy ảnh")
        self.copy_check.setChecked(True)
        self.yaml_check = QCheckBox("Tạo data.yaml")
        self.yaml_check.setChecked(True)
        form.addRow(self.copy_check)
        form.addRow(self.yaml_check)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        anns = self.main.current_annotations()
        labeled = sum(1 for a in anns if a.boxes or a.polygons)
        layout.addWidget(QLabel(f"Ảnh có nhãn: {labeled} / {len(anns)}"))
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self._start)
        layout.addWidget(self.export_btn)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Chọn thư mục xuất")
        if d:
            self.out_edit.setText(d)

    def _start(self):
        out_dir = self.out_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "Thiếu", "Chọn thư mục xuất.")
            return
        if not self.main.classes:
            QMessageBox.warning(self, "Thiếu", "Chưa có class nào.")
            return
        anns = self.main.current_annotations()
        if not any(a.boxes or a.polygons for a in anns):
            QMessageBox.warning(self, "Thiếu", "Chưa có ảnh nào được gán nhãn.")
            return
        self.export_btn.setEnabled(False)
        self.worker = _ExportWorker({
            "annotations": anns,
            "class_names": [c.name for c in self.main.classes],
            "out_dir": out_dir,
            "task": self.task_combo.currentText(),
            "train_ratio": self.train_spin.value() / 100.0,
            "copy_images": self.copy_check.isChecked(),
            "write_yaml": self.yaml_check.isChecked()})
        self.worker.progress.connect(
            lambda d, t: (self.progress.setMaximum(t), self.progress.setValue(d)))
        self.worker.done.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _done(self, result):
        self.export_btn.setEnabled(True)
        if result["yaml"]:
            self.main.last_export_yaml = result["yaml"]
        QMessageBox.information(
            self, "Xong",
            f"Train: {result['train']} ảnh, Val: {result['val']} ảnh\n"
            f"data.yaml: {result['yaml'] or '(không tạo)'}")

    def _failed(self, msg):
        self.export_btn.setEnabled(True)
        QMessageBox.critical(self, "Lỗi export", msg)
```

- [ ] **Step 2: Manual verify**

Run: `python main.py`, label 3+ images (mix of boxes and ≥1 polygon), Export:
- `detect` → `train/labels/*.txt` contain `id cx cy w h` lines; count matches 80/20 split; `data.yaml` valid.
- `obb` → 8-coordinate lines; a rotated box's corners differ from unrotated.
- `segment` → only images having polygons are exported.
- Missing output dir / no labels → warning boxes, no crash.

- [ ] **Step 3: Commit**

```bash
git add app/ui/export_dialog.py && git commit -m "feat: export dialog for detect/obb/segment datasets"
```

---

### Task 12: Auto-label service + video service

**Files:**
- Create: `app/services/auto_label_service.py`, `app/services/video_service.py`
- Test: `tests/test_auto_label_service.py`, `tests/test_video_service.py`

**Interfaces:**
- Produces:
  - `boxes_from_result(result, class_names) -> list[BoundingBox]` — pure parser for one ultralytics `Results` object; handles both `result.obb` (xywhr, radians → degrees) and `result.boxes` (xyxy).
  - `class AutoLabelService: load_model(path) -> list[str]` (returns class names from model), `predict(image_path, conf=0.5, iou=0.45) -> list[BoundingBox]`, attribute `class_names`.
  - `save_yolo_txt(boxes, txt_path, img_w, img_h, class_names)` — writes normalized detect-format lines (reuses `export_service.detect_line`).
  - `extract_frames(video_path, out_dir, every_n=10, progress_cb=None) -> list[str]` — writes `frame_000000.jpg`… under `out_dir`, returns paths.

- [ ] **Step 1: Write the failing tests**

`tests/test_auto_label_service.py`:
```python
import math
import os

from app.models.bounding_box import BoundingBox
from app.services.auto_label_service import boxes_from_result, save_yolo_txt


class _T:  # minimal stand-in for a tensor: only .tolist() is used
    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_boxes_from_result_plain_detect():
    res = _NS(obb=None,
              boxes=_NS(xyxy=_T([[10.0, 20.0, 50.0, 80.0]]), cls=_T([1.0])))
    out = boxes_from_result(res, ["Car", "Person"])
    b = out[0]
    assert (b.x, b.y, b.width, b.height, b.angle) == (10, 20, 40, 60, 0.0)
    assert b.class_name == "Person"


def test_boxes_from_result_obb_converts_radians():
    res = _NS(obb=_NS(xywhr=_T([[50.0, 50.0, 40.0, 20.0, math.pi / 2]]),
                      cls=_T([0.0])),
              boxes=None)
    out = boxes_from_result(res, ["Car"])
    b = out[0]
    assert (b.x, b.y, b.width, b.height) == (30, 40, 40, 20)  # cx,cy → top-left
    assert math.isclose(b.angle, 90.0)


def test_boxes_from_result_unknown_class_id_falls_back_to_str():
    res = _NS(obb=None, boxes=_NS(xyxy=_T([[0, 0, 1, 1]]), cls=_T([7.0])))
    assert boxes_from_result(res, ["Car"])[0].class_name == "7"


def test_save_yolo_txt(tmp_path):
    txt = str(tmp_path / "img.txt")
    save_yolo_txt([BoundingBox("Car", 50, 50, 100, 50)], txt, 200, 100, ["Car"])
    assert open(txt).read().strip() == "0 0.500000 0.750000 0.500000 0.500000"


def test_save_yolo_txt_skips_unknown_classes(tmp_path):
    txt = str(tmp_path / "img.txt")
    save_yolo_txt([BoundingBox("Ghost", 0, 0, 10, 10)], txt, 100, 100, ["Car"])
    assert open(txt).read().strip() == ""
```

`tests/test_video_service.py`:
```python
import os

import cv2
import numpy as np

from app.services.video_service import extract_frames


def _make_video(path, frames=25, size=64):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (size, size))
    for i in range(frames):
        vw.write(np.full((size, size, 3), i * 10 % 255, dtype=np.uint8))
    vw.release()


def test_extract_frames_every_n(tmp_path):
    video = str(tmp_path / "v.mp4")
    _make_video(video, frames=25)
    out = str(tmp_path / "frames")
    paths = extract_frames(video, out, every_n=10)
    assert [os.path.basename(p) for p in paths] == [
        "frame_000000.jpg", "frame_000010.jpg", "frame_000020.jpg"]
    assert all(os.path.isfile(p) for p in paths)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_auto_label_service.py tests/test_video_service.py -v`
Expected: FAIL — modules not found

- [ ] **Step 3: Implement**

`app/services/auto_label_service.py`:
```python
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
```

`app/services/video_service.py`:
```python
import os

import cv2


def extract_frames(video_path, out_dir, every_n=10, progress_cb=None):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Không mở được video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    paths = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % every_n == 0:
                path = os.path.join(out_dir, f"frame_{idx:06d}.jpg")
                cv2.imwrite(path, frame)
                paths.append(path)
                if progress_cb:
                    progress_cb(idx, total)
            idx += 1
    finally:
        cap.release()
    return paths
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_auto_label_service.py tests/test_video_service.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/auto_label_service.py app/services/video_service.py tests/test_auto_label_service.py tests/test_video_service.py && git commit -m "feat: ultralytics auto-label service and video frame extraction"
```

---

### Task 13: Auto-label window

**Files:**
- Create: `app/ui/auto_label_window.py`

**Interfaces:**
- Consumes: `AutoLabelService`, `save_yolo_txt`, `extract_frames` (Task 12); `LabelCanvas` (reused for preview editing); `stable_color`.
- Produces: `AutoLabelWindow(QMainWindow)` — model load, conf/iou controls, Images mode (batch save-all + human-in-the-loop preview with Save & Next / Skip), Video mode (extract every Nth frame + auto-label each frame).

- [ ] **Step 1: Implement**

`app/ui/auto_label_window.py`:
```python
import os

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog,
                               QHBoxLayout, QLabel, QLineEdit, QMainWindow,
                               QMessageBox, QProgressBar, QPushButton,
                               QSpinBox, QVBoxLayout, QWidget)
from PIL import Image

from app.models.label_class import stable_color
from app.services.auto_label_service import AutoLabelService, save_yolo_txt
from app.services.file_service import scan_images
from app.services.video_service import extract_frames
from app.ui.label_canvas import LabelCanvas


class _BatchWorker(QThread):
    """Runs detection over a list of images; optionally saves txt directly."""
    progress = Signal(int, int, str)
    detected = Signal(str, list)      # image_path, list[BoundingBox]
    done = Signal(int)
    failed = Signal(str)

    def __init__(self, service, image_paths, conf, iou, out_dir=None):
        super().__init__()
        self.service = service
        self.image_paths = image_paths
        self.conf = conf
        self.iou = iou
        self.out_dir = out_dir        # None => preview mode (emit only)

    def run(self):
        try:
            for i, path in enumerate(self.image_paths):
                boxes = self.service.predict(path, self.conf, self.iou)
                if self.out_dir:
                    with Image.open(path) as im:
                        w, h = im.size
                    stem = os.path.splitext(os.path.basename(path))[0]
                    save_yolo_txt(boxes, os.path.join(self.out_dir, stem + ".txt"),
                                  w, h, self.service.class_names)
                else:
                    self.detected.emit(path, boxes)
                self.progress.emit(i + 1, len(self.image_paths), path)
            self.done.emit(len(self.image_paths))
        except Exception as e:
            self.failed.emit(str(e))


class _VideoWorker(QThread):
    progress = Signal(str)
    done = Signal(int)
    failed = Signal(str)

    def __init__(self, service, video_path, out_dir, every_n, conf, iou):
        super().__init__()
        self.service = service
        self.video_path = video_path
        self.out_dir = out_dir
        self.every_n = every_n
        self.conf = conf
        self.iou = iou

    def run(self):
        try:
            frames_dir = os.path.join(self.out_dir, "frames")
            labels_dir = os.path.join(self.out_dir, "labels")
            os.makedirs(labels_dir, exist_ok=True)
            self.progress.emit("Đang tách frame...")
            frames = extract_frames(self.video_path, frames_dir, self.every_n)
            for i, fp in enumerate(frames):
                boxes = self.service.predict(fp, self.conf, self.iou)
                with Image.open(fp) as im:
                    w, h = im.size
                stem = os.path.splitext(os.path.basename(fp))[0]
                save_yolo_txt(boxes, os.path.join(labels_dir, stem + ".txt"),
                              w, h, self.service.class_names)
                self.progress.emit(f"Frame {i + 1}/{len(frames)}")
            self.done.emit(len(frames))
        except Exception as e:
            self.failed.emit(str(e))


class AutoLabelWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto Label")
        self.resize(1200, 800)
        self.service = AutoLabelService()
        self.worker = None
        self.preview = {}        # image_path -> list[BoundingBox]
        self.preview_paths = []
        self.preview_index = -1
        self.saved = 0
        self.skipped = 0

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # --- model row ---
        row = QHBoxLayout()
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("Đường dẫn model .pt / .onnx")
        btn_model = QPushButton("Chọn model...")
        btn_model.clicked.connect(self._pick_model)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.99)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.5)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.05, 0.99)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)
        row.addWidget(self.model_edit)
        row.addWidget(btn_model)
        row.addWidget(QLabel("Conf:"))
        row.addWidget(self.conf_spin)
        row.addWidget(QLabel("IoU:"))
        row.addWidget(self.iou_spin)
        root.addLayout(row)

        # --- mode + folders row ---
        row2 = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Ảnh (thư mục)", "Video"])
        self.in_edit = QLineEdit()
        self.in_edit.setPlaceholderText("Input (thư mục ảnh hoặc file video)")
        btn_in = QPushButton("Input...")
        btn_in.clicked.connect(self._pick_input)
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Thư mục output")
        btn_out = QPushButton("Output...")
        btn_out.clicked.connect(self._pick_output)
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(1, 1000)
        self.frame_spin.setValue(10)
        row2.addWidget(self.mode_combo)
        row2.addWidget(self.in_edit)
        row2.addWidget(btn_in)
        row2.addWidget(self.out_edit)
        row2.addWidget(btn_out)
        row2.addWidget(QLabel("Mỗi N frame:"))
        row2.addWidget(self.frame_spin)
        root.addLayout(row2)

        # --- actions row ---
        row3 = QHBoxLayout()
        self.run_btn = QPushButton("Chạy (lưu thẳng)")
        self.run_btn.clicked.connect(self._run_batch)
        self.preview_btn = QPushButton("Preview mode")
        self.preview_btn.clicked.connect(self._run_preview)
        row3.addWidget(self.run_btn)
        row3.addWidget(self.preview_btn)
        row3.addStretch()
        root.addLayout(row3)

        self.progress = QProgressBar()
        root.addWidget(self.progress)

        # --- preview canvas + nav ---
        self.canvas = LabelCanvas()
        root.addWidget(self.canvas, stretch=1)
        nav = QHBoxLayout()
        self.prev_btn = QPushButton("← Trước")
        self.save_next_btn = QPushButton("Lưu && Tiếp")
        self.skip_btn = QPushButton("Bỏ qua")
        self.del_all_btn = QPushButton("Xóa hết box")
        self.counter = QLabel("Đã lưu: 0 | Bỏ qua: 0")
        self.prev_btn.clicked.connect(lambda: self._show_preview(self.preview_index - 1))
        self.save_next_btn.clicked.connect(self._save_and_next)
        self.skip_btn.clicked.connect(self._skip)
        self.del_all_btn.clicked.connect(self._delete_all)
        for wdg in (self.prev_btn, self.save_next_btn, self.skip_btn,
                    self.del_all_btn, self.counter):
            nav.addWidget(wdg)
        nav.addStretch()
        root.addLayout(nav)
        self._set_preview_enabled(False)
        self.statusBar()

    # ---------- pickers ----------

    def _pick_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn model", "", "Model (*.pt *.onnx)")
        if not path:
            return
        try:
            names = self.service.load_model(path)
        except Exception as e:
            QMessageBox.critical(self, "Lỗi model", str(e))
            return
        self.model_edit.setText(path)
        self.canvas.class_colors = {n: stable_color(n) for n in names}
        self.statusBar().showMessage(
            f"Model OK — {len(names)} classes: {', '.join(names[:8])}...", 5000)

    def _pick_input(self):
        if self.mode_combo.currentIndex() == 0:
            p = QFileDialog.getExistingDirectory(self, "Thư mục ảnh")
        else:
            p, _ = QFileDialog.getOpenFileName(
                self, "Chọn video", "", "Video (*.mp4 *.avi *.mkv *.mov)")
        if p:
            self.in_edit.setText(p)

    def _pick_output(self):
        p = QFileDialog.getExistingDirectory(self, "Thư mục output")
        if p:
            self.out_edit.setText(p)

    def _validated(self):
        if self.service.model is None:
            QMessageBox.warning(self, "Thiếu", "Chọn model trước.")
            return None
        inp = self.in_edit.text().strip()
        out = self.out_edit.text().strip()
        if not inp or not out:
            QMessageBox.warning(self, "Thiếu", "Chọn input và output.")
            return None
        return inp, out

    # ---------- batch / video ----------

    def _run_batch(self):
        v = self._validated()
        if not v:
            return
        inp, out = v
        if self.mode_combo.currentIndex() == 1:
            self.worker = _VideoWorker(self.service, inp, out,
                                       self.frame_spin.value(),
                                       self.conf_spin.value(),
                                       self.iou_spin.value())
            self.worker.progress.connect(self.statusBar().showMessage)
            self.worker.done.connect(
                lambda n: QMessageBox.information(self, "Xong", f"Đã xử lý {n} frame."))
            self.worker.failed.connect(
                lambda m: QMessageBox.critical(self, "Lỗi", m))
            self.worker.start()
            return
        images = scan_images(inp)
        os.makedirs(out, exist_ok=True)
        self.worker = _BatchWorker(self.service, images, self.conf_spin.value(),
                                   self.iou_spin.value(), out_dir=out)
        self._wire_progress(self.worker)
        self.worker.done.connect(
            lambda n: QMessageBox.information(self, "Xong", f"Đã gán nhãn {n} ảnh."))
        self.worker.start()

    def _wire_progress(self, worker):
        worker.progress.connect(
            lambda d, t, p: (self.progress.setMaximum(t), self.progress.setValue(d),
                             self.statusBar().showMessage(os.path.basename(p))))
        worker.failed.connect(lambda m: QMessageBox.critical(self, "Lỗi", m))

    # ---------- preview mode ----------

    def _run_preview(self):
        v = self._validated()
        if not v:
            return
        inp, out = v
        if self.mode_combo.currentIndex() == 1:
            QMessageBox.information(self, "Chỉ ảnh",
                                    "Preview mode chỉ dùng cho thư mục ảnh.")
            return
        self.preview = {}
        self.preview_paths = scan_images(inp)
        self.saved = 0
        self.skipped = 0
        self.worker = _BatchWorker(self.service, self.preview_paths,
                                   self.conf_spin.value(), self.iou_spin.value())
        self._wire_progress(self.worker)
        self.worker.detected.connect(lambda p, b: self.preview.__setitem__(p, b))
        self.worker.done.connect(self._preview_ready)
        self.worker.start()

    def _preview_ready(self, n):
        if not self.preview_paths:
            return
        self._set_preview_enabled(True)
        self.canvas.labeling_enabled = True
        self.canvas.current_class = (self.service.class_names or [""])[0]
        self._show_preview(0)

    def _set_preview_enabled(self, on):
        for w in (self.prev_btn, self.save_next_btn, self.skip_btn, self.del_all_btn):
            w.setEnabled(on)

    def _show_preview(self, index):
        if not (0 <= index < len(self.preview_paths)):
            return
        self._stash_preview_edits()
        self.preview_index = index
        path = self.preview_paths[index]
        self.canvas.set_image(path)
        self.canvas.boxes = self.preview.get(path, [])
        self.canvas.polygons = []
        self.canvas.viewport().update()
        self.statusBar().showMessage(
            f"[{index + 1}/{len(self.preview_paths)}] {os.path.basename(path)}")

    def _stash_preview_edits(self):
        if 0 <= self.preview_index < len(self.preview_paths):
            self.preview[self.preview_paths[self.preview_index]] = self.canvas.boxes

    def _save_and_next(self):
        self._stash_preview_edits()
        path = self.preview_paths[self.preview_index]
        out = self.out_edit.text().strip()
        os.makedirs(out, exist_ok=True)
        with Image.open(path) as im:
            w, h = im.size
        stem = os.path.splitext(os.path.basename(path))[0]
        save_yolo_txt(self.preview[path], os.path.join(out, stem + ".txt"),
                      w, h, self.service.class_names)
        self.saved += 1
        self._advance()

    def _skip(self):
        self.skipped += 1
        self._advance()

    def _advance(self):
        self.counter.setText(f"Đã lưu: {self.saved} | Bỏ qua: {self.skipped}")
        if self.preview_index < len(self.preview_paths) - 1:
            self._show_preview(self.preview_index + 1)
        else:
            QMessageBox.information(self, "Hết", "Đã duyệt hết ảnh.")

    def _delete_all(self):
        self.canvas.boxes = []
        self._stash_preview_edits()
        self.canvas.viewport().update()
```

If Task 10 used the `try/except ImportError` guard for `open_auto_label`, remove it now.

- [ ] **Step 2: Manual verify**

Prereq: any YOLO model, e.g. `yolo11n.pt` (ultralytics auto-downloads: `python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"`).
Run: `python main.py` → Auto Label:
- Load `yolo11n.pt` → status shows 80 classes.
- Images mode, folder with a few photos containing people/cars, "Chạy (lưu thẳng)" → output `.txt` per image, normalized 5-field lines.
- "Preview mode" → boxes appear on canvas; edit/move/delete a box, "Lưu && Tiếp" writes edited txt; "Bỏ qua" doesn't write; counters update.
- Video mode with an `.mp4`: frames in `<out>/frames`, labels in `<out>/labels`.

- [ ] **Step 3: Commit**

```bash
git add app/ui/auto_label_window.py && git commit -m "feat: auto-label window with batch, preview review, and video modes"
```

---

### Task 14: Trainer + train dialog

**Files:**
- Create: `app/training/trainer.py`, `app/ui/train_dialog.py`
- Test: `tests/test_trainer.py`

**Interfaces:**
- Produces:
  - `resolve_model_name(base: str, task: str) -> str` — `"yolo11n.pt" + "obb"` → `"yolo11n-obb.pt"`, `+ "segment"` → `"yolo11n-seg.pt"`, `+ "detect"` → unchanged; custom paths (containing a path separator or an existing file) returned unchanged.
  - `class TrainWorker(QThread)` — signals `log_line = Signal(str)`, `finished_ok = Signal(str)` (save_dir), `failed = Signal(str)`; method `request_stop()` (sets trainer.stop via callback at next epoch end).
  - `TrainDialog(main_window)` — form + live log; prefills data.yaml from `main_window.last_export_yaml`.

- [ ] **Step 1: Write the failing test**

`tests/test_trainer.py`:
```python
from app.training.trainer import resolve_model_name


def test_resolve_model_name_suffixes():
    assert resolve_model_name("yolo11n.pt", "detect") == "yolo11n.pt"
    assert resolve_model_name("yolo11n.pt", "obb") == "yolo11n-obb.pt"
    assert resolve_model_name("yolo11s.pt", "segment") == "yolo11s-seg.pt"


def test_resolve_model_name_custom_path_untouched():
    assert resolve_model_name(r"C:\models\best.pt", "obb") == r"C:\models\best.pt"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trainer.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement trainer**

`app/training/trainer.py`:
```python
import logging
import os

from PySide6.QtCore import QThread, Signal

_SUFFIX = {"obb": "-obb", "segment": "-seg"}


def resolve_model_name(base, task):
    if os.sep in base or "/" in base or os.path.isfile(base):
        return base
    suffix = _SUFFIX.get(task, "")
    if not suffix or suffix in base:
        return base
    stem, ext = os.path.splitext(base)
    return f"{stem}{suffix}{ext}"


class _SignalLogHandler(logging.Handler):
    def __init__(self, emit_fn):
        super().__init__()
        self.emit_fn = emit_fn

    def emit(self, record):
        self.emit_fn(self.format(record))


class TrainWorker(QThread):
    log_line = Signal(str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, task, model, data_yaml, epochs, imgsz, batch, device):
        super().__init__()
        self.task = task
        self.model_name = resolve_model_name(model, task)
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.device = device
        self._stop = False

    def request_stop(self):
        self._stop = True
        self.log_line.emit(">>> Sẽ dừng sau epoch hiện tại...")

    def run(self):
        handler = _SignalLogHandler(self.log_line.emit)
        logger = logging.getLogger("ultralytics")
        logger.addHandler(handler)
        try:
            from ultralytics import YOLO
            model = YOLO(self.model_name)

            def on_epoch_end(trainer):
                if self._stop:
                    trainer.stop = True

            model.add_callback("on_train_epoch_end", on_epoch_end)
            kwargs = {"data": self.data_yaml, "epochs": self.epochs,
                      "imgsz": self.imgsz, "batch": self.batch}
            if self.device != "auto":
                kwargs["device"] = self.device
            results = model.train(**kwargs)
            save_dir = str(getattr(results, "save_dir", "runs"))
            self.finished_ok.emit(save_dir)
        except Exception as e:
            self.failed.emit(str(e))
        finally:
            logger.removeHandler(handler)
```

`app/ui/train_dialog.py`:
```python
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QFormLayout,
                               QHBoxLayout, QLineEdit, QMessageBox,
                               QPlainTextEdit, QPushButton, QSpinBox,
                               QVBoxLayout)

from app.training.trainer import TrainWorker


class TrainDialog(QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.setWindowTitle("Train YOLO")
        self.resize(760, 560)
        self.worker = None

        form = QFormLayout()
        self.task_combo = QComboBox()
        self.task_combo.addItems(["detect", "obb", "segment"])
        form.addRow("Task:", self.task_combo)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(["yolo11n.pt", "yolo11s.pt", "yolo11m.pt",
                                   "yolo11l.pt"])
        form.addRow("Model nền:", self.model_combo)

        data_row = QHBoxLayout()
        self.data_edit = QLineEdit(main_window.last_export_yaml or "")
        browse = QPushButton("...")
        browse.clicked.connect(self._browse_yaml)
        data_row.addWidget(self.data_edit)
        data_row.addWidget(browse)
        form.addRow("data.yaml:", data_row)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(100)
        form.addRow("Epochs:", self.epochs_spin)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 4096)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        form.addRow("Image size:", self.imgsz_spin)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(16)
        form.addRow("Batch:", self.batch_spin)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "0"])
        form.addRow("Device:", self.device_combo)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Bắt đầu train")
        self.stop_btn = QPushButton("Dừng")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        layout.addWidget(self.log, stretch=1)

    def _browse_yaml(self):
        p, _ = QFileDialog.getOpenFileName(self, "Chọn data.yaml", "",
                                           "YAML (*.yaml *.yml)")
        if p:
            self.data_edit.setText(p)

    def _start(self):
        data = self.data_edit.text().strip()
        if not data:
            QMessageBox.warning(self, "Thiếu", "Chọn file data.yaml (Export trước).")
            return
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log.clear()
        self.worker = TrainWorker(
            self.task_combo.currentText(),
            self.model_combo.currentText().strip(),
            data, self.epochs_spin.value(), self.imgsz_spin.value(),
            self.batch_spin.value(), self.device_combo.currentText())
        self.worker.log_line.connect(self.log.appendPlainText)
        self.worker.finished_ok.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _stop(self):
        if self.worker:
            self.worker.request_stop()

    def _done(self, save_dir):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log.appendPlainText(f"\n=== XONG. Kết quả: {save_dir} ===")
        QMessageBox.information(self, "Train xong", f"Kết quả lưu tại:\n{save_dir}")

    def _failed(self, msg):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log.appendPlainText(f"\n=== LỖI: {msg} ===")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Đang train",
                                "Bấm Dừng và đợi train kết thúc trước khi đóng.")
            event.ignore()
            return
        super().closeEvent(event)
```

Remove the Task-10 ImportError guards for `open_train`/`open_export` if used.

- [ ] **Step 4: Run test, then manual verify**

Run: `pytest tests/test_trainer.py -v` — expected PASS.
Manual: label ~10 images, Export (detect), Train with `yolo11n.pt`, epochs=2, imgsz=320, device=cpu:
- Log streams live epochs; UI stays responsive.
- "Dừng" stops after the current epoch.
- Completion dialog shows the `runs/...` path with weights inside.

- [ ] **Step 5: Commit**

```bash
git add app/training/trainer.py app/ui/train_dialog.py tests/test_trainer.py && git commit -m "feat: integrated YOLO training with live log and stop"
```

---

### Task 15: README + full E2E pass

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write the full README**

`README.md`:
```markdown
# DataLabeling Python

Công cụ gán nhãn ảnh (box, box xoay, polygon) kèm train YOLO tích hợp.
Bản Python của app C# WPF ImageLableing.

## Cài đặt

    pip install -r requirements.txt
    python main.py

## Quy trình

1. **Mở thư mục ảnh** — nhãn cũ trong `labels/*.json` tự load lại.
2. **Gán nhãn**: chọn class, kéo chuột vẽ box; kéo nút tròn xanh để xoay box;
   chế độ Polygon: click từng đỉnh, click gần điểm đầu để đóng.
3. **Auto Label** (tùy chọn): load model `.pt`/`.onnx`, chạy batch hoặc preview
   duyệt từng ảnh; hỗ trợ tách frame video.
4. **Export**: chọn task detect / obb / segment, chia train/val, sinh `data.yaml`.
5. **Train**: chọn model nền + epochs, theo dõi log trực tiếp; kết quả trong `runs/`.

## Phím tắt

| Phím | Chức năng |
|---|---|
| `A` / `←` | Ảnh trước |
| `D` / `→` | Ảnh sau |
| `Ctrl+S` | Lưu nhãn ảnh hiện tại |
| `Ctrl+C` / `Ctrl+V` | Copy / paste ROI |
| `Delete` | Xóa shape đang chọn |
| `1`–`9` | Chọn nhanh class |
| `0` | Fit ảnh |
| `Esc` | Hủy polygon đang vẽ |
| Lăn chuột | Zoom tại con trỏ |
| Kéo chuột giữa | Pan |

## Định dạng nhãn

`labels/<tên ảnh>.json` — giữ nguyên cấu trúc của bản C#
(`BoundingBoxes[{Id, ClassName, X, Y, Width, Height, Angle}]`,
`Polygons[{Id, ClassName, Points}]`, tọa độ pixel, góc độ).

## Test

    pytest -v
```

- [ ] **Step 2: Full E2E manual pass**

Run: `pytest -v` — all tests PASS.
Run: `python main.py` and walk the whole flow once: open folder → label (box + rotated + polygon) → save → reopen (labels reload) → auto-label preview → export obb → train 2 epochs cpu → check `runs/`.

- [ ] **Step 3: Commit**

```bash
git add README.md && git commit -m "docs: usage guide with shortcuts and workflow"
```
