# DataLabeling_Python — Design Spec

**Date:** 2026-07-28
**Goal:** Port the C# WPF image labeling app (`E:\DataLabeling\ImageLableing`) to Python at `E:\DataLabeling_Python`, adding an integrated YOLO training workflow so the user can label → export → train in one app.

## Technology

- **GUI:** PySide6, canvas built on `QGraphicsScene`/`QGraphicsView`.
- **ML:** `ultralytics` (YOLOv8/11) for auto-labeling and training — supports detect, OBB, and segmentation, and loads both `.pt` and `.onnx` models.
- **Video/image:** `opencv-python`; **config/data.yaml:** `pyyaml`.
- **Python:** 3.10+.

## Project structure

```
E:\DataLabeling_Python\
├── main.py                      # entry point
├── requirements.txt
├── README.md
└── app/
    ├── models/                  # dataclasses
    │   ├── bounding_box.py      # id, class_name, x, y, w, h, angle (deg)
    │   ├── polygon.py           # id, class_name, points [(x,y)...]
    │   ├── image_annotation.py  # image_path + boxes + polygons
    │   └── label_class.py       # name, color
    ├── services/
    │   ├── file_service.py      # JSON save/load, classes.json, folder scan
    │   ├── geometry.py          # rotation math, hit-testing, coord transforms
    │   ├── auto_label_service.py# ultralytics inference wrapper
    │   ├── video_service.py     # frame extraction (OpenCV)
    │   └── export_service.py    # YOLO detect/OBB/seg export + train/val split
    ├── training/
    │   └── trainer.py           # ultralytics train wrapper, background thread
    └── ui/
        ├── main_window.py
        ├── label_canvas.py      # drawing/editing boxes, rotated boxes, polygons
        ├── auto_label_window.py
        ├── export_dialog.py
        ├── manage_classes_dialog.py
        └── train_dialog.py
```

## Features

### Manual labeling (parity with C# app)

- Draw axis-aligned rectangles (click-drag, min 5×5 px); class = currently selected class.
- Rotated boxes: every box has `angle` (degrees); a rotation handle (↻) above the box rotates it around its center.
- Polygons (segmentation mode): click to add vertices, min 3 points, close by clicking near the first vertex.
- Select / move / resize (8-direction handles; for rotated boxes, hit-test in box-local space).
- Delete selected shape; class badge with per-class color drawn above each shape.
- Zoom with mouse wheel centered on cursor (clamp 0.1×–10×), pan (pointer/touch tool toggle + middle-drag), Fit Image.
- Image folder loading (`.jpg .jpeg .png .bmp .gif`, sorted by name), image list panel, Prev/Next, "X of Y" counter, mouse-coordinate readout.
- Manage classes dialog: add/delete classes; deleting a class warns with usage count and cascades removal across all images.

### Keyboard shortcuts

- `Ctrl+C` / `Ctrl+V`: copy/paste selected box (paste offset +20/+20 px, wrap to 0 if out of bounds).
- `Delete`: delete selected shape (polygon priority, then box).
- **New:** `A`/`D` and `←`/`→` — previous/next image; `Ctrl+S` — save current labels; `1`–`9` — quick-select class by index; `0` — fit image.

### Annotation storage

- Same JSON format as the C# app for compatibility:
  - Per-image: `<project>/labels/<imageName>.json` with `{ImagePath, ImageFileName, BoundingBoxes:[{Id, ClassName, X, Y, Width, Height, Angle}], Polygons:[{Id, ClassName, Points:[{X,Y}]}]}` — absolute pixel coords, angle in degrees.
  - Save All: `<project>/annotations/annotations.json`.
- **Fixes over C#:**
  - Opening a folder auto-loads existing per-image JSON labels.
  - Class list persisted to `<project>/classes.json` (name + color) and auto-loaded.
  - Class colors are stable: derived from `md5(class_name)` into a 16-color palette (C# used process-randomized `GetHashCode`).
  - No hardcoded 640×480 export path (that C# legacy exporter is dropped entirely).

### Export (YOLO for ultralytics)

- Dialog with task choice: **Detect** (boxes → `class cx cy w h`, rotated boxes exported via their axis-aligned bounding rect), **OBB** (`class x1 y1 x2 y2 x3 y3 x4 y4`, corners from center+angle, normalized, clamped [0,1]), **Segmentation** (`class x1 y1 x2 y2 ...`).
- Train/val split: only images with ≥1 annotation; shuffle with seed 42; ratio spinners default 80/20, must sum to 100.
- Output: `train/images`, `train/labels`, `val/images`, `val/labels` + `data.yaml` (path, train, val, nc, names — 0-based ordinal class ids). "Copy images" option.

### Auto-label window

- Load `.pt` or `.onnx` model via ultralytics; optional `data.yaml` for class names (else model's own names).
- Confidence/IoU thresholds adjustable in UI (defaults 0.5 / 0.45) — not hardcoded like C#.
- **Image mode:** input+output folder; "Save All" batch mode, or **Preview mode**: review each image (prev/next, Save & Next, Skip, delete box, delete all, change class, draw/move/resize, zoom/pan, saved/skipped counters). Saves normalized YOLO txt.
- **Video mode:** extract every Nth frame (default 10) to `<out>/frames/`, run detection per frame, write `<out>/labels/`, live preview with drawn boxes.

### Training (new)

- Train dialog: pick task (detect/obb/segment), base model (yolo11n/s/m/l or custom path), dataset `data.yaml` (pre-filled from last export), epochs, imgsz, batch, device (auto/cpu/0).
- Runs `ultralytics` training in a background thread (QThread); live log streamed into the dialog; results in `runs/`. Stop button to interrupt.

## Error handling

- Missing/corrupt JSON label files: skip with a warning in status bar, never crash.
- Model load failures (auto-label/train): message box with the exception text.
- Training errors surface in the log panel; the app stays responsive (all inference/training off the UI thread).

## Testing

- `pytest` unit tests for `geometry.py` (rotation, corner computation, hit-testing, coord transforms) and `export_service.py` (YOLO line formats, normalization/clamping, split determinism with seed 42).
- GUI verified manually.

## Out of scope (v1)

- Undo/redo (also absent in the C# app).
- COCO/VOC export.
- The unused C# `BoundingBoxCanvas` control and empty Project menu stubs.
