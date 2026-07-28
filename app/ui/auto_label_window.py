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
