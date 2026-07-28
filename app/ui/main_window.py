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
        try:
            from app.ui.train_dialog import TrainDialog
        except ImportError:
            QMessageBox.information(self, "Chưa có", "Tính năng đang được xây dựng")
            return
        TrainDialog(self).exec()
