import os

from PySide6.QtCore import QSize, QThread, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QProgressBar, QPushButton,
                               QSpinBox, QVBoxLayout)

from app.i18n import tr
from app.services.export_service import export_dataset
from app.ui import theme


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
        self.setWindowTitle("Export Dataset")
        self.resize(480, 340)

        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(10)
        self.task_combo = QComboBox()
        self.task_combo.addItems(["detect", "obb", "segment"])
        form.addRow(tr("Task:"), self.task_combo)

        out_row = QHBoxLayout()
        out_row.setSpacing(6)
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText(tr("Chọn thư mục xuất dataset…"))
        browse = QPushButton()
        browse.setIcon(theme.folder_icon())
        browse.setIconSize(QSize(18, 18))
        browse.setFixedWidth(40)
        browse.setAutoDefault(False)
        browse.clicked.connect(self._browse)
        out_row.addWidget(self.out_edit)
        out_row.addWidget(browse)
        form.addRow(tr("Thư mục xuất:"), out_row)

        self.train_spin = QSpinBox()
        self.train_spin.setRange(1, 99)
        self.train_spin.setValue(80)
        self.train_spin.setSuffix(" %")
        self.val_label = QLabel(tr("Val: {v}%").format(v=20))
        self.val_label.setProperty("dim", True)
        self.train_spin.valueChanged.connect(
            lambda v: self.val_label.setText(tr("Val: {v}%").format(v=100 - v)))
        ratio_row = QHBoxLayout()
        ratio_row.setSpacing(10)
        ratio_row.addWidget(self.train_spin)
        ratio_row.addWidget(self.val_label)
        ratio_row.addStretch()
        form.addRow(tr("Train:"), ratio_row)

        self.copy_check = QCheckBox(tr("Copy ảnh vào dataset"))
        self.copy_check.setChecked(True)
        self.yaml_check = QCheckBox(tr("Tạo data.yaml"))
        self.yaml_check.setChecked(True)
        form.addRow("", self.copy_check)
        form.addRow("", self.yaml_check)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)
        layout.addLayout(form)
        anns = self.main.current_annotations()
        labeled = sum(1 for a in anns if a.boxes or a.polygons)
        info = QLabel(tr("Ảnh có nhãn: {a} / {b}").format(
            a=labeled, b=len(anns)))
        info.setProperty("dim", True)
        layout.addWidget(info)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        layout.addWidget(self.progress)
        layout.addStretch()
        self.export_btn = QPushButton("📦  " + tr("Export dataset"))
        self.export_btn.setProperty("accent", True)
        self.export_btn.setMinimumHeight(36)
        self.export_btn.clicked.connect(self._start)
        layout.addWidget(self.export_btn)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, tr("Chọn thư mục xuất"))
        if d:
            self.out_edit.setText(d)

    def _start(self):
        out_dir = self.out_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, tr("Thiếu"), tr("Chọn thư mục xuất."))
            return
        if not self.main.classes:
            QMessageBox.warning(self, tr("Thiếu"), tr("Chưa có class nào."))
            return
        anns = self.main.current_annotations()
        if not any(a.boxes or a.polygons for a in anns):
            QMessageBox.warning(self, tr("Thiếu"),
                                tr("Chưa có ảnh nào được gán nhãn."))
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
            self, tr("Xong"),
            tr("Train: {a} ảnh, Val: {b} ảnh\ndata.yaml: {c}").format(
                a=result["train"], b=result["val"],
                c=result["yaml"] or tr("(không tạo)")))

    def _failed(self, msg):
        self.export_btn.setEnabled(True)
        QMessageBox.critical(self, tr("Lỗi export"), msg)
