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
