from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPlainTextEdit, QPushButton,
                               QSpinBox, QVBoxLayout)

from app.training.trainer import ConvertWorker, TrainWorker


class TrainDialog(QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.setWindowTitle("Train YOLO")
        self.resize(760, 560)
        self.worker = None
        self.converter = None

        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(10)
        self.task_combo = QComboBox()
        self.task_combo.addItems(["detect", "obb", "segment"])
        form.addRow("Task:", self.task_combo)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(["yolo26n.pt", "yolo26s.pt", "yolo26m.pt",
                                   "yolo26l.pt", "yolo11n.pt", "yolo11s.pt",
                                   "yolo11m.pt", "yolo11l.pt"])
        form.addRow("Model nền:", self.model_combo)

        data_row = QHBoxLayout()
        data_row.setSpacing(6)
        self.data_edit = QLineEdit(main_window.last_export_yaml or "")
        self.data_edit.setPlaceholderText("Đường dẫn data.yaml (Export trước)…")
        browse = QPushButton("📂")
        browse.setFixedWidth(40)
        browse.setAutoDefault(False)
        browse.clicked.connect(self._browse_yaml)
        data_row.addWidget(self.data_edit)
        data_row.addWidget(browse)
        form.addRow("data.yaml:", data_row)

        # hyper-params on one row
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(100)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 4096)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(16)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "0"])
        params = QHBoxLayout()
        params.setSpacing(8)
        for label, widget in (("Epochs", self.epochs_spin),
                              ("Image size", self.imgsz_spin),
                              ("Batch", self.batch_spin),
                              ("Device", self.device_combo)):
            lab = QLabel(label)
            lab.setProperty("dim", True)
            params.addWidget(lab)
            params.addWidget(widget, stretch=1)
        form.addRow("Tham số:", params)

        self.onnx_check = QCheckBox("Xuất ONNX sau khi train (best.onnx)")
        self.onnx_check.setChecked(True)
        form.addRow("", self.onnx_check)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)
        layout.addLayout(form)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.start_btn = QPushButton("🚀  Bắt đầu train")
        self.start_btn.setProperty("accent", True)
        self.start_btn.setMinimumHeight(34)
        self.start_btn.setAutoDefault(False)
        self.stop_btn = QPushButton("Dừng")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setAutoDefault(False)
        self.convert_btn = QPushButton("Convert .pt → ONNX…")
        self.convert_btn.setAutoDefault(False)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        self.convert_btn.clicked.connect(self._convert_pt)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.convert_btn)
        layout.addLayout(btn_row)
        log_head = QLabel("LOG TRAIN")
        log_head.setObjectName("sectionLabel")
        layout.addWidget(log_head)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setPlaceholderText("Log train sẽ hiện ở đây…")
        layout.addWidget(self.log, stretch=1)

    def _browse_yaml(self):
        p, _ = QFileDialog.getOpenFileName(self, "Chọn data.yaml", "",
                                           "YAML (*.yaml *.yml)")
        if p:
            self.data_edit.setText(p)

    # ---------- .pt -> ONNX ----------

    def _convert_pt(self):
        p, _ = QFileDialog.getOpenFileName(self, "Chọn model .pt", "",
                                           "PyTorch model (*.pt)")
        if not p:
            return
        self.convert_btn.setEnabled(False)
        self.log.appendPlainText(f">>> Đang convert sang ONNX: {p}")
        self.converter = ConvertWorker(p, self.imgsz_spin.value())
        self.converter.done.connect(self._convert_done)
        self.converter.failed.connect(self._convert_failed)
        self.converter.start()

    def _convert_done(self, onnx_path):
        self.convert_btn.setEnabled(True)
        self.log.appendPlainText(f">>> ONNX: {onnx_path}")
        QMessageBox.information(self, "Convert xong",
                                f"Đã tạo:\n{onnx_path}")

    def _convert_failed(self, msg):
        self.convert_btn.setEnabled(True)
        self.log.appendPlainText(f">>> Convert lỗi: {msg}")
        QMessageBox.critical(self, "Lỗi convert", msg)

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
            self.batch_spin.value(), self.device_combo.currentText(),
            onnx_after=self.onnx_check.isChecked())
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
        if self.converter and self.converter.isRunning():
            QMessageBox.warning(self, "Đang convert",
                                "Đợi convert ONNX xong trước khi đóng.")
            event.ignore()
            return
        super().closeEvent(event)
