from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGridLayout,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPlainTextEdit, QPushButton, QSpinBox,
                               QVBoxLayout)

from app.training.trainer import ConvertWorker, TrainWorker


class TrainDialog(QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.setWindowTitle("DeepAI")
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
        self.model_combo.setToolTip(
            "Tên model nền (tự tải về), hoặc chọn file .pt đã train\n"
            "trong models/ để TRAIN TIẾP từ trọng số đó")
        model_row = QHBoxLayout()
        model_row.setSpacing(6)
        model_row.addWidget(self.model_combo, 1)
        browse_model = QPushButton("📂")
        browse_model.setFixedWidth(40)
        browse_model.setAutoDefault(False)
        browse_model.setToolTip("Chọn file .pt đã train để train tiếp")
        browse_model.clicked.connect(self._browse_model)
        model_row.addWidget(browse_model)
        form.addRow("Model nền:", model_row)

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

        # hyper-params, two rows
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(300)
        self.epochs_spin.setToolTip(
            "Số epoch tối đa — kết hợp Patience nên cứ để cao")
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 4096)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(4)
        self.batch_spin.setToolTip("Giảm xuống 2 nếu máy yếu / hết RAM")
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "0"])
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 1000)
        self.patience_spin.setValue(50)
        self.patience_spin.setToolTip(
            "Early Stopping: không cải thiện sau N epoch thì tự dừng.\n"
            "0 = tắt early stopping (train đủ số epoch).")
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(
            ["auto", "AdamW", "SGD", "Adam", "NAdam", "RAdam", "RMSProp"])
        self.optimizer_combo.setToolTip(
            "auto: ultralytics tự chọn optimizer + learning rate\n"
            "(khi auto thì lr0 bị bỏ qua)")
        self.lr0_spin = QDoubleSpinBox()
        self.lr0_spin.setDecimals(4)
        self.lr0_spin.setRange(0.0001, 0.1)
        self.lr0_spin.setSingleStep(0.0005)
        self.lr0_spin.setValue(0.001)
        self.lr0_spin.setToolTip(
            "Learning rate ban đầu — chỉ có tác dụng khi Optimizer khác auto")
        # params grid: dim label above each field, 4 equal columns
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(4)
        cells = [("Epochs", self.epochs_spin, 0, 0),
                 ("Image size", self.imgsz_spin, 0, 1),
                 ("Batch", self.batch_spin, 0, 2),
                 ("Device", self.device_combo, 0, 3),
                 ("Patience (early stop)", self.patience_spin, 2, 0),
                 ("Optimizer", self.optimizer_combo, 2, 1),
                 ("lr0", self.lr0_spin, 2, 2)]
        for text, widget, row, col in cells:
            lab = QLabel(text)
            lab.setProperty("dim", True)
            grid.addWidget(lab, row, col)
            grid.addWidget(widget, row + 1, col)
        for c in range(4):
            grid.setColumnStretch(c, 1)
        grid.setRowMinimumHeight(2, 26)   # air between the two param rows

        self.pretrained_check = QCheckBox("Pretrained")
        self.pretrained_check.setChecked(True)
        self.pretrained_check.setToolTip(
            "Bắt đầu từ trọng số đã train sẵn (khuyên dùng) thay vì từ đầu")
        self.onnx_check = QCheckBox("Xuất ONNX")
        self.onnx_check.setChecked(True)
        self.onnx_check.setToolTip(
            "Train xong tự tạo thêm bản .onnx cạnh file .pt")
        checks = QVBoxLayout()
        checks.setSpacing(6)
        checks.addStretch()
        checks.addWidget(self.pretrained_check)
        checks.addWidget(self.onnx_check)
        grid.addLayout(checks, 2, 3, 2, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)
        head_cfg = QLabel("CẤU HÌNH")
        head_cfg.setObjectName("sectionLabel")
        layout.addWidget(head_cfg)
        layout.addLayout(form)
        layout.addSpacing(4)
        head_par = QLabel("THAM SỐ TRAIN")
        head_par.setObjectName("sectionLabel")
        layout.addWidget(head_par)
        layout.addLayout(grid)
        layout.addSpacing(4)
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

    def _browse_model(self):
        from app.training.trainer import app_models_dir
        p, _ = QFileDialog.getOpenFileName(
            self, "Chọn model .pt để train tiếp", app_models_dir(),
            "PyTorch model (*.pt)")
        if p:
            self.model_combo.setCurrentText(p)

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
            onnx_after=self.onnx_check.isChecked(),
            patience=self.patience_spin.value(),
            optimizer=self.optimizer_combo.currentText(),
            lr0=self.lr0_spin.value(),
            pretrained=self.pretrained_check.isChecked())
        self.worker.log_line.connect(self.log.appendPlainText)
        self.worker.finished_ok.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _stop(self):
        if self.worker:
            self.worker.request_stop()

    def _done(self, model_paths):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log.appendPlainText(f"\n=== XONG. Model đã lưu ===\n{model_paths}")
        QMessageBox.information(self, "Train xong",
                                f"Model đã lưu:\n{model_paths}")

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
