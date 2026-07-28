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


def export_onnx(pt_path, imgsz=640):
    """Convert a trained .pt checkpoint to ONNX; returns the .onnx path."""
    from ultralytics import YOLO
    return str(YOLO(pt_path).export(format="onnx", imgsz=imgsz))


class ConvertWorker(QThread):
    """Background .pt -> .onnx conversion."""
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, pt_path, imgsz=640):
        super().__init__()
        self.pt_path = pt_path
        self.imgsz = imgsz

    def run(self):
        try:
            self.done.emit(export_onnx(self.pt_path, self.imgsz))
        except Exception as e:
            self.failed.emit(str(e))


class TrainWorker(QThread):
    log_line = Signal(str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, task, model, data_yaml, epochs, imgsz, batch, device,
                 onnx_after=False):
        super().__init__()
        self.task = task
        self.model_name = resolve_model_name(model, task)
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.device = device
        self.onnx_after = onnx_after
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
                self.log_line.emit(
                    f">>> Epoch {trainer.epoch + 1}/{trainer.epochs} xong")
                if self._stop:
                    trainer.stop = True

            model.add_callback("on_train_epoch_end", on_epoch_end)
            kwargs = {"data": self.data_yaml, "epochs": self.epochs,
                      "imgsz": self.imgsz, "batch": self.batch}
            if self.device != "auto":
                kwargs["device"] = self.device
            results = model.train(**kwargs)
            save_dir = str(getattr(results, "save_dir", "runs"))
            if self.onnx_after:
                best = os.path.join(save_dir, "weights", "best.pt")
                if os.path.isfile(best):
                    self.log_line.emit(">>> Đang xuất ONNX...")
                    try:
                        onnx_path = export_onnx(best, self.imgsz)
                        self.log_line.emit(f">>> ONNX: {onnx_path}")
                    except Exception as e:
                        self.log_line.emit(f">>> Xuất ONNX lỗi: {e}")
            self.finished_ok.emit(save_dir)
        except Exception as e:
            self.failed.emit(str(e))
        finally:
            logger.removeHandler(handler)
