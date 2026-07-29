"""Tiny dictionary-based i18n. Vietnamese strings are the keys; English
translations live in EN. tr() returns the key itself for "vi" (or when a
translation is missing), so untranslated strings degrade gracefully.

Language is persisted via QSettings("MVA", "DataLabeling")."""
from PySide6.QtCore import QSettings

_LANGS = ("vi", "en")
_lang = None


def _settings():
    return QSettings("MVA", "DataLabeling")


def get_lang():
    global _lang
    if _lang is None:
        code = _settings().value("language", "vi")
        _lang = code if code in _LANGS else "vi"
    return _lang


def set_lang(code):
    global _lang
    if code in _LANGS:
        _lang = code
        _settings().setValue("language", code)


def tr(key):
    if get_lang() == "en":
        return EN.get(key, key)
    return key


EN = {
    # ---- main window / toolbar ----
    "Mở thư mục": "Open folder",
    "Lưu": "Save",
    "Lưu tất cả": "Save all",
    "Chế độ": "Mode",
    "Công cụ": "Tool",
    "Pointer (Vẽ ROI)": "Pointer (Draw ROI)",
    "Touch (Kéo ảnh)": "Touch (Pan image)",
    "Quản lý…": "Manage…",
    "Fit ảnh": "Fit image",
    "DANH SÁCH ẢNH": "IMAGE LIST",
    "0 ảnh": "0 images",
    "{a} / {b} có nhãn": "{a} / {b} labeled",
    "Ảnh {a} / {b}": "Image {a} / {b}",
    "Chọn thư mục ảnh": "Select image folder",
    "Trống": "Empty",
    "Thư mục không có ảnh.": "The folder contains no images.",
    "Đã lưu nhãn ảnh hiện tại": "Saved labels for the current image",
    "Đã lưu tất cả nhãn": "Saved all labels",
    "Chưa có ảnh\nBấm “Mở thư mục” để bắt đầu gán nhãn":
        "No image yet\nClick “Open folder” to start labeling",
    # ---- canvas ----
    "Chưa có ảnh": "No image yet",
    "Đã copy ROI": "ROI copied",
    "Đã paste ROI": "ROI pasted",
    "Chọn class trước khi vẽ": "Select a class before drawing",
    # ---- manage classes ----
    "Quản lý Classes": "Manage Classes",
    "Thêm...": "Add...",
    "Xóa": "Delete",
    "Đóng": "Close",
    "Thêm class": "Add class",
    "Tên class:": "Class name:",
    "Trùng tên": "Duplicate name",
    "Class '{name}' đã tồn tại.": "Class '{name}' already exists.",
    "Xóa class '{name}'?": "Delete class '{name}'?",
    "Xác nhận": "Confirm",
    "\nClass đang được dùng bởi {n} nhãn — xóa sẽ gỡ khỏi TẤT CẢ ảnh.":
        "\nThis class is used by {n} labels — deleting removes it from ALL images.",
    # ---- export dialog ----
    "Export YOLO Dataset": "Export YOLO Dataset",
    "Task:": "Task:",
    "Thư mục xuất:": "Output folder:",
    "Chọn thư mục xuất dataset…": "Select dataset output folder…",
    "Train:": "Train:",
    "Val: {v}%": "Val: {v}%",
    "Copy ảnh vào dataset": "Copy images into dataset",
    "Tạo data.yaml": "Create data.yaml",
    "Ảnh có nhãn: {a} / {b}": "Labeled images: {a} / {b}",
    "Export dataset": "Export dataset",
    "Chọn thư mục xuất": "Select output folder",
    "Thiếu": "Missing",
    "Chọn thư mục xuất.": "Select an output folder.",
    "Chưa có class nào.": "No classes defined yet.",
    "Chưa có ảnh nào được gán nhãn.": "No images have been labeled yet.",
    "Xong": "Done",
    "Train: {a} ảnh, Val: {b} ảnh\ndata.yaml: {c}":
        "Train: {a} images, Val: {b} images\ndata.yaml: {c}",
    "(không tạo)": "(not created)",
    "Lỗi export": "Export error",
    # ---- train dialog ----
    "CẤU HÌNH": "CONFIGURATION",
    "THAM SỐ TRAIN": "TRAINING PARAMETERS",
    "LOG TRAIN": "TRAINING LOG",
    "Model nền:": "Base model:",
    "Để trống = mặc định (chất lượng cao nhất)":
        "Leave empty = default (highest quality)",
    "Để trống để dùng model nền mặc định, hoặc chọn file .pt\nđã train trong models/ để TRAIN TIẾP từ trọng số đó":
        "Leave empty for the default base model, or pick a trained\n.pt from models/ to CONTINUE training from those weights",
    "Chọn file .pt đã train để train tiếp":
        "Pick a trained .pt to continue training",
    "Chọn model .pt để train tiếp": "Select a .pt model to continue training",
    "Đường dẫn data.yaml (Export trước)…":
        "Path to data.yaml (Export first)…",
    "Chọn data.yaml": "Select data.yaml",
    "Số epoch tối đa — kết hợp Patience nên cứ để cao":
        "Maximum epochs — with Patience set, keep this high",
    "Giảm xuống 2 nếu máy yếu / hết RAM":
        "Lower to 2 on weak machines / out of RAM",
    "Patience (early stop)": "Patience (early stop)",
    "Early Stopping: không cải thiện sau N epoch thì tự dừng.\n0 = tắt early stopping (train đủ số epoch).":
        "Early Stopping: stops when no improvement for N epochs.\n0 = disabled (train the full epoch count).",
    "auto: ultralytics tự chọn optimizer + learning rate\n(khi auto thì lr0 bị bỏ qua)":
        "auto: optimizer + learning rate chosen automatically\n(lr0 is ignored while auto)",
    "Learning rate ban đầu — chỉ có tác dụng khi Optimizer khác auto":
        "Initial learning rate — only used when Optimizer is not auto",
    "Bắt đầu từ trọng số đã train sẵn (khuyên dùng) thay vì từ đầu":
        "Start from pretrained weights (recommended) instead of scratch",
    "Bắt đầu train": "Start training",
    "Dừng": "Stop",
    "Log train sẽ hiện ở đây…": "Training log will appear here…",
    "Chọn file data.yaml (Export trước).":
        "Select a data.yaml file (run Export first).",
    "Train xong": "Training finished",
    "Model đã lưu:\n{p}": "Model saved:\n{p}",
    "\n=== XONG. Model đã lưu ===\n{p}": "\n=== DONE. Model saved ===\n{p}",
    "\n=== LỖI: {m} ===": "\n=== ERROR: {m} ===",
    "Đang train": "Training in progress",
    "Bấm Dừng và đợi train kết thúc trước khi đóng.":
        "Press Stop and wait for training to finish before closing.",
    # ---- trainer log lines ----
    ">>> Đang chuẩn bị model và dữ liệu…": ">>> Preparing model and data…",
    ">>> Bắt đầu train ({n} epochs)…": ">>> Training started ({n} epochs)…",
    ">>> Đang xuất model…": ">>> Exporting model…",
    ">>> Xuất ONNX lỗi: {e}": ">>> ONNX export failed: {e}",
    ">>> Model: {p}": ">>> Model: {p}",
    ">>> Sẽ dừng sau epoch hiện tại...": ">>> Stopping after the current epoch...",
    "Train xong nhưng không thấy model kết quả.":
        "Training finished but no result model was found.",
    # ---- auto label window ----
    "Auto Label": "Auto Label",
    "MODEL": "MODEL",
    "DỮ LIỆU": "DATA",
    "Đường dẫn model .pt / .onnx": "Path to .pt / .onnx model",
    "Chọn model…": "Select model…",
    "Chọn model": "Select model",
    "Lỗi model": "Model error",
    "Model OK — {n} classes: {names}...": "Model OK — {n} classes: {names}...",
    "Ảnh (thư mục)": "Images (folder)",
    "Video": "Video",
    "Thư mục ảnh input": "Input image folder",
    "File video (.mp4 / .avi / .mkv / .mov)": "Video file (.mp4 / .avi / .mkv / .mov)",
    "Input (thư mục ảnh hoặc file video)": "Input (image folder or video file)",
    "Input…": "Input…",
    "Thư mục output": "Output folder",
    "Output…": "Output…",
    "Mỗi N frame": "Every N frames",
    "Chạy (lưu thẳng)": "Run (save directly)",
    "Preview từng ảnh": "Preview each image",
    "Thư mục ảnh": "Image folder",
    "Chọn video": "Select video",
    "← Trước": "← Previous",
    "Lưu && Tiếp →": "Save && Next →",
    "Bỏ qua": "Skip",
    "Xóa hết box": "Clear all boxes",
    "Đã lưu: {a} | Bỏ qua: {b}": "Saved: {a} | Skipped: {b}",
    "Chưa có ảnh preview\nChọn model + input rồi bấm “Preview từng ảnh”":
        "No preview yet\nPick a model + input, then click “Preview each image”",
    "Chọn model trước.": "Select a model first.",
    "Chọn input và output.": "Select input and output.",
    "Đã xử lý {n} frame.": "Processed {n} frames.",
    "Đã gán nhãn {n} ảnh.": "Labeled {n} images.",
    "Chỉ ảnh": "Images only",
    "Preview mode chỉ dùng cho thư mục ảnh.":
        "Preview mode only works with image folders.",
    "Hết": "Finished",
    "Đã duyệt hết ảnh.": "All images reviewed.",
    "Đang tách frame...": "Extracting frames...",
    "Frame {a}/{b}": "Frame {a}/{b}",
    "Lỗi": "Error",
    # ---- language picker ----
    "Ngôn ngữ": "Language",
}
