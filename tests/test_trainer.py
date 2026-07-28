from app.training.trainer import resolve_model_name


def test_resolve_model_name_suffixes():
    assert resolve_model_name("yolo11n.pt", "detect") == "yolo11n.pt"
    assert resolve_model_name("yolo11n.pt", "obb") == "yolo11n-obb.pt"
    assert resolve_model_name("yolo11s.pt", "segment") == "yolo11s-seg.pt"


def test_resolve_model_name_custom_path_untouched():
    assert resolve_model_name(r"C:\models\best.pt", "obb") == r"C:\models\best.pt"
