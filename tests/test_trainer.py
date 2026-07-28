import os

from app.training.trainer import harvest_run, resolve_model_name


def test_resolve_model_name_suffixes():
    assert resolve_model_name("yolo11n.pt", "detect") == "yolo11n.pt"
    assert resolve_model_name("yolo11n.pt", "obb") == "yolo11n-obb.pt"
    assert resolve_model_name("yolo11s.pt", "segment") == "yolo11s-seg.pt"
    assert resolve_model_name("yolo26n.pt", "detect") == "yolo26n.pt"
    assert resolve_model_name("yolo26n.pt", "obb") == "yolo26n-obb.pt"
    assert resolve_model_name("yolo26s.pt", "segment") == "yolo26s-seg.pt"


def test_resolve_model_name_custom_path_untouched():
    assert resolve_model_name(r"C:\models\best.pt", "obb") == r"C:\models\best.pt"


def test_harvest_run_copies_pt_and_onnx(tmp_path):
    run = tmp_path / "run"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"pt")
    (run / "weights" / "best.onnx").write_bytes(b"onnx")
    (run / "results.csv").write_text("junk")
    models = tmp_path / "models"
    models.mkdir()
    copied = harvest_run(str(run), str(models), "pill_detect_20260728-1500")
    assert [os.path.basename(p) for p in copied] == [
        "pill_detect_20260728-1500.pt", "pill_detect_20260728-1500.onnx"]
    assert all(os.path.isfile(p) for p in copied)


def test_harvest_run_pt_only(tmp_path):
    run = tmp_path / "run"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"pt")
    models = tmp_path / "models"
    models.mkdir()
    copied = harvest_run(str(run), str(models), "m")
    assert [os.path.basename(p) for p in copied] == ["m.pt"]
