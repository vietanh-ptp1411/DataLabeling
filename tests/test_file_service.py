import json
import os

from app.models.bounding_box import BoundingBox
from app.models.image_annotation import ImageAnnotation
from app.models.label_class import LabelClass
from app.services import file_service as fs


def _touch(p):
    open(p, "wb").close()


def test_scan_images_filters_and_sorts(tmp_path):
    for name in ["b.jpg", "a.PNG", "c.txt", "d.gif"]:
        _touch(tmp_path / name)
    result = fs.scan_images(str(tmp_path))
    assert [os.path.basename(p) for p in result] == ["a.PNG", "b.jpg", "d.gif"]


def test_save_and_load_annotation_roundtrip(tmp_path):
    img = str(tmp_path / "img1.jpg")
    _touch(img)
    ann = ImageAnnotation(img, boxes=[BoundingBox("Car", 1, 2, 3, 4, angle=15)])
    path = fs.save_annotation(str(tmp_path), ann)
    assert path == os.path.join(str(tmp_path), "labels", "img1.json")
    loaded = fs.load_annotation(str(tmp_path), img)
    assert loaded.boxes[0].angle == 15
    assert loaded.boxes[0].class_name == "Car"


def test_load_annotation_missing_returns_none(tmp_path):
    assert fs.load_annotation(str(tmp_path), str(tmp_path / "nope.jpg")) is None


def test_load_annotation_corrupt_returns_none(tmp_path):
    os.makedirs(tmp_path / "labels")
    (tmp_path / "labels" / "img1.json").write_text("{not json", encoding="utf-8")
    assert fs.load_annotation(str(tmp_path), str(tmp_path / "img1.jpg")) is None


def test_save_all_annotations(tmp_path):
    anns = [ImageAnnotation(str(tmp_path / "a.jpg"), boxes=[BoundingBox("Car", 0, 0, 5, 5)])]
    path = fs.save_all_annotations(str(tmp_path), anns)
    data = json.loads(open(path, encoding="utf-8").read())
    assert "ExportDate" in data and len(data["Images"]) == 1


def test_classes_roundtrip(tmp_path):
    fs.save_classes(str(tmp_path), [LabelClass("Car"), LabelClass("Person")])
    loaded = fs.load_classes(str(tmp_path))
    assert [c.name for c in loaded] == ["Car", "Person"]


def test_load_classes_missing_returns_empty(tmp_path):
    assert fs.load_classes(str(tmp_path)) == []
