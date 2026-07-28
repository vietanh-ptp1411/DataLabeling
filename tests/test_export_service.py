import os

import yaml
from PIL import Image

from app.models.bounding_box import BoundingBox
from app.models.image_annotation import ImageAnnotation
from app.models.polygon import PolygonAnnotation
from app.services import export_service as ex


def test_detect_line_unrotated():
    b = BoundingBox("c", 50, 50, 100, 50)
    assert ex.detect_line(b, 0, 200, 100) == "0 0.500000 0.750000 0.500000 0.500000"


def test_detect_line_rotated_uses_aabb():
    b = BoundingBox("c", 0, 0, 40, 20, angle=90)  # AABB: (10,-10,20,40)
    line = ex.detect_line(b, 1, 100, 100)
    parts = line.split()
    assert parts[0] == "1"
    # cy = (-10 + 40/2)/100 = 0.10; h clamped inside [0,1]: 40/100
    assert parts[2] == "0.100000" and parts[4] == "0.400000"


def test_obb_line_corners_normalized():
    b = BoundingBox("c", 100, 100, 50, 20)
    line = ex.obb_line(b, 2, 200, 200)
    assert line == ("2 0.500000 0.500000 0.750000 0.500000 "
                    "0.750000 0.600000 0.500000 0.600000")


def test_obb_line_clamps_out_of_bounds():
    b = BoundingBox("c", 0, 0, 40, 20, angle=90)  # corner y=-10 → clamp 0
    values = [float(v) for v in ex.obb_line(b, 0, 100, 100).split()[1:]]
    assert all(0.0 <= v <= 1.0 for v in values)


def test_seg_line():
    p = PolygonAnnotation("c", [(0, 0), (50, 0), (50, 100)])
    assert ex.seg_line(p, 3, 100, 100) == "3 0.000000 0.000000 0.500000 0.000000 0.500000 1.000000"


def test_split_names_deterministic_and_complete():
    names = [f"img{i}" for i in range(10)]
    t1, v1 = ex.split_names(names, 0.8, seed=42)
    t2, v2 = ex.split_names(names, 0.8, seed=42)
    assert t1 == t2 and v1 == v2
    assert len(t1) == 8 and len(v1) == 2
    assert sorted(t1 + v1) == sorted(names)


def _make_dataset(tmp_path, n=5):
    anns = []
    for i in range(n):
        img = str(tmp_path / f"img{i}.jpg")
        Image.new("RGB", (100, 80)).save(img)
        anns.append(ImageAnnotation(
            img,
            boxes=[BoundingBox("Car", 10, 10, 30, 20)],
            polygons=[PolygonAnnotation("Car", [(0, 0), (10, 0), (10, 10)])]))
    return anns


def test_export_dataset_detect(tmp_path):
    anns = _make_dataset(tmp_path)
    out = str(tmp_path / "out")
    result = ex.export_dataset(anns, ["Car"], out, "detect", train_ratio=0.8)
    assert result["train"] == 4 and result["val"] == 1
    assert len(os.listdir(os.path.join(out, "train", "images"))) == 4
    assert len(os.listdir(os.path.join(out, "train", "labels"))) == 4
    data = yaml.safe_load(open(result["yaml"], encoding="utf-8"))
    assert data["nc"] == 1 and data["names"] == ["Car"]
    assert data["train"] == "train/images" and data["val"] == "val/images"


def test_export_dataset_segment_skips_images_without_polygons(tmp_path):
    img = str(tmp_path / "only_box.jpg")
    Image.new("RGB", (50, 50)).save(img)
    anns = [ImageAnnotation(img, boxes=[BoundingBox("Car", 0, 0, 5, 5)])]
    result = ex.export_dataset(anns, ["Car"], str(tmp_path / "o"), "segment")
    assert result["train"] == 0 and result["val"] == 0
