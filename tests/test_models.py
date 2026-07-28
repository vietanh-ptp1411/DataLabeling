from app.models.bounding_box import BoundingBox
from app.models.image_annotation import ImageAnnotation
from app.models.label_class import LabelClass, stable_color
from app.models.polygon import PolygonAnnotation


def test_stable_color_is_deterministic_hex():
    c1 = stable_color("Car")
    c2 = stable_color("Car")
    assert c1 == c2
    assert c1.startswith("#") and len(c1) == 7


def test_label_class_auto_color_and_dict():
    lc = LabelClass("Person")
    assert lc.color == stable_color("Person")
    d = lc.to_dict()
    assert d == {"Name": "Person", "Color": lc.color}
    assert LabelClass.from_dict(d) == lc


def test_bounding_box_center_and_csharp_dict_roundtrip():
    b = BoundingBox("Car", 10, 20, 100, 50, angle=30.0)
    assert b.center == (60.0, 45.0)
    d = b.to_dict()
    assert set(d) == {"Id", "ClassName", "X", "Y", "Width", "Height", "Angle"}
    b2 = BoundingBox.from_dict(d)
    assert (b2.x, b2.y, b2.width, b2.height, b2.angle, b2.class_name, b2.id) == (
        10, 20, 100, 50, 30.0, "Car", b.id)


def test_bounding_box_from_dict_defaults_angle_zero():
    b = BoundingBox.from_dict(
        {"Id": "i1", "ClassName": "Car", "X": 1, "Y": 2, "Width": 3, "Height": 4})
    assert b.angle == 0.0


def test_polygon_roundtrip_ignores_extra_keys():
    d = {"Id": "p1", "ClassName": "Person", "Tag": "legacy-guid",
         "Points": [{"X": 1, "Y": 2}, {"X": 3, "Y": 4}, {"X": 5, "Y": 6}]}
    p = PolygonAnnotation.from_dict(d)
    assert p.points == [(1, 2), (3, 4), (5, 6)]
    assert p.to_dict()["Points"][0] == {"X": 1, "Y": 2}


def test_image_annotation_roundtrip_and_filename():
    ann = ImageAnnotation(
        image_path=r"C:\data\img1.jpg",
        boxes=[BoundingBox("Car", 0, 0, 10, 10)],
        polygons=[PolygonAnnotation("Person", [(0, 0), (1, 0), (1, 1)])])
    d = ann.to_dict()
    assert d["ImageFileName"] == "img1.jpg"
    ann2 = ImageAnnotation.from_dict(d)
    assert len(ann2.boxes) == 1 and len(ann2.polygons) == 1
    assert ann2.boxes[0].class_name == "Car"
