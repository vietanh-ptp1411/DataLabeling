import math
import os

from app.models.bounding_box import BoundingBox
from app.services.auto_label_service import boxes_from_result, save_yolo_txt


class _T:  # minimal stand-in for a tensor: only .tolist() is used
    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_boxes_from_result_plain_detect():
    res = _NS(obb=None,
              boxes=_NS(xyxy=_T([[10.0, 20.0, 50.0, 80.0]]), cls=_T([1.0])))
    out = boxes_from_result(res, ["Car", "Person"])
    b = out[0]
    assert (b.x, b.y, b.width, b.height, b.angle) == (10, 20, 40, 60, 0.0)
    assert b.class_name == "Person"


def test_boxes_from_result_obb_converts_radians():
    res = _NS(obb=_NS(xywhr=_T([[50.0, 50.0, 40.0, 20.0, math.pi / 2]]),
                      cls=_T([0.0])),
              boxes=None)
    out = boxes_from_result(res, ["Car"])
    b = out[0]
    assert (b.x, b.y, b.width, b.height) == (30, 40, 40, 20)  # cx,cy → top-left
    assert math.isclose(b.angle, 90.0)


def test_boxes_from_result_unknown_class_id_falls_back_to_str():
    res = _NS(obb=None, boxes=_NS(xyxy=_T([[0, 0, 1, 1]]), cls=_T([7.0])))
    assert boxes_from_result(res, ["Car"])[0].class_name == "7"


def test_save_yolo_txt(tmp_path):
    txt = str(tmp_path / "img.txt")
    save_yolo_txt([BoundingBox("Car", 50, 50, 100, 50)], txt, 200, 100, ["Car"])
    assert open(txt).read().strip() == "0 0.500000 0.750000 0.500000 0.500000"


def test_save_yolo_txt_skips_unknown_classes(tmp_path):
    txt = str(tmp_path / "img.txt")
    save_yolo_txt([BoundingBox("Ghost", 0, 0, 10, 10)], txt, 100, 100, ["Car"])
    assert open(txt).read().strip() == ""
