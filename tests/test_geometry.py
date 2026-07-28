import math

from app.models.bounding_box import BoundingBox
from app.services import geometry as geo


def test_rotate_point_90_clockwise_screen_coords():
    # y grows downward: rotating the "up" vector (0,-1) by +90° gives "right" (1,0)
    x, y = geo.rotate_point(0, -1, 0, 0, 90)
    assert math.isclose(x, 1, abs_tol=1e-9) and math.isclose(y, 0, abs_tol=1e-9)


def test_box_corners_unrotated():
    b = BoundingBox("c", 100, 100, 50, 20)
    assert geo.box_corners(b) == [(100, 100), (150, 100), (150, 120), (100, 120)]


def test_box_corners_rotated_90():
    b = BoundingBox("c", 0, 0, 40, 20, angle=90)
    corners = geo.box_corners(b)
    expected = [(30, -10), (30, 30), (10, 30), (10, -10)]
    for (cx, cy), (ex, ey) in zip(corners, expected):
        assert math.isclose(cx, ex, abs_tol=1e-9)
        assert math.isclose(cy, ey, abs_tol=1e-9)


def test_point_in_box_respects_rotation():
    b = BoundingBox("c", 0, 0, 40, 20, angle=90)
    assert geo.point_in_box(b, 25, 25)          # inside only when rotated
    assert not geo.point_in_box(BoundingBox("c", 0, 0, 40, 20), 25, 25)


def test_point_in_polygon():
    tri = [(0, 0), (10, 0), (0, 10)]
    assert geo.point_in_polygon(tri, 2, 2)
    assert not geo.point_in_polygon(tri, 8, 8)


def test_aabb_of():
    assert geo.aabb_of([(30, -10), (30, 30), (10, 30), (10, -10)]) == (10, -10, 20, 40)


def test_angle_from_center_zero_is_up_clockwise():
    assert math.isclose(geo.angle_from_center(0, 0, 0, -5), 0.0, abs_tol=1e-9)
    assert math.isclose(geo.angle_from_center(0, 0, 5, 0), 90.0, abs_tol=1e-9)
    assert math.isclose(geo.angle_from_center(0, 0, 0, 5), 180.0, abs_tol=1e-9)


def test_clamp01():
    assert geo.clamp01(-0.2) == 0.0 and geo.clamp01(1.7) == 1.0 and geo.clamp01(0.5) == 0.5
