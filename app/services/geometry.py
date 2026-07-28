"""Pure geometry helpers. Screen coordinate convention: y grows downward,
positive angles rotate clockwise, angle 0 = pointing up."""
import math


def rotate_point(px, py, cx, cy, angle_deg):
    a = math.radians(angle_deg)
    dx, dy = px - cx, py - cy
    return (cx + dx * math.cos(a) - dy * math.sin(a),
            cy + dx * math.sin(a) + dy * math.cos(a))


def box_corners(box):
    cx, cy = box.center
    pts = [(box.x, box.y), (box.x + box.width, box.y),
           (box.x + box.width, box.y + box.height), (box.x, box.y + box.height)]
    if not box.angle:
        return pts
    return [rotate_point(px, py, cx, cy, box.angle) for px, py in pts]


def point_in_box(box, x, y):
    cx, cy = box.center
    lx, ly = rotate_point(x, y, cx, cy, -box.angle)
    return box.x <= lx <= box.x + box.width and box.y <= ly <= box.y + box.height


def point_in_polygon(points, x, y):
    inside = False
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            t = (y - y1) / (y2 - y1)
            if x < x1 + t * (x2 - x1):
                inside = not inside
    return inside


def aabb_of(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def angle_from_center(cx, cy, x, y):
    return math.degrees(math.atan2(x - cx, -(y - cy))) % 360


def clamp01(v):
    return max(0.0, min(1.0, v))
