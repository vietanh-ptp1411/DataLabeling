import uuid
from dataclasses import dataclass, field


@dataclass
class PolygonAnnotation:
    class_name: str
    points: list  # list[tuple[float, float]]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        return {"Id": self.id, "ClassName": self.class_name,
                "Points": [{"X": x, "Y": y} for x, y in self.points]}

    @classmethod
    def from_dict(cls, d: dict) -> "PolygonAnnotation":
        pts = [(p["X"], p["Y"]) for p in d.get("Points", [])]
        return cls(d["ClassName"], pts, d.get("Id") or str(uuid.uuid4()))
