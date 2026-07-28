import uuid
from dataclasses import dataclass, field


@dataclass
class BoundingBox:
    class_name: str
    x: float
    y: float
    width: float
    height: float
    angle: float = 0.0  # degrees, 0 = upright, clockwise
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    def clone(self) -> "BoundingBox":
        return BoundingBox(self.class_name, self.x, self.y,
                           self.width, self.height, self.angle)

    def to_dict(self) -> dict:
        return {"Id": self.id, "ClassName": self.class_name, "X": self.x,
                "Y": self.y, "Width": self.width, "Height": self.height,
                "Angle": self.angle}

    @classmethod
    def from_dict(cls, d: dict) -> "BoundingBox":
        return cls(d["ClassName"], d["X"], d["Y"], d["Width"], d["Height"],
                   d.get("Angle", 0.0), d.get("Id") or str(uuid.uuid4()))
