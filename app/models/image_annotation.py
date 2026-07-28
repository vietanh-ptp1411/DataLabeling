import os
from dataclasses import dataclass, field

from app.models.bounding_box import BoundingBox
from app.models.polygon import PolygonAnnotation


@dataclass
class ImageAnnotation:
    image_path: str
    boxes: list = field(default_factory=list)
    polygons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"ImagePath": self.image_path,
                "ImageFileName": os.path.basename(self.image_path),
                "BoundingBoxes": [b.to_dict() for b in self.boxes],
                "Polygons": [p.to_dict() for p in self.polygons]}

    @classmethod
    def from_dict(cls, d: dict) -> "ImageAnnotation":
        return cls(d.get("ImagePath", ""),
                   [BoundingBox.from_dict(b) for b in d.get("BoundingBoxes", [])],
                   [PolygonAnnotation.from_dict(p) for p in d.get("Polygons", [])])
