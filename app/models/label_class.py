import hashlib
from dataclasses import dataclass

PALETTE = [
    "#FFD700", "#FF6347", "#32CD32", "#1E90FF", "#FF69B4", "#8A2BE2",
    "#00CED1", "#FFA500", "#ADFF2F", "#DC143C", "#00FA9A", "#4169E1",
    "#FF4500", "#9370DB", "#20B2AA", "#F08080",
]


def stable_color(name: str) -> str:
    digest = int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16)
    return PALETTE[digest % len(PALETTE)]


@dataclass
class LabelClass:
    name: str
    color: str = ""

    def __post_init__(self):
        if not self.color:
            self.color = stable_color(self.name)

    def to_dict(self) -> dict:
        return {"Name": self.name, "Color": self.color}

    @classmethod
    def from_dict(cls, d: dict) -> "LabelClass":
        return cls(d["Name"], d.get("Color", ""))
