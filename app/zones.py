"""Store zones (shelf / checkout / exit) defined as polygons in image pixels."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ZONE_KINDS = ("shelf", "checkout", "exit")


@dataclass
class Zone:
    name: str
    kind: str  # one of ZONE_KINDS
    points: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kind not in ZONE_KINDS:
            raise ValueError(f"Unknown zone kind {self.kind!r}; expected one of {ZONE_KINDS}")
        if len(self.points) < 3:
            raise ValueError(f"Zone {self.name!r} needs at least 3 points")
        self.points = [(float(x), float(y)) for x, y in self.points]

    def contains(self, x: float, y: float) -> bool:
        """Ray-casting point-in-polygon test."""
        inside = False
        pts = self.points
        j = len(pts) - 1
        for i in range(len(pts)):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if (yi > y) != (yj > y):
                x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x < x_cross:
                    inside = not inside
            j = i
        return inside

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "points": [list(p) for p in self.points]}


class ZoneMap:
    def __init__(self, zones: list[Zone]):
        self.zones = zones

    def kinds_at(self, x: float, y: float) -> set[str]:
        return {z.kind for z in self.zones if z.contains(x, y)}

    def to_dict(self) -> dict:
        return {"zones": [z.to_dict() for z in self.zones]}

    @classmethod
    def from_dict(cls, data: dict) -> "ZoneMap":
        return cls([Zone(z["name"], z["kind"], [tuple(p) for p in z["points"]]) for z in data.get("zones", [])])

    @classmethod
    def load(cls, path: str | Path) -> "ZoneMap":
        path = Path(path)
        if not path.exists():
            return cls([])
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
