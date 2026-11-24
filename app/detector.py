"""Person detection + tracking.

`YoloPoseTracker` uses an Ultralytics YOLO pose model with ByteTrack, which
gives us, per frame, a stable track id, a box and 17 body keypoints for every
person. `ScriptedTracker` replays pre-defined tracks and is used for tests and
the offline demo (no model download needed).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


@dataclass
class Person:
    track_id: Optional[int]
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    confidence: float = 1.0
    keypoints: Optional[np.ndarray] = None   # (17, 3): x, y, conf


class PersonTracker(Protocol):
    def update(self, frame: np.ndarray, frame_index: int) -> list[Person]: ...


class YoloPoseTracker:
    def __init__(self, model: str = "yolo11n-pose.pt", conf: float = 0.4, imgsz: int = 640,
                 device: Optional[str] = None, tracker: str = "bytetrack.yaml"):
        from ultralytics import YOLO  # imported lazily: heavy dependency

        self.model = YOLO(model)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.tracker = tracker

    def update(self, frame: np.ndarray, frame_index: int) -> list[Person]:
        res = self.model.track(frame, persist=True, tracker=self.tracker, classes=[0], conf=self.conf,
                               imgsz=self.imgsz, device=self.device, verbose=False)[0]
        persons: list[Person] = []
        if res.boxes is None or len(res.boxes) == 0:
            return persons
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        ids = res.boxes.id.int().cpu().numpy() if res.boxes.id is not None else [None] * len(boxes)
        kps = res.keypoints.data.cpu().numpy() if res.keypoints is not None else [None] * len(boxes)
        for b, c, i, k in zip(boxes, confs, ids, kps):
            persons.append(Person(None if i is None else int(i), tuple(map(float, b)), float(c), k))
        return persons


class ScriptedTracker:
    """Replays tracks: {track_id: [(frame_index, (x1,y1,x2,y2)), ...]} with linear interpolation."""

    def __init__(self, script: dict[int, list[tuple[int, tuple[float, float, float, float]]]]):
        self.script = {tid: sorted(kfs) for tid, kfs in script.items()}

    def boxes_at(self, frame_index: int) -> list[Person]:
        out = []
        for tid, kfs in self.script.items():
            if frame_index < kfs[0][0] or frame_index > kfs[-1][0]:
                continue
            for (f0, b0), (f1, b1) in zip(kfs, kfs[1:]):
                if f0 <= frame_index <= f1:
                    t = 0.0 if f1 == f0 else (frame_index - f0) / (f1 - f0)
                    box = tuple(a + (b - a) * t for a, b in zip(b0, b1))
                    out.append(Person(tid, box))
                    break
            else:
                out.append(Person(tid, tuple(kfs[-1][1])))
        return out

    def update(self, frame: np.ndarray, frame_index: int) -> list[Person]:
        return self.boxes_at(frame_index)
