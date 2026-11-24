"""Offline demo without a camera or AI model.

Generates a synthetic shop video with two scripted people:
  #1 — takes an item, pays at checkout, leaves      -> no alert
  #2 — takes an item, walks past checkout, leaves   -> ALERT
and runs the full pipeline (zones, behaviour, clips, storage, notifier) on it.

    python -m app.demo            # writes data/demo/annotated.mp4 and an event clip
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .behavior import Rules
from .config import ROOT
from .detector import ScriptedTracker
from .notifier import NullNotifier
from .pipeline import CameraPipeline
from .storage import EventStore
from .zones import ZoneMap

W, H, FPS = 1280, 720, 10
BW, BH, FOOT_Y = 80, 220, 620


def _box(cx: float) -> tuple[float, float, float, float]:
    return (cx - BW / 2, FOOT_Y - BH, cx + BW / 2, FOOT_Y)


# frame -> centre x. Shelf ~x200, checkout ~x690, exit ~x1150
SCRIPT = {
    1: [(0, _box(1150)), (30, _box(220)), (80, _box(220)), (100, _box(690)), (170, _box(690)),
        (195, _box(1150)), (210, _box(1150))],
    2: [(40, _box(1150)), (70, _box(200)), (120, _box(200)), (160, _box(1150)), (180, _box(1150))],
}
N_FRAMES = 230
PERSON_COLORS = {1: (180, 120, 60), 2: (60, 60, 170)}


def render_frame(tracker: ScriptedTracker, i: int) -> np.ndarray:
    img = np.full((H, W, 3), 205, np.uint8)
    cv2.rectangle(img, (0, FOOT_Y - 260), (W, H), (175, 175, 175), -1)  # floor
    cv2.rectangle(img, (40, 120), (420, 330), (120, 90, 60), -1)        # shelf
    for k in range(5):
        cv2.rectangle(img, (60 + k * 72, 150), (110 + k * 72, 200), (40, 160, 220), -1)
    cv2.rectangle(img, (560, 260), (820, 380), (90, 90, 90), -1)        # checkout desk
    cv2.rectangle(img, (1060, 80), (1240, 360), (70, 120, 70), -1)      # door
    for p in tracker.boxes_at(i):
        x1, y1, x2, y2 = map(int, p.bbox)
        c = PERSON_COLORS.get(p.track_id, (100, 100, 100))
        cv2.circle(img, ((x1 + x2) // 2, y1 + 25), 25, c, -1)
        cv2.rectangle(img, (x1 + 10, y1 + 50), (x2 - 10, y2), c, -1)
    return img


def run_demo(out_dir: Path, notifier=None, write_video: bool = True) -> tuple[CameraPipeline, EventStore]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zones = ZoneMap.load(ROOT / "config" / "zones" / "cam1.json")
    tracker = ScriptedTracker(SCRIPT)
    store = EventStore(out_dir / "events.db")
    pipe = CameraPipeline("demo", "Demo kamera", tracker, zones, Rules(), store, notifier or NullNotifier(),
                          out_dir, FPS, clip_pre=4, clip_post=3)
    writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H)) \
        if write_video else None
    for i in range(N_FRAMES):
        out = pipe.process(i, i / FPS, render_frame(tracker, i))
        if writer:
            writer.write(out)
    if writer:
        writer.release()
    pipe.finish()
    return pipe, store


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pipe, store = run_demo(ROOT / "data" / "demo")
    for e in store.list():
        print(f"Hodisa #{e['id']}: track {e['track_id']} [{e['severity']}] {e['reason']}\n  rasm: {e['snapshot_path']}\n  video: {e['clip_path']}")
