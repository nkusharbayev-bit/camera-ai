"""Multi-camera offline demo: a shop with 3 cameras.

  cam_shelf  — sees only the shelves
  cam_till   — sees only the checkout
  cam_door   — sees only the entrance/exit

Each camera gives its own local track ids (they do NOT match across cameras).
  Person A (blue shirt, black trousers): shelf -> checkout -> door  => no alert
  Person B (red shirt, beige trousers):  shelf -> door (no checkout) => ALERT
  Person C (green shirt): enters, walks to the shelf briefly, still inside => no alert

    python -m app.demo_store
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .behavior import BehaviorAnalyzer, Rules
from .config import ROOT
from .detector import ScriptedTracker
from .notifier import NullNotifier
from .pipeline import CameraPipeline
from .reid import GlobalRegistry
from .storage import EventStore
from .zones import Zone, ZoneMap

W, H, FPS = 960, 540, 10
BW, BH, FOOT = 90, 260, 500

PEOPLE = {  # name -> (shirt BGR, trousers BGR)
    "A": ((200, 90, 30), (30, 30, 30)),
    "B": ((40, 40, 200), (140, 190, 215)),
    "C": ((60, 170, 60), (110, 70, 40)),
}


def box(cx):
    return (cx - BW / 2, FOOT - BH, cx + BW / 2, FOOT)


# camera -> {local_track_id: (person, [(frame, box)...])}
SCRIPTS = {
    "cam_door": {   # door zone is on the right half of this camera
        11: ("A", [(0, box(800)), (15, box(300)), (20, box(100))]),
        12: ("B", [(10, box(800)), (25, box(300)), (30, box(100))]),
        13: ("C", [(20, box(800)), (35, box(300)), (40, box(100))]),
        14: ("A", [(205, box(100)), (220, box(800)), (235, box(800))]),
        15: ("B", [(150, box(100)), (165, box(800)), (180, box(800))]),
    },
    "cam_shelf": {
        21: ("A", [(30, box(100)), (45, box(450)), (95, box(450)), (110, box(900))]),
        22: ("B", [(40, box(100)), (55, box(600)), (110, box(600)), (125, box(900))]),
        23: ("C", [(50, box(100)), (60, box(300)), (75, box(300)), (85, box(80))]),
    },
    "cam_till": {
        31: ("A", [(120, box(100)), (135, box(480)), (185, box(480)), (195, box(900))]),
        32: ("B", [(130, box(100)), (140, box(300)), (145, box(900))]),  # walks past without stopping
    },
}

ZONES = {
    "cam_door": ZoneMap([Zone("Eshik", "exit", [(600, 300), (960, 300), (960, 540), (600, 540)])]),
    "cam_shelf": ZoneMap([Zone("Javon", "shelf", [(300, 300), (760, 300), (760, 540), (300, 540)])]),
    "cam_till": ZoneMap([Zone("Kassa", "checkout", [(380, 300), (600, 300), (600, 540), (380, 540)])]),
}
N_FRAMES = 245


def render(camera_id: str, tracker: ScriptedTracker, i: int) -> np.ndarray:
    img = np.full((H, W, 3), 200, np.uint8)
    cv2.rectangle(img, (0, 240), (W, H), (165, 165, 165), -1)
    cv2.putText(img, camera_id, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 60), 2)
    names = {tid: who for tid, (who, _) in SCRIPTS[camera_id].items()}
    for p in tracker.boxes_at(i):
        x1, y1, x2, y2 = map(int, p.bbox)
        shirt, trousers = PEOPLE[names[p.track_id]]
        cv2.circle(img, ((x1 + x2) // 2, y1 + 25), 24, (150, 180, 220), -1)
        cv2.rectangle(img, (x1 + 8, y1 + 50), (x2 - 8, y1 + 140), shirt, -1)
        cv2.rectangle(img, (x1 + 14, y1 + 140), (x2 - 14, y2), trousers, -1)
    return img


def run_store_demo(out_dir: Path, notifier=None) -> tuple[dict, EventStore, BehaviorAnalyzer]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = EventStore(out_dir / "events.db")
    analyzer = BehaviorAnalyzer(rules=Rules(), registry=GlobalRegistry())
    notifier = notifier or NullNotifier()
    pipes, trackers = {}, {}
    for cam, script in SCRIPTS.items():
        trackers[cam] = ScriptedTracker({tid: kfs for tid, (_, kfs) in script.items()})
        pipes[cam] = CameraPipeline(cam, cam, trackers[cam], ZONES[cam], analyzer.rules, store, notifier,
                                    out_dir, FPS, clip_pre=3, clip_post=2, analyzer=analyzer)
    for i in range(N_FRAMES):  # cameras run in lock-step here; in production each has its own thread
        for cam, pipe in pipes.items():
            pipe.process(i, i / FPS, render(cam, trackers[cam], i))
    for pipe in pipes.values():
        pipe.finish()
    return pipes, store, analyzer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _, store, _ = run_store_demo(ROOT / "data" / "demo_store")
    for e in store.list():
        print(f"Hodisa #{e['id']}: kamera {e['camera_id']}, kameralar {e['cameras']}\n"
              f"  {e['reason']}\n  eshik rasmi: {e['snapshot_path']}\n  javon rasmi: {e['shelf_snapshot_path']}")
