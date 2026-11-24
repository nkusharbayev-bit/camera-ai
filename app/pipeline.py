"""Per-camera processing: frames -> tracker -> (shared, store-wide) behaviour -> events.

Several CameraPipeline objects (one per camera) share one BehaviorAnalyzer so
that a person seen at the shelf on camera 1 and at the door on camera 3 is
treated as the same visit.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .behavior import BehaviorAnalyzer, Rules
from .detector import PersonTracker
from .recorder import ClipRecorder, save_snapshot
from .reid import GlobalRegistry
from .storage import EventStore
from .zones import ZoneMap

log = logging.getLogger(__name__)

ZONE_COLORS = {"shelf": (0, 170, 255), "checkout": (0, 200, 0), "exit": (32, 32, 255)}  # BGR, same as web panel
ZONE_LABELS = {"shelf": "JAVON", "checkout": "KASSA", "exit": "CHIQISH"}


def draw_overlay(frame: np.ndarray, zone_map: ZoneMap, observations, analyzer: BehaviorAnalyzer) -> np.ndarray:
    img = frame.copy()
    layer = img.copy()
    for z in zone_map.zones:
        cv2.fillPoly(layer, [np.array(z.points, dtype=np.int32)], ZONE_COLORS[z.kind])
    img = cv2.addWeighted(layer, 0.2, img, 0.8, 0)
    for z in zone_map.zones:
        pts = np.array(z.points, dtype=np.int32)
        cv2.polylines(img, [pts], True, ZONE_COLORS[z.kind], 2)
        cv2.putText(img, ZONE_LABELS[z.kind], tuple(int(v) for v in pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    ZONE_COLORS[z.kind], 2)
    for ob in observations:
        st = analyzer.tracks.get(ob.key) if ob.key is not None else None
        color = (0, 0, 255) if st and st.alerted else (0, 165, 255) if st and st.visited_shelf else (255, 255, 255)
        x1, y1, x2, y2 = map(int, ob.person.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"#{ob.key}" if ob.key is not None else "?"
        if st:
            label += f" J:{st.shelf_seconds:.0f}s K:{st.checkout_seconds:.0f}s"
            if st.concealed:
                label += " !"
        cv2.putText(img, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


def evidence_jpeg(frame: np.ndarray, bbox, max_width: int = 960) -> bytes:
    img = frame.copy()
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 3)
    h, w = img.shape[:2]
    if w > max_width:
        img = cv2.resize(img, (max_width, int(h * max_width / w)))
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


class CameraPipeline:
    def __init__(self, camera_id: str, camera_name: str, tracker: PersonTracker, zone_map: ZoneMap,
                 rules: Rules, store: EventStore, notifier, data_dir: Path, fps: float,
                 clip_pre: float = 6.0, clip_post: float = 6.0, analyzer: BehaviorAnalyzer | None = None):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.tracker = tracker
        self.zone_map = zone_map
        self.analyzer = analyzer or BehaviorAnalyzer(zone_map, rules, GlobalRegistry())
        self.store = store
        self.notifier = notifier
        self.data_dir = Path(data_dir)
        self.recorder = ClipRecorder(self.data_dir / "clips", fps, clip_pre, clip_post)
        self.latest_frame_path = self.data_dir / "frames" / f"{camera_id}.jpg"
        self.event_ids: list[int] = []

    def process(self, frame_index: int, ts: float, frame: np.ndarray) -> np.ndarray:
        persons = self.tracker.update(frame, frame_index)
        alerts = self.analyzer.update(persons, ts, self.camera_id, self.zone_map, frame)
        observations = self.analyzer.last_observations.get(self.camera_id, [])

        for ob in observations:  # keep a photo of each person at the shelf as evidence
            if ob.key is not None and "shelf" in ob.kinds and self.analyzer.needs_shelf_evidence(ob.key, ts):
                self.analyzer.set_shelf_evidence(ob.key, self.camera_name, evidence_jpeg(frame, ob.person.bbox), ts)

        annotated = draw_overlay(frame, self.zone_map, observations, self.analyzer)
        self.recorder.push(annotated)

        if frame_index % 50 == 0:  # raw frame for drawing zones in the settings UI
            self.latest_frame_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.latest_frame_path), frame)

        for alert in alerts:
            self._raise(alert, annotated)
        return annotated

    def _raise(self, alert, annotated: np.ndarray) -> None:
        event_id = self.store.add(self.camera_id, alert)
        snaps = self.data_dir / "snapshots"
        self.store.set_snapshot(event_id, save_snapshot(annotated, alert.bbox, snaps / f"event_{event_id}.jpg"))
        if alert.shelf_evidence:
            cam_name, jpeg = alert.shelf_evidence
            p = snaps / f"event_{event_id}_shelf.jpg"
            p.write_bytes(jpeg)
            self.store.set_shelf_snapshot(event_id, str(p), cam_name)
        self.event_ids.append(event_id)
        log.warning("Shubhali holat #%s (kamera %s, odam #%s): %s", event_id, self.camera_name,
                    alert.track_id, alert.reason)
        self.notifier.send_photo(self.store.get(event_id), self.camera_name)
        self.recorder.start_clip(event_id, self._clip_done)

    def _clip_done(self, event_id: int, path: str) -> None:
        self.store.set_clip(event_id, path)
        self.notifier.send_clip(self.store.get(event_id), self.camera_name)

    def finish(self) -> None:
        self.recorder.flush()
        self.notifier.wait()
