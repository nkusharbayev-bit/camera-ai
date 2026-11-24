"""Store-wide behaviour state machine.

Rule: a person who spent time at a shelf, did NOT spend time at the checkout,
and then reached the exit is flagged as *suspicious*. If the pose model also
saw a hand held at the hip/pocket after the shelf visit, severity is high.

Works with one camera or many: with a `GlobalRegistry`, the same person seen
on the shelf camera, the checkout camera and the door camera shares one state.

The system never decides that someone is a thief -- it only raises an alert
for a human to review.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

from .reid import GlobalRegistry, appearance_signature

# COCO keypoint indices used by YOLO pose models
L_WRIST, R_WRIST, L_HIP, R_HIP = 9, 10, 11, 12


@dataclass
class Rules:
    min_shelf_seconds: float = 3.0       # time at shelf before we assume an item may have been taken
    min_checkout_seconds: float = 4.0    # time at checkout that counts as "paid"
    min_exit_seconds: float = 0.3        # debounce: must stay in exit zone this long
    conceal_seconds: float = 1.0         # wrist-at-hip duration that counts as possible concealment
    require_concealment: bool = False    # if True, only alert when concealment was also seen
    track_ttl_seconds: float = 900.0     # forget a person after not being seen on any camera this long
    max_gap_seconds: float = 1.0         # ignore time gaps larger than this when accumulating dwell

    @classmethod
    def from_dict(cls, d: dict | None) -> "Rules":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrackState:
    track_id: int                 # global person id (or local track id in single-camera mode)
    first_seen: float
    last_seen: float
    shelf_seconds: float = 0.0
    checkout_seconds: float = 0.0
    exit_seconds: float = 0.0
    conceal_seconds: float = 0.0
    concealed: bool = False
    alerted: bool = False
    path: list[str] = field(default_factory=list)  # ordered list of zone kinds visited
    cameras: list[str] = field(default_factory=list)
    shelf_evidence: Optional[tuple[str, bytes]] = None  # (camera name, jpeg) from the shelf moment
    shelf_evidence_ts: float = -1e9

    @property
    def visited_shelf(self) -> bool:
        return "shelf" in self.path

    def note_zone(self, kind: str) -> None:
        if not self.path or self.path[-1] != kind:
            self.path.append(kind)

    def merge(self, other: "TrackState") -> None:
        self.first_seen = min(self.first_seen, other.first_seen)
        self.last_seen = max(self.last_seen, other.last_seen)
        self.shelf_seconds += other.shelf_seconds
        self.checkout_seconds += other.checkout_seconds
        self.conceal_seconds += other.conceal_seconds
        self.concealed |= other.concealed
        self.alerted |= other.alerted
        for k in other.path:
            self.note_zone(k)
        for c in other.cameras:
            if c not in self.cameras:
                self.cameras.append(c)
        if self.shelf_evidence is None:
            self.shelf_evidence = other.shelf_evidence


@dataclass
class Alert:
    track_id: int
    timestamp: float
    severity: str  # "high" | "medium"
    reason: str
    shelf_seconds: float
    checkout_seconds: float
    concealed: bool
    bbox: Optional[tuple[float, float, float, float]] = None
    camera_id: Optional[str] = None
    cameras: list[str] = field(default_factory=list)
    shelf_evidence: Optional[tuple[str, bytes]] = None


@dataclass
class Observation:
    """What the analyzer saw for one person on one camera in the last update (for drawing)."""
    person: object
    key: Optional[int]
    kinds: set


def hand_at_hip(keypoints, bbox, conf_thr: float = 0.4) -> bool:
    """Heuristic: a wrist is close to a hip (hand in pocket / under clothes / into bag)."""
    if keypoints is None:
        return False
    try:
        kp = [tuple(map(float, k)) for k in keypoints]
    except (TypeError, ValueError):
        return False
    if len(kp) <= R_HIP:
        return False
    x1, y1, x2, y2 = bbox
    h = max(y2 - y1, 1.0)
    thr = 0.12 * h

    def ok(i):
        return len(kp[i]) < 3 or kp[i][2] >= conf_thr

    for w in (L_WRIST, R_WRIST):
        if not ok(w):
            continue
        for hp in (L_HIP, R_HIP):
            if not ok(hp):
                continue
            dx, dy = kp[w][0] - kp[hp][0], kp[w][1] - kp[hp][1]
            if (dx * dx + dy * dy) ** 0.5 < thr:
                return True
    return False


class BehaviorAnalyzer:
    """One instance per store. Thread-safe: camera threads call update() concurrently."""

    def __init__(self, zone_map=None, rules: Rules | None = None, registry: GlobalRegistry | None = None):
        self.zone_map = zone_map          # default zones (single-camera use)
        self.rules = rules or Rules()
        self.registry = registry
        self.tracks: dict[int, TrackState] = {}
        self.last_observations: dict[str, list[Observation]] = {}
        self._lock = threading.Lock()

    def key_for(self, camera_id: str, track_id: int, frame=None, bbox=None, ts: float = 0.0) -> int:
        if self.registry is None:
            return track_id
        sig = appearance_signature(frame, bbox) if frame is not None and bbox is not None else None
        return self.registry.resolve(camera_id, track_id, sig, ts)

    def update(self, persons, timestamp: float, camera_id: str = "cam", zone_map=None, frame=None) -> list[Alert]:
        """persons: objects with .track_id, .bbox (x1,y1,x2,y2), .keypoints (or None)."""
        zm = zone_map or self.zone_map
        has_exit = any(z.kind == "exit" for z in zm.zones)
        r = self.rules
        alerts: list[Alert] = []
        observations: list[Observation] = []
        keyed = []
        for p in persons:  # appearance matching outside the lock (CPU work)
            key = None if p.track_id is None else self.key_for(camera_id, p.track_id, frame, p.bbox, timestamp)
            keyed.append((p, key))

        with self._lock:
            if self.registry is not None:
                for old, new in self.registry.pop_merges():
                    if old in self.tracks:
                        src = self.tracks.pop(old)
                        if new in self.tracks:
                            self.tracks[new].merge(src)
                        else:
                            src.track_id = new
                            self.tracks[new] = src

            for p, key in keyed:
                x1, y1, x2, y2 = p.bbox
                kinds = zm.kinds_at((x1 + x2) / 2.0, y2)  # floor position = bottom-centre of the box
                observations.append(Observation(p, key, kinds))
                if key is None:
                    continue
                st = self.tracks.get(key)
                if st is None:
                    st = self.tracks[key] = TrackState(key, timestamp, timestamp)
                dt = timestamp - st.last_seen
                if dt < 0 or dt > r.max_gap_seconds:
                    dt = 0.0
                st.last_seen = max(st.last_seen, timestamp)
                if camera_id not in st.cameras:
                    st.cameras.append(camera_id)

                for kind in ("shelf", "checkout", "exit"):
                    if kind in kinds:
                        st.note_zone(kind)
                if "shelf" in kinds:
                    st.shelf_seconds += dt
                if "checkout" in kinds:
                    st.checkout_seconds += dt
                if "exit" in kinds:
                    st.exit_seconds += dt
                elif has_exit:  # only a camera that can see the door may reset the exit timer
                    st.exit_seconds = 0.0

                if st.visited_shelf and "checkout" not in kinds and hand_at_hip(getattr(p, "keypoints", None), p.bbox):
                    st.conceal_seconds += dt
                    if st.conceal_seconds >= r.conceal_seconds:
                        st.concealed = True

                alert = self._check(st, timestamp, p.bbox, "exit" in kinds, camera_id)
                if alert:
                    alerts.append(alert)

            self.last_observations[camera_id] = observations
            self._expire(timestamp)
        return alerts

    def needs_shelf_evidence(self, key: int, ts: float, every: float = 1.0) -> bool:
        st = self.tracks.get(key)
        return st is not None and not st.alerted and ts - st.shelf_evidence_ts >= every

    def set_shelf_evidence(self, key: int, camera_name: str, jpeg: bytes, ts: float) -> None:
        with self._lock:
            st = self.tracks.get(key)
            if st is not None:
                st.shelf_evidence = (camera_name, jpeg)
                st.shelf_evidence_ts = ts

    def _check(self, st: TrackState, ts: float, bbox, in_exit: bool, camera_id: str) -> Optional[Alert]:
        r = self.rules
        if st.alerted or not in_exit or st.exit_seconds < r.min_exit_seconds:
            return None
        took_item = st.shelf_seconds >= r.min_shelf_seconds
        paid = st.checkout_seconds >= r.min_checkout_seconds
        if not took_item or paid:
            return None
        if r.require_concealment and not st.concealed:
            return None
        st.alerted = True
        if st.concealed:
            severity = "high"
            reason = "Javondan narsa oldi, yashirgandek harakat qildi va kassadan o'tmay chiqdi"
        else:
            severity = "medium"
            reason = "Javon oldida turdi va kassadan o'tmay chiqishga yo'naldi"
        return Alert(st.track_id, ts, severity, reason, round(st.shelf_seconds, 1),
                     round(st.checkout_seconds, 1), st.concealed, tuple(map(float, bbox)),
                     camera_id, list(st.cameras), st.shelf_evidence)

    def _expire(self, now: float) -> None:
        ttl = self.rules.track_ttl_seconds
        for tid in [t for t, s in self.tracks.items() if now - s.last_seen > ttl]:
            del self.tracks[tid]
