"""Cross-camera person matching by clothing colour (no face recognition).

Shops usually have 4-5 cameras: the shelf may be on one camera, the checkout
on another, the door on a third. To know whether the person at the door
visited the shelf and the checkout, each local track (camera, track id) is
mapped to a store-wide *global id* by comparing appearance signatures:
colour histograms of the upper and lower body (shirt / trousers).

This is deliberately simple and anonymous. It works when clothing differs;
two people dressed alike may be confused, which is why every alert carries
photos from both the shelf and the door for a human to check.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

H_BINS, S_BINS, V_BINS = 12, 4, 4


def appearance_signature(frame: np.ndarray, bbox) -> Optional[np.ndarray]:
    """Unit-length HSV histogram of upper (torso) + lower (legs) body. None if crop is too small."""
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    H, W = frame.shape[:2]
    x1, x2 = max(0, x1), min(W, x2)
    y1, y2 = max(0, y1), min(H, y2)
    w, h = x2 - x1, y2 - y1
    if w < 16 or h < 40:
        return None
    # trim the edges of the box (background) and the head
    cx1, cx2 = x1 + int(0.2 * w), x2 - int(0.2 * w)
    parts = [(y1 + int(0.18 * h), y1 + int(0.52 * h)), (y1 + int(0.55 * h), y1 + int(0.92 * h))]
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    feats = []
    for py1, py2 in parts:
        crop = hsv[py1 - y1:py2 - y1, cx1 - x1:cx2 - x1]
        if crop.size == 0:
            return None
        hist = cv2.calcHist([crop], [0, 1, 2], None, [H_BINS, S_BINS, V_BINS], [0, 180, 0, 256, 0, 256])
        hist = np.sqrt(hist.flatten())  # Hellinger-style: tames dominant colours
        feats.append(hist / (np.linalg.norm(hist) + 1e-9))
    v = np.concatenate(feats)
    return v / (np.linalg.norm(v) + 1e-9)


@dataclass
class _Identity:
    gid: int
    signature: np.ndarray
    last_seen: float
    cameras: dict = field(default_factory=dict)  # camera_id -> (local track id, last seen on that camera)


class GlobalRegistry:
    """Thread-safe map (camera_id, local_track_id) -> global person id."""

    def __init__(self, match_threshold: float = 0.80, window_seconds: float = 900.0,
                 min_samples: int = 3, ema: float = 0.15):
        self.match_threshold = match_threshold
        self.window_seconds = window_seconds
        self.min_samples = min_samples
        self.ema = ema
        self._ids: dict[int, _Identity] = {}
        self._local: dict[tuple[str, int], dict] = {}  # -> {"gid", "sig", "n", "final"}
        self._next = 1
        self._merges: list[tuple[int, int]] = []  # (provisional gid, matched earlier gid)
        self._lock = threading.Lock()

    def resolve(self, camera_id: str, track_id: int, signature: Optional[np.ndarray], ts: float) -> int:
        key = (camera_id, track_id)
        with self._lock:
            loc = self._local.get(key)
            if loc is None:
                loc = self._local[key] = {"gid": None, "sig": None, "n": 0, "final": False}
            if signature is not None:
                loc["sig"] = signature if loc["sig"] is None else self._blend(loc["sig"], signature, 0.3)
                loc["n"] += 1

            # While the track is young we may still re-link it to an earlier identity.
            if not loc["final"] and loc["sig"] is not None and loc["n"] >= self.min_samples:
                best = self._best_match(camera_id, track_id, loc["sig"], ts)
                if best is not None and best != loc["gid"]:
                    if loc["gid"] is not None:
                        self._merges.append((loc["gid"], best))
                    self._release(key, loc["gid"])
                    loc["gid"] = best
                loc["final"] = True

            if loc["gid"] is None:
                loc["gid"] = self._new_identity(loc["sig"], ts)
            ident = self._ids[loc["gid"]]
            ident.last_seen = ts
            ident.cameras[camera_id] = (track_id, ts)
            if signature is not None:
                ident.signature = signature if ident.signature is None else self._blend(ident.signature, signature, self.ema)
            self._expire(ts)
            return loc["gid"]

    def pop_merges(self) -> list[tuple[int, int]]:
        with self._lock:
            m, self._merges = self._merges, []
            return m

    def is_final(self, camera_id: str, track_id: int) -> bool:
        loc = self._local.get((camera_id, track_id))
        return bool(loc and loc["final"])

    # -- internals (lock held) --
    @staticmethod
    def _blend(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
        v = (1 - w) * a + w * b
        return v / (np.linalg.norm(v) + 1e-9)

    def _new_identity(self, sig, ts) -> int:
        gid = self._next
        self._next += 1
        self._ids[gid] = _Identity(gid, sig, ts)
        return gid

    def _best_match(self, camera_id, track_id, sig, ts) -> Optional[int]:
        best, best_sim = None, self.match_threshold
        own_gid = self._local[(camera_id, track_id)]["gid"]
        for gid, ident in self._ids.items():
            if gid == own_gid or ident.signature is None or ts - ident.last_seen > self.window_seconds:
                continue
            # the same person cannot be two different live tracks on the same camera
            other = ident.cameras.get(camera_id)
            if other is not None and other[0] != track_id and ts - other[1] < 2.0:
                continue
            sim = float(np.dot(sig, ident.signature))
            if sim > best_sim:
                best, best_sim = gid, sim
        return best

    def _release(self, key, gid) -> None:
        if gid is None or gid not in self._ids:
            return
        ident = self._ids[gid]
        if ident.cameras.get(key[0], (None,))[0] == key[1]:
            del ident.cameras[key[0]]
        # drop the provisional identity if nothing else refers to it
        if not ident.cameras and not any(l["gid"] == gid for k, l in self._local.items() if k != key):
            del self._ids[gid]

    def _expire(self, now: float) -> None:
        old = [g for g, i in self._ids.items() if now - i.last_seen > self.window_seconds]
        for g in old:
            del self._ids[g]
        if old:
            gone = set(old)
            self._local = {k: v for k, v in self._local.items() if v["gid"] not in gone}
