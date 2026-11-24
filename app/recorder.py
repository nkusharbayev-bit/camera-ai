"""Keeps the last N seconds of frames in memory and writes an evidence clip
(pre + post seconds around the alert) when an event fires."""
from __future__ import annotations

import collections
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class _PendingClip:
    event_id: int
    path: Path
    frames: list[np.ndarray]
    remaining: int
    on_done: Callable[[int, str], None]


@dataclass
class ClipRecorder:
    out_dir: Path
    fps: float
    pre_seconds: float = 6.0
    post_seconds: float = 6.0
    _buffer: collections.deque = field(init=False)
    _pending: list[_PendingClip] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._buffer = collections.deque(maxlen=max(1, int(self.pre_seconds * self.fps)))

    def push(self, frame: np.ndarray) -> None:
        self._buffer.append(frame)
        done = []
        for clip in self._pending:
            clip.frames.append(frame)
            clip.remaining -= 1
            if clip.remaining <= 0:
                done.append(clip)
        for clip in done:
            self._pending.remove(clip)
            self._write(clip)

    def start_clip(self, event_id: int, on_done: Callable[[int, str], None]) -> None:
        path = self.out_dir / f"event_{event_id}.mp4"
        self._pending.append(_PendingClip(event_id, path, list(self._buffer),
                                          max(1, int(self.post_seconds * self.fps)), on_done))

    def flush(self) -> None:
        """Write any clips still waiting for post-event frames (e.g. the video ended)."""
        for clip in list(self._pending):
            self._pending.remove(clip)
            self._write(clip)

    def _write(self, clip: _PendingClip) -> None:
        if not clip.frames:
            return
        h, w = clip.frames[0].shape[:2]
        writer = cv2.VideoWriter(str(clip.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        for f in clip.frames:
            writer.write(f)
        writer.release()
        log.info("Klip saqlandi: %s", clip.path)
        try:
            clip.on_done(clip.event_id, str(clip.path))
        except Exception:  # never let a notifier error kill the video loop
            log.exception("Klip callback xatosi")


def save_snapshot(frame: np.ndarray, bbox, path: Path) -> str:
    img = frame.copy()
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return str(path)
