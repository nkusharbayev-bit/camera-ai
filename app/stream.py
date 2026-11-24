"""Video sources: RTSP camera streams (with auto-reconnect) and local files."""
from __future__ import annotations

import logging
import threading
import time
from typing import Iterator

import cv2
import numpy as np

log = logging.getLogger(__name__)


def is_live(source: str) -> bool:
    s = str(source).lower()
    return s.startswith(("rtsp://", "rtmp://", "http://", "https://")) or s.isdigit()


class FileSource:
    """Reads every frame of a video file; timestamps come from the file's FPS."""

    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Videoni ochib bo'lmadi: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0

    def frames(self) -> Iterator[tuple[int, float, np.ndarray]]:
        i = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield i, i / self.fps, frame
            i += 1
        self.cap.release()


class RtspSource:
    """Grabs frames in a background thread and always hands out the newest one,
    so slow inference never builds up lag. Reconnects if the camera drops."""

    def __init__(self, url: str, target_fps: float = 10.0, reconnect_delay: float = 3.0):
        self.url = url
        self.target_fps = target_fps
        self.fps = target_fps
        self.reconnect_delay = reconnect_delay
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _open(self):
        cap = cv2.VideoCapture(int(self.url)) if self.url.isdigit() else cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _reader(self) -> None:
        while not self._stop.is_set():
            cap = self._open()
            if not cap.isOpened():
                log.warning("Kameraga ulanib bo'lmadi, %ss dan keyin qayta urinish", self.reconnect_delay)
                time.sleep(self.reconnect_delay)
                continue
            log.info("Kameraga ulandi")
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    log.warning("Kamera oqimi uzildi, qayta ulanmoqda")
                    break
                with self._lock:
                    self._frame = frame
                    self._seq += 1
            cap.release()
            time.sleep(self.reconnect_delay)

    def frames(self) -> Iterator[tuple[int, float, np.ndarray]]:
        i, last_seq, period = 0, -1, 1.0 / self.target_fps
        try:
            while not self._stop.is_set():
                t0 = time.time()
                with self._lock:
                    frame, seq = self._frame, self._seq
                if frame is not None and seq != last_seq:
                    last_seq = seq
                    yield i, time.time(), frame.copy()
                    i += 1
                time.sleep(max(0.0, period - (time.time() - t0)))
        finally:
            self.stop()

    def stop(self) -> None:
        self._stop.set()


def open_source(source: str, target_fps: float = 10.0):
    return RtspSource(source, target_fps) if is_live(source) else FileSource(source)
