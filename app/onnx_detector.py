"""Lightweight person + pose detection with onnxruntime (no torch).

Used by the Windows store agent so the installer stays small and runs fast on
an ordinary CPU. Expects a YOLO pose model exported to ONNX
(output shape (1, 56, N): 4 box + 1 score + 17*3 keypoints).

Tracking is a simple IoU tracker: good enough for a shop camera where a few
people move at walking speed, and it has no extra dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .detector import Person


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out, r, (left, top)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


@dataclass
class _Track:
    id: int
    box: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))
    hits: int = 1
    missed: int = 0


class IouTracker:
    """Greedy IoU matching with constant-velocity prediction."""

    def __init__(self, iou_threshold: float = 0.25, max_missed: int = 15, min_hits: int = 2):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.tracks: list[_Track] = []
        self._next_id = 1

    def update(self, boxes: np.ndarray) -> list[Optional[int]]:
        """boxes: (N, 4). Returns a track id per box (None while a track is still tentative)."""
        predicted = np.array([t.box + t.velocity for t in self.tracks]).reshape(-1, 4)
        ious = iou_matrix(predicted, boxes)
        assigned: list[Optional[int]] = [None] * len(boxes)
        used_t, used_d = set(), set()
        for ti, di in sorted(np.ndindex(*ious.shape), key=lambda p: -ious[p]) if ious.size else []:
            if ious[ti, di] < self.iou_threshold:
                break
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t = self.tracks[ti]
            t.velocity = 0.5 * t.velocity + 0.5 * (boxes[di] - t.box)
            t.box = boxes[di].astype(float)
            t.hits += 1
            t.missed = 0
            assigned[di] = t.id if t.hits >= self.min_hits else None
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.missed += 1
                t.box = t.box + t.velocity
                t.velocity *= 0.5
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        for di in range(len(boxes)):
            if di not in used_d:
                self.tracks.append(_Track(self._next_id, boxes[di].astype(float)))
                if self.min_hits <= 1:
                    assigned[di] = self._next_id
                self._next_id += 1
        return assigned


class OnnxPoseTracker:
    def __init__(self, model_path: str, conf: float = 0.4, iou: float = 0.5, imgsz: int = 640,
                 threads: int = 2):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.tracker = IouTracker()

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        img, r, (left, top) = letterbox(frame, self.imgsz)
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.session.run(None, {self.input_name: blob})[0][0].T  # (N, 56)
        scores = out[:, 4]
        keep = scores >= self.conf
        out, scores = out[keep], scores[keep]
        if len(out) == 0:
            return np.zeros((0, 4)), np.zeros(0), np.zeros((0, 17, 3))
        cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        idx = cv2.dnn.NMSBoxes([[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes],
                               scores.tolist(), self.conf, self.iou)
        idx = np.array(idx).reshape(-1)
        boxes, scores, kps = boxes[idx], scores[idx], out[idx, 5:].reshape(-1, 17, 3).copy()
        # undo letterbox
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / r
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / r
        kps[:, :, 0] = (kps[:, :, 0] - left) / r
        kps[:, :, 1] = (kps[:, :, 1] - top) / r
        H, W = frame.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
        return boxes, scores, kps

    def update(self, frame: np.ndarray, frame_index: int) -> list[Person]:
        boxes, scores, kps = self.detect(frame)
        ids = self.tracker.update(boxes)
        return [Person(tid, tuple(map(float, b)), float(s), k) for tid, b, s, k in zip(ids, boxes, scores, kps)]
