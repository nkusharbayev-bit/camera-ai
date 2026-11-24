"""Loads config/cameras.yaml. Secrets (Telegram token) come from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .behavior import Rules

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class CameraConfig:
    id: str
    name: str
    source: str
    zones_file: Path
    fps: float = 10.0


@dataclass
class AppConfig:
    cameras: list[CameraConfig]
    rules: Rules
    data_dir: Path
    model: str = "yolo11n-pose.pt"
    device: str | None = None
    conf: float = 0.4
    clip_pre_seconds: float = 6.0
    clip_post_seconds: float = 6.0
    telegram_token: str | None = None
    telegram_chat_id: str | None = None
    extra: dict = field(default_factory=dict)

    def camera(self, camera_id: str) -> CameraConfig:
        for c in self.cameras:
            if c.id == camera_id:
                return c
        raise KeyError(f"Kamera topilmadi: {camera_id}")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "events.db"


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_config(path: str | Path | None = None) -> AppConfig:
    path = _resolve(path or os.environ.get("CAMERA_AI_CONFIG", "config/cameras.yaml"))
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cams = [CameraConfig(id=str(c["id"]), name=c.get("name", c["id"]),
                         source=os.path.expandvars(str(c["source"])),
                         zones_file=_resolve(c.get("zones", f"config/zones/{c['id']}.json")),
                         fps=float(c.get("fps", 10)))
            for c in raw.get("cameras", [])]
    det = raw.get("detector", {})
    clip = raw.get("clip", {})
    return AppConfig(
        cameras=cams,
        rules=Rules.from_dict(raw.get("rules")),
        data_dir=_resolve(raw.get("data_dir", "data")),
        model=det.get("model", "yolo11n-pose.pt"),
        device=det.get("device"),
        conf=float(det.get("conf", 0.4)),
        clip_pre_seconds=float(clip.get("pre_seconds", 6)),
        clip_post_seconds=float(clip.get("post_seconds", 6)),
        telegram_token=os.environ.get("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID"),
    )
