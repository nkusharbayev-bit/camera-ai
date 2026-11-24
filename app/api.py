"""Web panel API: events list/review, zone editor.

    uvicorn app.api:create_app --factory --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

from .config import ROOT, load_config
from .storage import STATUSES, EventStore
from .zones import ZONE_KINDS, Zone, ZoneMap

security = HTTPBasic(auto_error=False)


class Review(BaseModel):
    status: str


class ZoneIn(BaseModel):
    name: str
    kind: str
    points: list[list[float]]


class ZonesIn(BaseModel):
    zones: list[ZoneIn]


def create_app(config_path: str | None = None) -> FastAPI:
    cfg = load_config(config_path)
    store = EventStore(cfg.db_path)
    user = os.environ.get("PANEL_USER", "admin")
    password = os.environ.get("PANEL_PASSWORD")

    def auth(creds: HTTPBasicCredentials | None = Depends(security)):
        if not password:
            return  # no password configured: local-only use
        if not creds or not (secrets.compare_digest(creds.username, user)
                             and secrets.compare_digest(creds.password, password)):
            raise HTTPException(401, "Login kerak", headers={"WWW-Authenticate": "Basic"})

    app = FastAPI(title="Camera AI", dependencies=[Depends(auth)])

    def _file(path: str | None) -> FileResponse:
        if not path or not Path(path).exists():
            raise HTTPException(404, "Fayl topilmadi")
        # only serve files inside the data dir
        if cfg.data_dir.resolve() not in Path(path).resolve().parents:
            raise HTTPException(403)
        return FileResponse(path)

    @app.get("/")
    def index():
        return FileResponse(ROOT / "web" / "index.html")

    @app.get("/api/cameras")
    def cameras():
        return [{"id": c.id, "name": c.name} for c in cfg.cameras]

    @app.get("/api/events")
    def events(limit: int = 100, camera_id: str | None = None, status: str | None = None):
        return store.list(limit, camera_id, status)

    @app.get("/api/events/{event_id}/snapshot")
    def snapshot(event_id: int):
        e = store.get(event_id) or {}
        return _file(e.get("snapshot_path"))

    @app.get("/api/events/{event_id}/clip")
    def clip(event_id: int):
        e = store.get(event_id) or {}
        return _file(e.get("clip_path"))

    @app.post("/api/events/{event_id}/review")
    def review(event_id: int, body: Review):
        if body.status not in STATUSES:
            raise HTTPException(400, f"status: {STATUSES}")
        if not store.review(event_id, body.status, "panel"):
            raise HTTPException(404)
        return store.get(event_id)

    @app.get("/api/cameras/{camera_id}/frame")
    def frame(camera_id: str):
        try:
            cfg.camera(camera_id)
        except KeyError:
            raise HTTPException(404)
        return _file(str(cfg.data_dir / "frames" / f"{camera_id}.jpg"))

    @app.get("/api/cameras/{camera_id}/zones")
    def get_zones(camera_id: str):
        try:
            return ZoneMap.load(cfg.camera(camera_id).zones_file).to_dict()
        except KeyError:
            raise HTTPException(404)

    @app.put("/api/cameras/{camera_id}/zones")
    def put_zones(camera_id: str, body: ZonesIn):
        try:
            cam = cfg.camera(camera_id)
            zm = ZoneMap([Zone(z.name, z.kind, [tuple(p) for p in z.points]) for z in body.zones])
        except KeyError:
            raise HTTPException(404)
        except ValueError as e:
            raise HTTPException(400, f"{e}. Turlar: {ZONE_KINDS}")
        zm.save(cam.zones_file)
        return {"ok": True, "note": "Kamera jarayonini qayta ishga tushiring"}

    return app

