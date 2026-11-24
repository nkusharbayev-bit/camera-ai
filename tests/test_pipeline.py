import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.bot import ReviewBot
from app.demo import run_demo
from app.notifier import TelegramNotifier
from app.storage import EventStore


class RecordingNotifier:
    def __init__(self):
        self.photos, self.clips = [], []

    def send_photo(self, event, cam):
        self.photos.append(event)

    def send_clip(self, event, cam):
        self.clips.append(event)

    def wait(self, timeout=0):
        pass


def test_demo_end_to_end(tmp_path):
    n = RecordingNotifier()
    pipe, store = run_demo(tmp_path, notifier=n, write_video=False)
    events = store.list()
    assert len(events) == 1, events
    e = events[0]
    assert e["track_id"] == 2  # the one who skipped the checkout
    assert Path(e["snapshot_path"]).exists() and Path(e["clip_path"]).stat().st_size > 0
    assert len(n.photos) == 1 and len(n.clips) == 1


def test_telegram_notifier_sends_photo_and_video(tmp_path):
    calls = []

    def handler(req: httpx.Request):
        calls.append(req)
        return httpx.Response(200, json={"ok": True, "result": {}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    snap, clip = tmp_path / "s.jpg", tmp_path / "c.mp4"
    snap.write_bytes(b"jpg")
    clip.write_bytes(b"mp4")
    ev = {"id": 7, "created_at": 0, "severity": "high", "reason": "test", "shelf_seconds": 5,
          "checkout_seconds": 0, "snapshot_path": str(snap), "clip_path": str(clip)}
    tn = TelegramNotifier("TOKEN", "123", client=client)
    tn.send_photo(ev, "cam")
    tn.send_clip(ev, "cam")
    tn.wait(5)
    assert [c.url.path for c in calls] == ["/botTOKEN/sendPhoto", "/botTOKEN/sendVideo"]
    assert b"confirm:7" in calls[0].content


def test_bot_review_callback(tmp_path):
    store = EventStore(tmp_path / "e.db")
    from app.behavior import Alert
    eid = store.add("cam1", Alert(1, 0.0, "medium", "r", 5, 0, False))
    sent = []
    client = httpx.Client(transport=httpx.MockTransport(
        lambda req: sent.append(json.loads(req.content)) or httpx.Response(200, json={"ok": True})))
    bot = ReviewBot("T", store, allowed_chat_id="42", client=client)
    bot.handle({"callback_query": {"id": "q", "data": f"confirm:{eid}", "from": {"username": "ali"},
                                   "message": {"message_id": 1, "chat": {"id": 42}}}})
    assert store.get(eid)["status"] == "confirmed" and store.get(eid)["reviewed_by"] == "ali"
    # other chats are not allowed to review
    bot.handle({"callback_query": {"id": "q2", "data": f"false:{eid}", "from": {"id": 9},
                                   "message": {"message_id": 1, "chat": {"id": 999}}}})
    assert store.get(eid)["status"] == "confirmed"


@pytest.fixture
def api(tmp_path, monkeypatch):
    cfg = tmp_path / "cameras.yaml"
    cfg.write_text(f"""
cameras:
  - id: cam1
    name: Test
    source: video.mp4
    zones: {tmp_path / 'zones.json'}
data_dir: {tmp_path / 'data'}
""")
    monkeypatch.delenv("PANEL_PASSWORD", raising=False)
    from app.api import create_app
    return TestClient(create_app(str(cfg))), tmp_path


def test_api_zones_and_events(api):
    client, tmp = api
    assert client.get("/api/cameras").json() == [{"id": "cam1", "name": "Test"}]
    zones = {"zones": [{"name": "J", "kind": "shelf", "points": [[0, 0], [10, 0], [10, 10]]}]}
    assert client.put("/api/cameras/cam1/zones", json=zones).status_code == 200
    assert client.get("/api/cameras/cam1/zones").json()["zones"][0]["kind"] == "shelf"
    bad = {"zones": [{"name": "X", "kind": "toilet", "points": [[0, 0], [1, 0], [1, 1]]}]}
    assert client.put("/api/cameras/cam1/zones", json=bad).status_code == 400

    from app.behavior import Alert
    store = EventStore(tmp / "data" / "events.db")
    eid = store.add("cam1", Alert(3, 1.0, "high", "r", 5, 0, True))
    assert client.get("/api/events").json()[0]["id"] == eid
    r = client.post(f"/api/events/{eid}/review", json={"status": "false_alarm"})
    assert r.json()["status"] == "false_alarm"
    assert client.get(f"/api/events/{eid}/clip").status_code == 404


def test_api_password(api, monkeypatch):
    _, tmp = api
    monkeypatch.setenv("PANEL_PASSWORD", "s3cret")
    from app.api import create_app
    c = TestClient(create_app(str(tmp / "cameras.yaml")))
    assert c.get("/api/cameras").status_code == 401
    assert c.get("/api/cameras", auth=("admin", "s3cret")).status_code == 200
