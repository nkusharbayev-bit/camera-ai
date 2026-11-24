import httpx
import numpy as np

from app.connectors.ezviz import EzvizClient, EzvizError
from app.demo_store import run_store_demo
from app.onnx_detector import IouTracker
from app.reid import GlobalRegistry


def test_store_demo_three_cameras(tmp_path):
    _, store, analyzer = run_store_demo(tmp_path)
    events = store.list()
    assert len(events) == 1, events
    e = events[0]
    assert e["camera_id"] == "cam_door"
    assert set(e["cameras"].split(",")) == {"cam_door", "cam_shelf", "cam_till"}
    assert e["shelf_snapshot_path"] and e["shelf_camera"] == "cam_shelf"
    # person A paid (checkout time >= 4s) and is the same global person on all three cameras
    paid = [s for s in analyzer.tracks.values() if s.checkout_seconds >= 4]
    assert len(paid) == 1 and len(paid[0].cameras) == 3 and not paid[0].alerted


def _sig(seed):
    v = np.random.default_rng(seed).random(384)
    return v / np.linalg.norm(v)


def test_registry_links_same_person_across_cameras():
    reg = GlobalRegistry(min_samples=2)
    a, b = _sig(1), _sig(2)
    for t in range(3):
        ga = reg.resolve("door", 1, a, t)
        gb = reg.resolve("door", 2, b, t)
    assert ga != gb
    for t in range(10, 13):
        g = reg.resolve("shelf", 7, a + 0.01 * _sig(3), t)
    assert g == ga
    assert (g, ga) and reg.pop_merges()  # provisional id was merged


def test_iou_tracker_keeps_ids():
    tr = IouTracker(min_hits=1)
    ids1 = tr.update(np.array([[0, 0, 10, 20], [50, 0, 60, 20]], float))
    ids2 = tr.update(np.array([[52, 0, 62, 20], [1, 0, 11, 20]], float))
    assert ids1 == [ids2[1], ids2[0]]


def test_ezviz_client_flow():
    calls = []

    def handler(req: httpx.Request):
        calls.append(req.url.path)
        if req.url.path.endswith("/token/get"):
            return httpx.Response(200, json={"code": "200", "data": {"accessToken": "T", "expireTime": 9e12,
                                                                     "areaDomain": "https://ieu.example"}})
        if req.url.path.endswith("/live/address/get"):
            assert b"deviceSerial=BA1" in req.content and b"accessToken=T" in req.content
            return httpx.Response(200, json={"code": "200", "data": {"url": "https://x/live.m3u8"}})
        return httpx.Response(200, json={"code": "20018", "msg": "device not belong to user"})

    c = EzvizClient("k", "s", client=httpx.Client(transport=httpx.MockTransport(handler)))
    addr = c.live_address("BA1", 1, "hls")
    assert addr.url.endswith(".m3u8") and c.area_url == "https://ieu.example"
    try:
        c.device_info("BA1")
        raise AssertionError("expected error")
    except EzvizError as e:
        assert e.code == "20018"
