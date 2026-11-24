import numpy as np

from app.behavior import BehaviorAnalyzer, Rules, hand_at_hip
from app.detector import Person
from app.zones import Zone, ZoneMap

ZONES = ZoneMap([
    Zone("shelf", "shelf", [(0, 0), (100, 0), (100, 100), (0, 100)]),
    Zone("kassa", "checkout", [(200, 0), (300, 0), (300, 100), (200, 100)]),
    Zone("exit", "exit", [(400, 0), (500, 0), (500, 100), (400, 100)]),
])


def at(x, tid=1, kp=None):
    # foot point = bottom-centre -> (x, 50)
    return Person(tid, (x - 10, 0, x + 10, 50), keypoints=kp)


def walk(analyzer, plan, t0=0.0, dt=0.1, tid=1, kp=None):
    """plan: list of (x, seconds). Returns all alerts and the end time."""
    alerts, t = [], t0
    for x, secs in plan:
        for _ in range(int(round(secs / dt))):
            alerts += analyzer.update([at(x, tid, kp)], t)
            t += dt
    return alerts, t


def test_zone_contains():
    z = ZONES.zones[0]
    assert z.contains(50, 50) and not z.contains(150, 50)
    assert ZONES.kinds_at(250, 50) == {"checkout"}


def test_paid_customer_no_alert():
    a = BehaviorAnalyzer(ZONES)
    alerts, _ = walk(a, [(150, 1), (50, 5), (150, 1), (250, 6), (350, 1), (450, 2)])
    assert alerts == []


def test_skipped_checkout_alerts_once():
    a = BehaviorAnalyzer(ZONES)
    alerts, _ = walk(a, [(50, 5), (350, 1), (450, 3)])
    assert len(alerts) == 1
    assert alerts[0].severity == "medium" and alerts[0].track_id == 1


def test_walking_through_checkout_quickly_still_alerts():
    a = BehaviorAnalyzer(ZONES)
    alerts, _ = walk(a, [(50, 5), (250, 1), (450, 2)])
    assert len(alerts) == 1


def test_entering_through_exit_no_alert():
    a = BehaviorAnalyzer(ZONES)
    alerts, _ = walk(a, [(450, 3), (350, 2), (450, 3)])
    assert alerts == []


def test_brief_shelf_glance_no_alert():
    a = BehaviorAnalyzer(ZONES)
    alerts, _ = walk(a, [(50, 1), (450, 3)])
    assert alerts == []


def _kp(conceal: bool):
    kp = np.zeros((17, 3))
    kp[:, 2] = 0.9
    kp[11] = (0, 30, 0.9)
    kp[12] = (10, 30, 0.9)
    kp[9] = (1, 31, 0.9) if conceal else (0, 5, 0.9)
    kp[10] = (40, 5, 0.9)
    return kp


def test_hand_at_hip():
    assert hand_at_hip(_kp(True), (0, 0, 20, 50))
    assert not hand_at_hip(_kp(False), (0, 0, 20, 50))
    assert not hand_at_hip(None, (0, 0, 20, 50))


def test_concealment_gives_high_severity():
    a = BehaviorAnalyzer(ZONES)
    alerts, _ = walk(a, [(50, 5), (350, 2), (450, 2)], kp=_kp(True))
    assert len(alerts) == 1 and alerts[0].severity == "high" and alerts[0].concealed


def test_require_concealment():
    a = BehaviorAnalyzer(ZONES, Rules(require_concealment=True))
    alerts, _ = walk(a, [(50, 5), (450, 3)], kp=_kp(False))
    assert alerts == []


def test_tracks_expire_and_gaps_ignored():
    a = BehaviorAnalyzer(ZONES, Rules(track_ttl_seconds=2))
    a.update([at(50)], 0.0)
    a.update([at(50)], 100.0)  # huge gap -> not counted as 100s at shelf
    assert a.tracks[1].shelf_seconds == 0.0
    a.update([], 200.0)
    assert 1 not in a.tracks
