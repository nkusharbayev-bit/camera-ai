"""SQLite storage for suspicious events."""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    track_id INTEGER,
    created_at REAL NOT NULL,
    video_ts REAL,
    severity TEXT,
    reason TEXT,
    shelf_seconds REAL,
    checkout_seconds REAL,
    concealed INTEGER,
    snapshot_path TEXT,
    clip_path TEXT,
    status TEXT NOT NULL DEFAULT 'new',   -- new | confirmed | false_alarm
    reviewed_by TEXT,
    reviewed_at REAL,
    shelf_snapshot_path TEXT,
    shelf_camera TEXT,
    cameras TEXT
);
"""
MIGRATIONS = ["ALTER TABLE events ADD COLUMN shelf_snapshot_path TEXT",
              "ALTER TABLE events ADD COLUMN shelf_camera TEXT",
              "ALTER TABLE events ADD COLUMN cameras TEXT"]

STATUSES = ("new", "confirmed", "false_alarm")


class EventStore:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(events)")}
            for m in MIGRATIONS:
                if m.split()[5] not in cols:
                    self._conn.execute(m)
            self._conn.commit()

    def add(self, camera_id: str, alert, snapshot_path: str | None = None) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO events (camera_id, track_id, created_at, video_ts, severity, reason, shelf_seconds,"
                " checkout_seconds, concealed, snapshot_path, cameras) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (camera_id, alert.track_id, time.time(), alert.timestamp, alert.severity, alert.reason,
                 alert.shelf_seconds, alert.checkout_seconds, int(alert.concealed), snapshot_path,
                 ",".join(getattr(alert, "cameras", None) or [camera_id])),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def set_snapshot(self, event_id: int, path: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE events SET snapshot_path=? WHERE id=?", (path, event_id))
            self._conn.commit()

    def set_shelf_snapshot(self, event_id: int, path: str, camera_name: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE events SET shelf_snapshot_path=?, shelf_camera=? WHERE id=?",
                               (path, camera_name, event_id))
            self._conn.commit()

    def set_clip(self, event_id: int, clip_path: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE events SET clip_path=? WHERE id=?", (clip_path, event_id))
            self._conn.commit()

    def review(self, event_id: int, status: str, reviewer: str | None = None) -> bool:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        with self._lock:
            cur = self._conn.execute("UPDATE events SET status=?, reviewed_by=?, reviewed_at=? WHERE id=?",
                                     (status, reviewer, time.time(), event_id))
            self._conn.commit()
            return cur.rowcount > 0

    def get(self, event_id: int) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        return dict(row) if row else None

    def list(self, limit: int = 100, camera_id: str | None = None, status: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM events WHERE 1=1", []
        if camera_id:
            q += " AND camera_id=?"
            args.append(camera_id)
        if status:
            q += " AND status=?"
            args.append(status)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            return [dict(r) for r in self._conn.execute(q, args).fetchall()]
