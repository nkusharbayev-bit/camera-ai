"""Telegram notifications via the Bot HTTP API.

Each alert is sent as a photo (immediately) and then the evidence video clip,
with inline buttons so the shop owner can mark it confirmed / false alarm.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time

import httpx

log = logging.getLogger(__name__)

SEVERITY_LABEL = {"high": "🔴 YUQORI", "medium": "🟠 O'RTA"}


def review_keyboard(event_id: int) -> dict:
    return {"inline_keyboard": [[
        {"text": "✅ Tasdiqlash", "callback_data": f"confirm:{event_id}"},
        {"text": "❌ Yolg'on signal", "callback_data": f"false:{event_id}"},
    ]]}


def caption_for(event: dict, camera_name: str) -> str:
    when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event["created_at"]))
    return (f"⚠️ Shubhali holat #{event['id']}\n"
            f"Daraja: {SEVERITY_LABEL.get(event['severity'], event['severity'])}\n"
            f"Kamera: {camera_name}\n"
            f"Vaqt: {when}\n"
            f"Sabab: {event['reason']}\n"
            f"Javonda: {event['shelf_seconds']}s · Kassada: {event['checkout_seconds']}s\n\n"
            f"Iltimos, videoni ko'rib tasdiqlang.")


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, api_base: str = "https://api.telegram.org",
                 client: httpx.Client | None = None):
        self.base = f"{api_base}/bot{token}"
        self.chat_id = chat_id
        self.client = client or httpx.Client(timeout=60)
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # Public API: non-blocking, safe to call from the video loop.
    def send_photo(self, event: dict, camera_name: str) -> None:
        self._q.put(("photo", event, camera_name))

    def send_clip(self, event: dict, camera_name: str) -> None:
        self._q.put(("clip", event, camera_name))

    def wait(self, timeout: float = 30) -> None:
        end = time.time() + timeout
        while not self._q.empty() or self._q.unfinished_tasks:
            if time.time() > end:
                break
            time.sleep(0.05)

    def _worker(self) -> None:
        while True:
            kind, event, cam = self._q.get()
            try:
                if kind == "photo" and event.get("snapshot_path"):
                    self._post("sendPhoto", event["snapshot_path"], "photo",
                               caption=caption_for(event, cam),
                               reply_markup=json.dumps(review_keyboard(event["id"])))
                elif kind == "clip" and event.get("clip_path"):
                    self._post("sendVideo", event["clip_path"], "video",
                               caption=f"🎥 Hodisa #{event['id']} videosi")
            except Exception:
                log.exception("Telegramga yuborishda xato")
            finally:
                self._q.task_done()

    def _post(self, method: str, file_path: str, field: str, **data) -> dict:
        for attempt in range(3):
            with open(file_path, "rb") as fh:
                r = self.client.post(f"{self.base}/{method}", data={"chat_id": self.chat_id, **data},
                                     files={field: fh})
            if r.status_code == 200:
                return r.json()
            log.warning("Telegram %s xato %s: %s", method, r.status_code, r.text[:200])
            time.sleep(2 ** attempt)
        raise RuntimeError(f"Telegram {method} failed")


class NullNotifier:
    """Used when Telegram is not configured: just logs."""

    def send_photo(self, event: dict, camera_name: str) -> None:
        log.warning("SIGNAL #%s [%s] %s", event["id"], event["severity"], event["reason"])

    def send_clip(self, event: dict, camera_name: str) -> None:
        log.info("Klip tayyor: %s", event.get("clip_path"))

    def wait(self, timeout: float = 0) -> None:
        pass
