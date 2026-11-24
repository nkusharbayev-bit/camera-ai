"""Telegram bot: handles the review buttons and simple commands.

    TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... python -m app.bot

Commands: /start (shows your chat id), /hodisalar (last 10 events).
"""
from __future__ import annotations

import logging
import time

import httpx

from .config import load_config
from .storage import EventStore

log = logging.getLogger(__name__)

STATUS_LABEL = {"new": "🆕 yangi", "confirmed": "✅ tasdiqlangan", "false_alarm": "❌ yolg'on signal"}


class ReviewBot:
    def __init__(self, token: str, store: EventStore, allowed_chat_id: str | None,
                 api_base: str = "https://api.telegram.org", client: httpx.Client | None = None):
        self.base = f"{api_base}/bot{token}"
        self.store = store
        self.allowed = str(allowed_chat_id) if allowed_chat_id else None
        self.client = client or httpx.Client(timeout=40)
        self.offset = 0

    def call(self, method: str, **params) -> dict:
        r = self.client.post(f"{self.base}/{method}", json=params)
        r.raise_for_status()
        return r.json()

    def handle(self, update: dict) -> None:
        if cq := update.get("callback_query"):
            self._on_callback(cq)
        elif msg := update.get("message"):
            self._on_message(msg)

    def _authorized(self, chat_id) -> bool:
        return self.allowed is None or str(chat_id) == self.allowed

    def _on_callback(self, cq: dict) -> None:
        msg = cq.get("message", {})
        chat_id = msg.get("chat", {}).get("id")
        if not self._authorized(chat_id):
            self.call("answerCallbackQuery", callback_query_id=cq["id"], text="Ruxsat yo'q")
            return
        try:
            action, raw_id = cq.get("data", "").split(":", 1)
            event_id = int(raw_id)
        except ValueError:
            return
        status = {"confirm": "confirmed", "false": "false_alarm"}.get(action)
        if not status:
            return
        user = cq.get("from", {})
        reviewer = user.get("username") or str(user.get("id"))
        ok = self.store.review(event_id, status, reviewer)
        text = f"Hodisa #{event_id}: {STATUS_LABEL[status]}" if ok else "Hodisa topilmadi"
        self.call("answerCallbackQuery", callback_query_id=cq["id"], text=text)
        if ok and msg.get("message_id"):
            self.call("editMessageReplyMarkup", chat_id=chat_id, message_id=msg["message_id"],
                      reply_markup={"inline_keyboard": [[{"text": text, "callback_data": "noop:0"}]]})

    def _on_message(self, msg: dict) -> None:
        chat_id = msg["chat"]["id"]
        text = (msg.get("text") or "").strip()
        if text.startswith("/start"):
            self.call("sendMessage", chat_id=chat_id,
                      text=f"Camera AI boti.\nSizning chat ID: {chat_id}\n"
                           f"Uni TELEGRAM_CHAT_ID ga yozing. /hodisalar — oxirgi hodisalar.")
        elif text.startswith("/hodisalar"):
            if not self._authorized(chat_id):
                return
            rows = self.store.list(limit=10)
            if not rows:
                body = "Hozircha hodisa yo'q."
            else:
                body = "\n".join(
                    f"#{e['id']} {time.strftime('%d.%m %H:%M', time.localtime(e['created_at']))} "
                    f"{e['camera_id']} — {STATUS_LABEL.get(e['status'], e['status'])}" for e in rows)
            self.call("sendMessage", chat_id=chat_id, text=body)

    def poll_forever(self) -> None:
        log.info("Bot ishga tushdi")
        while True:
            try:
                res = self.call("getUpdates", offset=self.offset, timeout=30)
                for upd in res.get("result", []):
                    self.offset = upd["update_id"] + 1
                    try:
                        self.handle(upd)
                    except Exception:
                        log.exception("Update xatosi")
            except Exception:
                log.exception("getUpdates xatosi, 5s kutish")
                time.sleep(5)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    if not cfg.telegram_token:
        raise SystemExit("TELEGRAM_BOT_TOKEN berilmagan")
    ReviewBot(cfg.telegram_token, EventStore(cfg.db_path), cfg.telegram_chat_id).poll_forever()


if __name__ == "__main__":
    main()
