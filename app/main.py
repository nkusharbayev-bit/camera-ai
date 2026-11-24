"""Run the analysis for one camera.

    python -m app.main --camera cam1                  # source from config/cameras.yaml
    python -m app.main --camera cam1 --source shop.mp4 --out data/annotated.mp4
"""
from __future__ import annotations

import argparse
import logging
import signal

import cv2

from .config import load_config
from .detector import YoloPoseTracker
from .notifier import NullNotifier, TelegramNotifier
from .pipeline import CameraPipeline
from .storage import EventStore
from .stream import open_source
from .zones import ZoneMap


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Camera AI — do'kon kamerasini tahlil qilish")
    ap.add_argument("--config", default=None)
    ap.add_argument("--camera", required=True, help="cameras.yaml dagi kamera id")
    ap.add_argument("--source", help="Video fayl yoki RTSP manzil (configdagini almashtiradi)")
    ap.add_argument("--out", help="Annotatsiyalangan videoni shu faylga yozish")
    ap.add_argument("--show", action="store_true", help="Oynada ko'rsatish (GUI kerak)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("camera-ai")

    cfg = load_config(args.config)
    cam = cfg.camera(args.camera)
    source = args.source or cam.source
    zone_map = ZoneMap.load(cam.zones_file)
    if not zone_map.zones:
        log.warning("Zonalar belgilanmagan (%s). Web paneldan zonalarni chizing.", cam.zones_file)

    if cfg.telegram_token and cfg.telegram_chat_id:
        notifier = TelegramNotifier(cfg.telegram_token, cfg.telegram_chat_id)
    else:
        log.warning("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID berilmagan — signallar faqat logga yoziladi")
        notifier = NullNotifier()

    src = open_source(source, cam.fps)
    tracker = YoloPoseTracker(cfg.model, conf=cfg.conf, device=cfg.device)
    pipe = CameraPipeline(cam.id, cam.name, tracker, zone_map, cfg.rules, EventStore(cfg.db_path),
                          notifier, cfg.data_dir, src.fps, cfg.clip_pre_seconds, cfg.clip_post_seconds)

    stopping = False

    def _stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    writer = None
    log.info("Tahlil boshlandi: %s (%s)", cam.name, "jonli" if hasattr(src, "url") else "fayl")
    for idx, ts, frame in src.frames():
        if stopping:
            break
        out = pipe.process(idx, ts, frame)
        if args.out:
            if writer is None:
                h, w = out.shape[:2]
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), src.fps, (w, h))
            writer.write(out)
        if args.show:
            cv2.imshow(cam.name, out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    if hasattr(src, "stop"):
        src.stop()
    if writer:
        writer.release()
    pipe.finish()
    log.info("Tugadi. Hodisalar soni: %d", len(pipe.event_ids))


if __name__ == "__main__":
    main()
