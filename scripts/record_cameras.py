"""Do'kon kameralaridan sinov uchun video yozib olish (mahalliy tarmoqda, RTSP orqali).

    python scripts/record_cameras.py --ip 192.168.1.64 --user admin --password PAROL --brand hikvision \
        --channels 1 2 3 4 --minutes 20

Har bir kanal uchun data/recordings/kanal_N.mp4 yoziladi (past sifatli oqim, kam joy oladi).
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from urllib.parse import quote

import cv2

URLS = {
    "hikvision": "rtsp://{u}:{p}@{ip}:554/Streaming/Channels/{ch}02",          # 02 = sub stream
    "dahua": "rtsp://{u}:{p}@{ip}:554/cam/realmonitor?channel={ch}&subtype=1",
}


def record(url: str, out: Path, seconds: float, results: dict, ch: int) -> None:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        results[ch] = "❌ ulanmadi (IP, parol yoki kanal raqamini tekshiring)"
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    if fps > 60:
        fps = 15
    writer, frames, end = None, 0, time.time() + seconds
    while time.time() < end:
        ok, frame = cap.read()
        if not ok:
            break
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(frame)
        frames += 1
    cap.release()
    if writer:
        writer.release()
    results[ch] = f"✅ {frames} kadr -> {out}" if frames else "❌ kadr kelmadi"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True, help="registrator (NVR) yoki kamera IP manzili")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", required=True)
    ap.add_argument("--brand", choices=URLS, default="hikvision")
    ap.add_argument("--channels", type=int, nargs="+", default=[1])
    ap.add_argument("--minutes", type=float, default=20)
    args = ap.parse_args()

    out_dir = Path("data/recordings")
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    threads = []
    for ch in args.channels:
        url = URLS[args.brand].format(u=quote(args.user, safe=""), p=quote(args.password, safe=""), ip=args.ip, ch=ch)
        t = threading.Thread(target=record, args=(url, out_dir / f"kanal_{ch}.mp4", args.minutes * 60, results, ch))
        t.start()
        threads.append(t)
    print(f"{len(threads)} ta kanal {args.minutes} daqiqa yozilmoqda...")
    for t in threads:
        t.join()
    for ch in sorted(results):
        print(f"Kanal {ch}: {results[ch]}")


if __name__ == "__main__":
    main()
