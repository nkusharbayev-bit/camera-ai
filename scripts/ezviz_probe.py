"""0-qadam: EZVIZ / Hik-Connect bulutidan kamerani o'qib bo'ladimi — tekshiruv.

Kerak:
  1. open.ezvizlife.com da dasturchi hisobi -> appKey va appSecret
  2. Sinov kamerasi: seriya raqami (masalan, BA1234567) va yorliqdagi 6 harfli kod

Ishga tushirish:
  EZVIZ_APP_KEY=... EZVIZ_APP_SECRET=... \
  python scripts/ezviz_probe.py --serial BA1234567 --code ABCDEF --minutes 60

Natija: har bir qadam OK/XATO, oqim qancha vaqt uzilmay ishladi, nechta kadr,
qancha trafik, CPU sarfi va AI qancha odam topgani. Natijani menga yuboring.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import resource
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.connectors.ezviz import DEFAULT_BASE, EzvizClient, EzvizError  # noqa: E402


def step(name, fn):
    try:
        res = fn()
        print(f"✅ {name}")
        return res
    except EzvizError as e:
        print(f"❌ {name}: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"❌ {name}: {type(e).__name__}: {e}")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", required=True)
    ap.add_argument("--code", required=True, help="kamera yorlig'idagi 6 harfli verification code")
    ap.add_argument("--channel", type=int, default=1)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--model", default="models/yolo11n-pose-416.onnx")
    args = ap.parse_args()

    c = EzvizClient(os.environ["EZVIZ_APP_KEY"], os.environ["EZVIZ_APP_SECRET"], args.base)
    report = {"serial": args.serial}

    report["token"] = bool(step("Token olish", c.token))
    print(f"   region: {c.area_url}")
    step("Qurilmani hisobga qo'shish", lambda: c.add_device(args.serial, args.code))
    info = step("Qurilma ma'lumoti", lambda: c.device_info(args.serial))
    if info:
        print("  ", json.dumps(info, ensure_ascii=False)[:300])

    addr = None
    for proto in ("hls", "rtmp", "flv"):
        addr = step(f"Jonli oqim manzili ({proto})",
                    lambda p=proto: c.live_address(args.serial, args.channel, p, verify_code=args.code))
        if addr:
            report["protocol"] = proto
            print(f"   amal qilish muddati: {time.ctime(addr.expires_at)}")
            break
    if not addr:
        report["result"] = "oqim manzili olinmadi"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    tracker = None
    if Path(args.model).exists():
        from app.onnx_detector import OnnxPoseTracker
        m = re.search(r"-(\d{3})$", Path(args.model).stem)
        tracker = OnnxPoseTracker(args.model, imgsz=int(m.group(1)) if m else 640, threads=1)

    cap = cv2.VideoCapture(addr.url, cv2.CAP_FFMPEG)
    opened = cap.isOpened()
    print(("✅" if opened else "❌") + " Oqimni ochish")
    t0 = time.time()
    end = t0 + args.minutes * 60
    frames = analysed = people_max = drops = 0
    last_analyse = 0.0
    longest_gap = 0.0
    last_ok = t0
    shape = None
    cpu0 = resource.getrusage(resource.RUSAGE_SELF)
    while opened and time.time() < end:
        ok, frame = cap.read()
        now = time.time()
        if not ok:
            drops += 1
            print(f"   ⚠️ uzilish #{drops} ({now - t0:.0f}s da), qayta ulanmoqda")
            cap.release()
            time.sleep(2)
            try:
                addr = c.live_address(args.serial, args.channel, report["protocol"], verify_code=args.code)
            except EzvizError as e:
                print(f"   ❌ manzil yangilanmadi: {e}")
            cap = cv2.VideoCapture(addr.url, cv2.CAP_FFMPEG)
            continue
        longest_gap = max(longest_gap, now - last_ok)
        last_ok = now
        frames += 1
        shape = frame.shape
        if tracker and now - last_analyse >= 0.5:  # 2 kadr/s
            last_analyse = now
            people_max = max(people_max, len(tracker.update(frame, analysed)))
            analysed += 1
        if frames % 500 == 0:
            print(f"   {now - t0:.0f}s: {frames} kadr, {analysed} tahlil, uzilishlar {drops}")
    cap.release()
    cpu1 = resource.getrusage(resource.RUSAGE_SELF)
    dur = time.time() - t0
    report.update({
        "seconds": round(dur), "frames": frames, "fps": round(frames / max(dur, 1), 1),
        "resolution": None if shape is None else f"{shape[1]}x{shape[0]}",
        "drops": drops, "longest_gap_s": round(longest_gap, 1),
        "analysed_frames": analysed, "max_people_seen": people_max,
        "cpu_percent_of_one_core": round(100 * ((cpu1.ru_utime + cpu1.ru_stime) - (cpu0.ru_utime + cpu0.ru_stime)) / max(dur, 1)),
        "result": "OK" if frames and dur > 0.9 * args.minutes * 60 else "to'liq ishlamadi",
    })
    print("\n=== NATIJA (shuni menga yuboring) ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
