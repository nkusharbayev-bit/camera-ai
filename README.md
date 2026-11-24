# 📷 Camera AI — do'kon uchun o'g'irlikka qarshi video tahlil (MVP)

Tizim do'kon kamerasining (yoki registratorning) RTSP oqimiga ulanadi. U har bir odamni kuzatadi va
**javondan narsa olib, kassadan o'tmay chiqib ketayotgan** odamni aniqlaganda Telegram'ga rasm
va video klip yuboradi. Egasi yoki qo'riqchi bitta tugma bilan "✅ Tasdiqlash" yoki "❌ Yolg'on signal"ni tanlaydi.

> ⚠️ Tizim hech kimni "o'g'ri" deb hukm qilmaydi. U faqat **shubhali holatni odamga ko'rsatadi**.
> Oxirgi qarorni doim odam qabul qiladi.

## Qanday ishlaydi

```
Kamera (RTSP) ─▶ YOLO pose + ByteTrack ─▶ Zonalar ─▶ Xulq-atvor qoidasi ─▶ Hodisa
                  (odam, ID, skelet)      javon/kassa/chiqish                ├─ SQLite
                                                                            ├─ rasm + 12s video
                                                                            ├─ Telegram (tugmalar bilan)
                                                                            └─ Web panel
```

Har bir odam uchun quyidagilar hisoblanadi:
- **javonda** necha soniya turgani (≥ 3s bo'lsa, "narsa olgan bo'lishi mumkin" deb hisoblanadi);
- **kassada** necha soniya turgani (≥ 4s bo'lsa, "to'lagan" deb hisoblanadi);
- **yashirish belgisi**: javondan keyin qo'li bel yoki cho'ntak yonida ≥ 1s tursa, bu belgilanadi (skelet nuqtalari orqali).

Signal qoidasi: odam javonda turgan, kassada to'xtamagan va chiqish zonasiga kirgan bo'lsa, **🟠 O'RTA** darajali signal beriladi.
Yashirish harakati ham ko'rilgan bo'lsa, daraja **🔴 YUQORI** bo'ladi.
Barcha chegaralar `config/cameras.yaml` → `rules` bo'limida sozlanadi.

## Tez boshlash

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
```

**1. Kamerasiz demo** (sun'iy video bilan; model yuklab olinmaydi):

```bash
python -m app.demo
```

Natija `data/demo/` papkasida bo'ladi: `annotated.mp4`, hodisa rasmi va klipi.
Demoda ikki odam bor: #1 to'laydi va signal bermaydi, #2 kassadan o'tmay chiqadi va signal beradi.

**2. O'z videongiz bilan** (do'kondan yozib olingan mp4):

```bash
python -m app.main --camera cam1 --source dokon.mp4 --out data/natija.mp4
```

**3. Jonli kamera bilan:** `config/cameras.yaml` faylida `source` qatoriga RTSP manzilni yozing.

| Marka | RTSP manzil namunasi |
|---|---|
| Hikvision | `rtsp://admin:PAROL@192.168.1.64:554/Streaming/Channels/101` |
| Dahua | `rtsp://admin:PAROL@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0` |
| Boshqa | Kamera qo'llanmasi yoki ONVIF Device Manager dasturi orqali topiladi |

Manzilni avval VLC'da tekshiring (Media → Open Network Stream). VLC'da ochilsa, tizim ham ulanadi.

## Telegram

1. @BotFather'da yangi bot yarating va tokenni oling.
2. Botni ishga tushiring:
   ```bash
   TELEGRAM_BOT_TOKEN=... python -m app.bot
   ```
3. Botga `/start` yozing. U sizning chat ID'ingizni aytadi. Guruhga qo'shsangiz, guruh ID'sini aytadi.
4. `.env` fayliga `TELEGRAM_BOT_TOKEN` va `TELEGRAM_CHAT_ID` ni yozing.

Bot buyruqlari: `/hodisalar` oxirgi 10 ta hodisani ko'rsatadi. Tasdiqlash tugmalarini faqat `TELEGRAM_CHAT_ID` dagi chat bosa oladi.

## Web panel (zonalarni chizish va hodisalar)

```bash
uvicorn app.api:create_app --factory --host 0.0.0.0 --port 8000
```

Brauzerda `http://localhost:8000` manzilini oching.
- **Zonalar** bo'limida kamera kadri ustiga javon, kassa va chiqish zonalarini sichqoncha bilan chizasiz.
  Zona odamning **oyog'i** turadigan pol qismini qamrashi kerak. Kadr kamera jarayoni ishga tushgandan keyin paydo bo'ladi.
- **Hodisalar** bo'limida rasm, video, tasdiqlash va filtr bor.
- `PANEL_PASSWORD` berilsa, panel login va parol so'raydi.

## Docker (do'kondagi kompyuterda doimiy ishlatish)

```bash
cp .env.example .env
```

`.env` faylini to'ldiring, keyin:

```bash
docker compose up -d --build
```

Uchta servis ishga tushadi: `cam1` (tahlil), `bot` (Telegram) va `panel` (8000-port).
Har bir qo'shimcha kamera uchun `docker-compose.yml` faylida yangi blok qo'shing.

## Uskuna tavsiyasi

| Kamera soni | Kompyuter |
|---|---|
| 1–2 | Oddiy mini-PC (Intel i5, 8 GB). CPU'da 5–10 kadr/s, `yolo11n-pose.pt` |
| 4–8 | NVIDIA GPU'li kompyuter (RTX 3060) yoki Jetson Orin, `yolo11s-pose.pt` |

## Kamerani joylashtirish (aniqlik uchun eng muhim omil)

- Javon, kassa va chiqish iloji boricha **bitta kamerada** ko'rinsin. Kamera yuqoridan va qiyalab qarasin.
  MVP odamni bir kameradan ikkinchisiga o'tganda "tanimaydi".
- Odamning oyog'i ko'rinishi kerak, chunki zona tekshiruvi oyoq nuqtasi bo'yicha qilinadi.
- Yorug'lik yetarli bo'lsin. Asosiy oqimdan (`101`, `subtype=0`) foydalaning.

## Cheklovlar (halol)

- **Yolg'on signallar bo'ladi.** Masalan, narsani ko'rib qaytarib qo'ygan xaridor ham "javonda turgan" hisoblanadi.
  Boshlang'ich bosqichda chegaralarni moslang yoki `require_concealment: true` qiling.
- Gavjum joyda bir odamning ID'si boshqasiga o'tib ketishi mumkin.
- "Yashirish" belgisi oddiy evristika. U sumka, ko'ylak ichi va boshqa holatlarda xato qilishi mumkin.
- Yuzni tanib olish **qasddan qo'shilmagan**. O'zbekistonning "Shaxsga doir ma'lumotlar to'g'risida"gi qonuni
  bo'yicha bunday tizimlar uchun alohida talablar bor. Do'konga **"Videokuzatuv olib boriladi"** yozuvini osing.

## Keyingi bosqichlar

1. Kassa (POS) integratsiyasi: chek urilgan vaqt bilan solishtirish. Bu yolg'on signallarni keskin kamaytiradi.
2. Tasdiqlangan va yolg'on signallardan o'z modelini o'qitish (fine-tuning).
3. Ko'p kamera va odamni kameralararo kuzatish (Re-ID).
4. Ko'p do'kon (SaaS), statistika: tashriflar soni va gavjum soatlar.

## Loyiha tuzilmasi

```
app/zones.py      zonalar (poligonlar)
app/detector.py   YOLO pose + ByteTrack; ScriptedTracker (test uchun)
app/behavior.py   xulq-atvor holat mashinasi va signal qoidasi
app/pipeline.py   bitta kamera uchun hamma narsani bog'laydi
app/recorder.py   hodisa atrofidagi video klipni yozish
app/storage.py    SQLite
app/notifier.py   Telegram'ga yuborish
app/bot.py        Telegram tugmalari va buyruqlari
app/api.py        web panel API; web/index.html — panel
app/main.py       ishga tushirish (bitta kamera)
app/demo.py       kamerasiz demo
tests/            avtomatik testlar (pytest)
```
