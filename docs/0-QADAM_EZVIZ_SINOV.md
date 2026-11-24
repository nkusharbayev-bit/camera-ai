# 0-qadam: kamera ilovasi bulutidan videoni olib bo'ladimi?

Bu sinov butun biznesning asosini tekshiradi. Natijasiga ko'ra serverni quramiz.

## Nima kerak
1. **Bitta EZVIZ yoki Hikvision kamera**, Hik-Connect yoki EZVIZ ilovasiga ulangan bo'lishi kerak.
   Tanishingizniki ham bo'ladi. Yangi EZVIZ kamera ~$25 turadi.
2. Kameraning **seriya raqami** (masalan, `BA1234567`) va yorlig'idagi **6 harfli kod** (Verification code).
   Ikkalasi ham kamera tagidagi yorliqda yoki ilovada: Sozlamalar → Qurilma haqida.
3. **EZVIZ dasturchi hisobi.** https://open.ezvizlife.com saytida ro'yxatdan o'ting.
   "My Application" bo'limida ilova yarating va `AppKey`, `AppSecret` ni oling.
4. Python o'rnatilgan kompyuter (Windows ham bo'ladi) yoki keyin olinadigan VPS.

## Ishga tushirish

Kompyuteringizda `camera-ai` papkasini oching va kutubxonalarni o'rnating:
```bash
pip install -r requirements.txt onnxruntime
```

Windows'da `set`, Linux/Mac'da `export` ishlating:
```bash
export EZVIZ_APP_KEY=sizning_appkey
```
```bash
export EZVIZ_APP_SECRET=sizning_appsecret
```

Sinovni boshlang (1 soat ishlaydi):
```bash
python scripts/ezviz_probe.py --serial BA1234567 --code ABCDEF --minutes 60
```

## Natija
Oxirida `=== NATIJA ===` ostida JSON chiqadi. **Shuni menga yuboring.** Undan quyidagilarni ko'ramiz:

| Ko'rsatkich | Nimani bildiradi |
|---|---|
| `result: OK` | 1 soat uzluksiz video olindi, yo'l ishlaydi |
| `drops`, `longest_gap_s` | oqim necha marta uzildi (masalan, 5 daqiqalik cheklov bormi) |
| `resolution`, `fps` | qanday sifatda kelyapti |
| `cpu_percent_of_one_core` | bitta kamera serverdan qancha CPU yeydi, shundan server narxini hisoblaymiz |
| `max_people_seen` | AI odamlarni topyaptimi |

Qaysidir qadamda ❌ chiqsa, xato matnini ham yuboring. Masalan, "device not belong to user" yoki "service not open" kabi kodlardan sababini aniqlaymiz.

**Shuningdek, EZVIZ dasturchi kabinetida quyidagilarni tekshirib, menga yozing:**
- jonli oqim (Live) xizmati pullimi;
- oylik bepul limit qancha.
