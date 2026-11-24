# Tanish do'konda sinov rejasi (kamera sotib olmasdan)

Sinov bilan ikki narsani aniqlaymiz:
1. **Ulanish:** do'kon kamerasidan videoni olib bo'ladimi (bulut orqali yoki do'kon tarmog'i orqali).
2. **Aniqlik:** AI shu do'konning haqiqiy videosida odamlarni to'g'ri kuzatadimi va qancha yolg'on signal beradi.

Do'kon kompyuterga ham, sozlamalarga ham hech narsa o'zgartirilmaydi.

---

## 1-qadam. Do'konchi bilan gaplashish (telefonda, 5 daqiqa)
Quyidagilarni so'rang va menga yozing:
- [ ] Nechta kamera bor? Har biri nimani ko'rsatadi: javon, kassa, eshik?
- [ ] Kamera markasi: Hikvision, EZVIZ, Dahua yoki boshqa? Telefonida qaysi ilova bor: Hik-Connect, EZVIZ, DMSS, XMEye?
- [ ] Registrator (NVR/DVR) bormi yoki kameralar to'g'ridan-to'g'ri ilovaga ulanganmi?
- [ ] Registrator yoki kameraning admin paroli unda bormi? (Ko'pincha o'rnatgan usta biladi.)
- [ ] Har bir kameradan **bitta skrinshot**: ilovada jonli ko'rinishni ochib, ekranni rasmga oling. Shundan kamera joylashuvi tahlilga yaraydimi, bilib olaman.

**Ruxsat:** do'konchidan videoni sinov uchun yozib olishga og'zaki yoki yozma rozilik oling.
Video faqat AI'ni sozlash uchun ishlatiladi va keyin o'chiriladi.

## 2-qadam. Do'konga borish (30–40 daqiqa, noutbuk bilan)
Kerak: noutbuk (Windows bo'lsa ham bo'ladi), do'kon Wi-Fi'iga ulanish.

**a) Video yozib olish (eng muhimi).** Noutbukni do'kon Wi-Fi'iga ulang.
Kutubxonalarni o'rnating:
```bash
pip install opencv-python-headless
```
Yozishni boshlang:
```bash
python scripts/record_cameras.py --ip REGISTRATOR_IP --password PAROL --brand hikvision --channels 1 2 3 4 --minutes 20
```
- Registrator IP manzilini ilovadan (Qurilma → Tarmoq) yoki routerdagi ro'yxatdan topasiz.
- Python yo'q bo'lsa, VLC dasturida ham yozish mumkin: Media → Open Network Stream → `rtsp://admin:PAROL@IP:554/Streaming/Channels/102` → yozish tugmasi.
- Eng yaxshisi, **gavjum vaqtda** 20 daqiqa yozing.

**b) Sahnalashtirilgan sinov (5 daqiqa).** Siz yoki do'stingiz quyidagilarni bajaradi:
1. javondan narsa olib, kassada to'lab chiqadi;
2. javondan narsa olib, cho'ntakka solib, kassadan o'tmay eshikka boradi (keyin qaytib to'laydi 🙂);
3. shunchaki kirib, aylanib chiqadi.

Har birining vaqtini yozib qo'ying. Shu bilan tizim **to'g'ri signal beradimi** va **ortiqcha signal bermaydimi**, tekshiramiz.

**c) Bulut sinovi (ixtiyoriy).** EZVIZ yoki Hik-Connect bo'lsa:
```bash
python scripts/ezviz_probe.py --serial ... --code ...
```
Yo'riqnoma: `docs/0-QADAM_EZVIZ_SINOV.md`.

⚠️ Bu skript kamerani sizning dasturchi hisobingizga qo'shishga urinadi. Kamera do'konchining hisobiga bog'langan bo'lsa, xato qaytaradi va **do'konchining ilovasidan hech narsa o'chmaydi**. Shunday bo'lsa ham, bu qadamdan oldin do'konchiga aytib qo'ying.
Bulut sinovi ishlamasa ham muammo emas: hozircha a) va b) yetarli.

## 3-qadam. Menga yuboring
- Yozilgan videolar (`data/recordings/`). Hajmi katta bo'lsa, Google Drive yoki Telegram orqali yuboring.
- 1-qadamdagi javoblar va skrinshotlar.
- Sahnalashtirilgan sinov vaqtlari.

## Keyin men nima qilaman
1. Videolar ustida zonalarni chizaman va tizimni ishga tushiraman. Signallarni sizning yozuvlaringiz bilan solishtiraman.
2. Natijani aniq raqamlar bilan aytaman:
   - nechta haqiqiy holat topildi;
   - nechta yolg'on signal chiqdi;
   - kamera joylashuvi yetarlimi.
3. Kerak bo'lsa, chegaralarni moslayman: javonda turish vaqti, kassa vaqti va kiyim bo'yicha bog'lash.
4. Ulanish yo'lini tanlaymiz: bulut (EZVIZ) yoki do'kondagi kompyuter. Keyin obuna serveri va botni quraman.
