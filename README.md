# Canlı Altyazı + Ses Analizi (Türkçe, Vietnamca)

Türkçe ve Vietnamca konuşmayı metne çevirir; iki şekilde:

- **Canlı altyazı** — konuşurken sözler ekranda (laptop ya da projeksiyon) anında belirir.
  Konferans, sunum, ders için. Oturum bitince kayıt otomatik olarak analize gider.
- **Ses analizi** — bir ses dosyası yüklenir; kimin ne zaman ne söylediği satır satır çıkar
  (konuşmacı ayrımı). Başka sistemlere API ve imzalı webhook ile bağlanır.

İkisi **aynı modelleri ve aynı çekirdeği** kullanır; tamamen **çevrimdışı** çalışır.
Dil seçilir ya da sesten bulunur.

| Dil | Model | Not |
|---|---|---|
| Türkçe | faster-whisper medium (MIT) | Kelime zamanlı; GPU varsa GPU. Hızlı seçenekler: small, Omnilingual-300M |
| Vietnamca | Zipformer-68M int8 (Apache-2.0) | Yalnızca CPU yeterli; gerçek zamanın ~40 katı hızlı. VietMed'de WER %10,1 |
Bu repo üç projenin birleşimidir: Live-stream (canlı akış), Vietnamese-speech-to-text-models
(ölçülmüş toplu iş çekirdeği) ve fine-tuning (ilk Türkçe sistem). Neyin nereden alındığı:
[ADR 0007](docs/adr/0007-birlesik-sistem.md).

## Hızlı başlangıç (Windows)

```bat
baslat.bat
```

İlk çalıştırmada modeller iner (~2,1 GB; internet bir kez gerekir), sonra sunucu açılır.
Yalnızca bir dil isteniyorsa `.env` içinde `LANGUAGES=tr` (ya da `vi`): o dilin modeli iner,
dil tespiti modeli yüklenmez.

| Adres | Ne |
|---|---|
| http://127.0.0.1:8000/canli | Canlı altyazı |
| http://127.0.0.1:8000 | Dosya yükle → konuşmacılı transkript, geçmiş, düzeltme |
| http://127.0.0.1:8000/docs | API belgesi |

Redis, PostgreSQL ya da Docker gerekmez (SQLite + iş parçacıkları). Durdurmak: `durdur.bat`.

Linux / WSL / elle:

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[server]"
python scripts/download_models.py
python -m live_stream                                   # ya da: live-stream
```

### NVIDIA GPU (canlı altyazı için önerilen)

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

`WHISPER_DEVICE=auto` (varsayılan) GPU'yu kendisi bulur; CUDA açılamazsa uyarı verip CPU ile
devam eder. Hangi cihazın kullanıldığı canlı ekranın sağ üstünde görünür (`medium · GPU`).
Windows'ta DLL yolları uygulama tarafından ayarlanır.

## Canlı altyazı

```
Mikrofon ─► Tarayıcı (AudioWorklet, 16 kHz, 100 ms paketler) ─► WebSocket /ws/live
                                                                     │
       Silero VAD ─► cümle sonu (700 ms duraklama) ─┬─► ara metin   (~1,2 sn'de bir)
                                                    └─► kesin metin ─► ekran + disk
          dil: seçilen (tr | vi) ya da "otomatik": ilk cümleden bulunur, oturum boyunca sabit
                                                                     │
                                         oturum bitince .wav ─► toplu iş: konuşmacı ayrımlı transkript
```

| Tuş | İşlev |
|---|---|
| `F` | Sunum modu (tam ekran, büyük punto, son satırlar) |
| `Esc` | Sunum modundan çık |
| `+` / `-` | Yazıyı büyüt / küçült |

- **Dil:** sayfadaki listeden Türkçe, Vietnamca ya da Otomatik. Otomatikte ilk cümleden sonra
  bulunan dil ve güveni ekranın sağ üstünde görünür. Türkçe: Whisper (ara metin beam 1, kesin
  metin beam 5 + önceki cümleler, büyük harf ve noktalamalı). Vietnamca: Zipformer (küçük harf,
  noktalamasız; çok hızlı olduğu için GPU gerekmez).
- **Dosyayla test:** bir kaydı gerçek zaman hızında çalar; konferanstan önce evde prova için.
- **Kayıtlar:** her oturum `recordings/` altına anında yazılır (çökmeye dayanıklı):
  `.wav` (ham ses), `.txt` (`[00:01:23] metin`), `.jsonl`.
- **Konuşmacılı transkript:** oturum bitince ekranda "Konuşmacılı transkript ↗" bağlantısı çıkar;
  soru-cevap bölümü dahil kimin ne söylediği ana sayfada açılır, satır satır düzeltilebilir.
  Kapatmak: ayarlarda kutucuk ya da `LIVE_AUTO_ANALYZE=false`.
- **Oturum sonu bildirimi (webhook):** `LIVE_WEBHOOK_URL` tanımlıysa konuşma bitince
  (Durdur ya da bağlantı kopması) o adrese önce `live.ended` (canlı metin, dil, süre), konuşmacı
  ayrımı bitince `job.done` (kimin ne söylediği) gider; imzalı ve tekrar denemeli.
  Oturum başına da verilebilir: `/ws/live?callback_url=...`. Ayrıntı:
  [docs/entegrasyon.md](docs/entegrasyon.md#41-canlı-oturum-bitti-liveended).
- **Bağlantı koparsa** ses kesilmeden yeniden bağlanır; ekran uykusu engellenir.

**Konferans ipuçları**

- Laptop mikrofonu salon yankısını da alır; yaka mikrofonu ya da ses masası çıkışı kaliteyi çok
  artırır (ekrandaki listeden seçilir).
- Sunumdaki özel terimleri ve adları `HOTWORDS`'e yazın (ör. `YOLOv8 TÜBİTAK Bursa Teknik`);
  Vietnamca için `VI_HOTWORDS` bir terim dosyası (BÜYÜK HARF, ör. `hotwords/tibbi_vi.txt`).
- Cümleler ortadan bölünüyorsa `END_SILENCE_MS`'i artırın (900–1000), metin geç geliyorsa azaltın.
- Ekrandaki **gecikme** 2 sn'yi aşıyorsa `LIVE_MODEL=faster-whisper-small` (indirmek:
  `python scripts/download_models.py --tr faster-whisper-small`). Toplu iş medium'da kalır.

## Ses analizi (toplu iş)

```
ses ─► 16 kHz mono ─► VAD (Silero) ─► dil ─► ASR (tr: faster-whisper, vi: Zipformer) ─► kelimeler
                         │                                                              │
                         └─► konuşmacı ayrımı (CAM++ imzası + spektral kümeleme) ───────┤
                                                                                        ▼
                                                    birleştirme ─► [0.1 - 3.7] speaker_00: ...
```

- Dil istekte verilir (`tr`, `vi`) ya da sesten bulunur (Whisper-tiny, emin olamazsa -small);
  desteklenmeyen dildeki ses reddedilir (`rejected`), yanlış modelle anlamsız metin üretilmez.
- Konuşmacı sayısı istekte verilmezse sesten bulunur.
- Whisper'ın sessizlikte uydurduğu kalıplar ("Altyazı M.K.", "İzlediğiniz için teşekkürler")
  ve tekrar döngüleri atılır; başka alfabeden uydurma kelimeler süzülür.
- Yazıya dökülmemiş konuşma aralıkları yeniden çözülür (güvenlik ağı); kapsama oranı sonuçta.
- Düşük güvenli kelimeler işaretlenir; arayüzde her satır düzeltilebilir. Düzeltmeler ince
  ayar verisine dönüşür (onaylı, test/eğitim ayrımlı).
- Hedef: **1 dakikalık ses en geç 1 dakikada biter** (CPU). Kendi bilgisayarında ölç:
  `python benchmarks/kabul_testi.py kayit.wav`.

```bash
curl -F file=@kayit.wav http://localhost:8000/jobs                          # dil sesten bulunur
curl -F file=@kayit.wav -F language=vi http://localhost:8000/jobs           # Vietnamca
curl -F file=@kayit.wav -F num_speakers=2 -F callback_url=https://ornek.com/hook http://localhost:8000/jobs
curl http://localhost:8000/jobs/<job_id>                                     # durum + sonuç
python -m live_stream.interface.cli kayit.wav --lang tr --speakers 2       # API'siz, tek dosya
```

Bağlanacak sistem için alanlar, durumlar, webhook imzası ve tekrar kuralları:
[docs/entegrasyon.md](docs/entegrasyon.md).

### Tam mod (sunucu, çok sayıda mesaj)

Çok sayıda eşzamanlı mesaj (ör. 4000 sesli geri bildirim, yalnızca CPU) için:
API + Redis Streams kuyruğu + PostgreSQL + ayrı işçiler.

```bash
docker compose up -d --build          # ya da Windows'ta: baslat_tam.bat
```

Canlı altyazı bu modda yoktur; o, konuşmacının kendi bilgisayarında yerel modda çalışır.

## Ayarlar

`.env.example` → `.env` (baslat.bat ilk çalıştırmada kendisi kopyalar). En önemlileri:

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `LANGUAGES` | `tr,vi` | Açık diller; tek dil yazılırsa tespit yapılmaz |
| `TR_MODEL` | `faster-whisper-medium` | `faster-whisper-small` daha hızlı; `omnilingual-300m` en hızlı |
| `VI_MODEL` | `zipformer-68m` | ya da ince ayarlı modelin klasörü (`finetune/`) |
| `LIVE_LANGUAGE` | `tr` | Canlı sayfanın varsayılan dili: `tr`, `vi`, `auto` |
| `WHISPER_DEVICE` | `auto` | `auto`, `cuda`, `cpu` (sunucuda `cpu`) |
| `HOTWORDS` / `VI_HOTWORDS` | — | Türkçe terimler (boşlukla) / Vietnamca terim dosyası |
| `LIVE_MODEL` | (TR_MODEL) | Türkçe canlı altyazının modeli; farklıysa ayrıca yüklenir |
| `END_SILENCE_MS` | `700` | Cümle bitti demek için duraklama |
| `PARTIAL_INTERVAL_SEC` | `1.2` | Ara metin sıklığı; yavaş bilgisayarda artırın ya da `0` |
| `LIVE_AUTO_ANALYZE` | `true` | Oturum bitince konuşmacı ayrımlı transkript |
| `LIVE_WEBHOOK_URL` | — | Oturum bitince bildirim adresi (`live.ended`, sonra `job.done`) |
| `WEBHOOK_SECRET` | — | Bildirim imzası (HMAC-SHA256) |
| `NUM_SPEAKERS` | `0` | 0 = sesten bul |
| `CONCURRENCY` | `auto` | Aynı anda işlenen dosya; çekirdek ve belleğe göre |

| Donanım | Önerilen |
|---|---|
| NVIDIA GPU 4 GB (ör. RTX 3050 Ti) | Toplu iş `medium`; canlı gecikme yüksekse `LIVE_MODEL=faster-whisper-small` |
| Yalnızca CPU, 6+ çekirdek | `small`, `PARTIAL_INTERVAL_SEC=2` |
| Zayıf CPU | `omnilingual-300m` (noktalamasız, küçük harf) |

## Proje yapısı

```
src/live_stream/
  domain/          veri tipleri (dış kütüphane yok)
  application/     iş kuralları ve portlar
    transcribe.py    toplu iş: dil → ASR → konuşmacı → birleştirme
    live.py          canlı: cümle sonu tespiti (Segmenter), oturum (LiveSession), dil kilidi
    jobs.py, fusion.py, coverage.py, language.py, maintenance.py, config.py, ports.py
  infrastructure/  modeller ve depolama
    faster_whisper_asr.py   Türkçe ASR (toplu + canlı, GPU→CPU yedeği)
    sherpa_asr.py           Vietnamca Zipformer (+ Türkçe yedekleri Omnilingual, Whisper-turbo)
    whisper_lid.py          dil tespiti (Whisper-tiny → small)
    sherpa_vad.py, live_vad.py, cluster_diarizer.py, recorder.py,
    db/ (SQLAlchemy + Alembic), redis_streams.py, thread_queue.py, webhook.py, catalog.py
  interface/       local_server.py (yerel: toplu + canlı), live_server.py, api.py (tam mod),
                   stream_worker.py, dispatcher.py, bakim.py, cli.py, static/ (index, canli)
finetune/          Vietnamca Zipformer ince ayar hattı (WSL2 + GPU) ve Kapı 2
hotwords/          Vietnamca terim listesi örneği
benchmarks/ scripts/ tests/{unit,db,integration} docs/{adr, olcumler.md, entegrasyon.md}
```

Katman kuralı (`lint-imports`): içteki katman dıştakini bilmez; `application` hiçbir model
kütüphanesini import etmez. Canlı akış ve toplu iş aynı portlardan geçer.

## Geliştirme

```bash
pip install -e ".[dev,server]"
pytest                                  # birim + depo testleri (model gerekmez)
pytest -m models                        # gerçek modellerle (models/ içinde ne varsa)
lint-imports                            # katman kuralları
python benchmarks/kabul_testi.py a.wav b.wav --es-zamanli 4
```

Diğer ölçümler: `benchmarks/diyalog_testi.py` (DER, cpWER), `yerel_dogruluk_testi.py` (WER),
`konusmaci_testi.py`, `turkce_dogruluk_testi.py`, `dil_tespiti_testi.py`, `dogruluk_testi.py`
(Vietnamca Zipformer). VietMed'den test seti: `scripts/vietmed_hazirla.py`. Sonuçlar: [docs/olcumler.md](docs/olcumler.md).

## Yol haritası

- [x] Toplu iş çekirdeği (Vietnamese-speech-to-text-models'ten)
- [x] Canlı altyazı çekirdeğe katıldı: aynı model, ONNX Silero (PyTorch'suz), oturum sonu toplu iş
- [x] Vietnamca: toplu iş (Zipformer + dil tespiti) ve canlı altyazı (dil seçimi, otomatik dil)
- [x] Konuşma bitince webhook: `live.ended` + konuşmacılı `job.done`, aynı outbox'tan
- [ ] Canlı ekranda konuşmacı etiketi (cümle başına CAM++ imzası, oturum sonunda kesinleşir)
- [ ] Türkçe doğru metinli ölçüm: FLEURS tr + kendi sunum provası (WER, DER; canlı ve toplu)
- [ ] Canlı mod için model karşılaştırması: medium / small / Omnilingual, gecikme × WER
- [ ] Vietnamca ince ayarın çalıştırılması ve Kapı 2 kararı (`finetune/`)

## Lisans

MIT
