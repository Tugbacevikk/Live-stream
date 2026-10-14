# Canlı Altyazı (Live-stream)

Konuşurken sesi **anında metne dönüştüren**, tamamen **çevrimdışı** çalışan canlı altyazı uygulaması.
Konferans, sunum ve ders gibi ortamlarda konuşmacının sözlerini ekranda (laptop veya projeksiyon) gerçek zamanlı gösterir.

[Ses Analizi Sistemi](https://github.com/Tugbacevikk/fine-tuning) projesinin canlı akış kardeşidir: aynı Faster-Whisper altyapısını kullanır,
konferans sonrası kaydedilen ses dosyası konuşmacı ayrımı için o projeye verilebilir.

## Özellikler

- **Gerçek zamanlı metin:** konuşurken gri ara metin, cümle bitince kesinleşmiş metin
- **Akıllı cümle sonu tespiti:** Silero VAD ile duraklamalar algılanır; alkış/öksürük gibi kısa sesler elenir
- **GPU hızlandırma:** NVIDIA GPU varsa otomatik kullanılır, CUDA sorunu olursa çökmeden CPU'ya geçer
- **Sunum modu:** tam ekran, büyük puntolu, yalnızca son 3 satır (projeksiyona uygun)
- **Çökme güvenliği:** ses (`.wav`) ve metin (`.txt`, `.jsonl`) anında diske yazılır
- **Otomatik yeniden bağlanma:** bağlantı koparsa ses kesilmeden tekrar bağlanır
- **Ekran uykusu engelleme:** konuşma sırasında laptop ekranı kararmaz
- **Uydurma (halüsinasyon) filtresi:** Whisper'ın sessizlikte ürettiği "Altyazı M.K." gibi kalıplar gösterilmez
- **Dosyayla test:** bir kaydı gerçek zaman hızında çalarak, konferans öncesi evde deneme

## Nasıl çalışır?

```
Mikrofon ─► Tarayıcı (AudioWorklet, 16 kHz PCM, 100 ms paketler)
              │  WebSocket
              ▼
          Sunucu ─► Silero VAD ─► Cümle sonu tespiti ─┬─► Ara metin   (beam=1, hızlı)
                                                      └─► Kesin metin (beam=5, doğru) ─► Ekran + disk
```

1. Tarayıcı mikrofon sesini 16 kHz'e çevirip her 100 ms'de sunucuya yollar.
2. Sunucu her 32 ms'lik pencerede "konuşma var mı?" diye bakar.
3. Konuşma sürerken her ~1,2 sn'de bir **ara metin** üretilir.
4. ~0,7 sn duraklama olunca cümle biter, **kesin metin** üretilir (önceki cümleler bağlam olarak verilir).

## Kurulum

Python 3.10+ gerekir.

```bash
git clone https://github.com/Tugbacevikk/live-stream.git
cd live-stream
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/WSL: source .venv/bin/activate
pip install -e .
```

### NVIDIA GPU ile (önerilen)

```bash
pip install -e ".[gpu]"
```

Bu komut CUDA 12 için cuBLAS ve cuDNN 9 kütüphanelerini kurar; Windows'ta DLL yolları uygulama tarafından otomatik ayarlanır.
GPU açılamazsa uygulama uyarı verip CPU ile devam eder. Hangi cihazın kullanıldığı ekranın sağ üstünde görünür (`small · GPU`).

## Çalıştırma

```bash
python -m live_stream
```

Windows'ta `baslat.bat` dosyasına çift tıklamak da yeterli.
Ardından tarayıcıda **http://localhost:8000** adresini açın ve **Başlat**'a basın.

> İlk çalıştırmada Whisper modeli indirilir (`small` ≈ 470 MB). Konferanstan **önce** internetli bir yerde bir kez çalıştırın;
> sonrasında model önbellekten yüklenir ve internet gerekmez.

### Kısayollar

| Tuş | İşlev |
|-----|-------|
| `F` | Sunum modu (tam ekran) aç/kapat |
| `Esc` | Sunum modundan çık |
| `+` / `-` | Yazıyı büyüt / küçült |

## Ayarlar

`.env.example` dosyasını `.env` olarak kopyalayıp düzenleyin. En önemlileri:

| Değişken | Varsayılan | Açıklama |
|----------|-----------|----------|
| `WHISPER_MODEL` | `small` | `tiny`, `base`, `small`, `medium`. Büyüdükçe doğruluk artar, gecikme artar |
| `WHISPER_DEVICE` | `auto` | `auto`, `cuda`, `cpu` |
| `WHISPER_LANGUAGE` | `tr` | Konuşma dili |
| `WHISPER_HOTWORDS` | — | Sunumdaki özel terimler, isimler (ör. `YOLOv8 TÜBİTAK Bursa Teknik`) |
| `END_SILENCE_MS` | `700` | Cümle bitti demek için gereken duraklama |
| `PARTIAL_INTERVAL_SEC` | `1.2` | Ara metin sıklığı; yavaş bilgisayarda artırın veya `0` yapın |
| `SAVE_RECORDINGS` | `true` | Ses ve metni `recordings/` klasörüne kaydet |

Cümle sonu, ara metin sıklığı ve kayıt ayarları ekrandaki **Ayarlar** bölümünden de değiştirilebilir.

### Hangi model?

| Donanım | Önerilen |
|---------|----------|
| NVIDIA GPU (4 GB+) | `small`, isterseniz `medium` deneyin |
| Sadece CPU (6+ çekirdek) | `base` veya `small` + `PARTIAL_INTERVAL_SEC=2` |
| Zayıf CPU | `tiny` |

Ekrandaki **gecikme** göstergesi 2 saniyenin üstüne çıkıyorsa daha küçük bir model seçin.

## Konferans için ipuçları

- **Mikrofon:** laptop mikrofonu salon yankısını da alır. Yaka mikrofonu veya ses masasından çıkış kaliteyi çok artırır; ekrandaki listeden seçebilirsiniz.
- **Terimler:** sunumdaki teknik kelimeleri `WHISPER_HOTWORDS`'e yazın.
- **Prova:** sunum provanızı kaydedip **Dosyayla test** ile çalarak ayarları önceden deneyin.
- **Duraklama ayarı:** cümleleriniz ortadan bölünüyorsa `END_SILENCE_MS`'i artırın (900–1000), metin geç geliyorsa azaltın (500–600).

## Kayıtlar

Her oturum `recordings/` klasörüne yazılır:

```
recordings/2026-10-14_10-35-00.wav     # ham ses (16 kHz mono)
recordings/2026-10-14_10-35-00.txt     # [00:01:23] zaman damgalı metin
recordings/2026-10-14_10-35-00.jsonl   # makine okunur sonuçlar
```

Konferans sonrası `.wav` dosyasını [Ses Analizi Sistemi](https://github.com/Tugbacevikk/fine-tuning)'ne yükleyerek
konuşmacı ayrımlı (soru-cevap bölümü dahil) daha doğru bir transkript elde edebilirsiniz.

## WebSocket protokolü

Kendi istemcinizi yazmak isterseniz: `ws://localhost:8000/ws`

- **Gönderilen:** ikili mesaj = Int16 little-endian PCM, 16 kHz, mono. Bitirmek için `{"type": "stop"}`.
- **Alınan:** `ready`, `partial`, `final`, `discard`, `stats`, `error`, `done` tipli JSON mesajlar (ayrıntı: `src/live_stream/server.py`).

## Proje yapısı

```
src/live_stream/
├── config.py        # .env / ortam değişkeni ayarları
├── vad.py           # Silero ve enerji tabanlı konuşma algılayıcı
├── segmenter.py     # cümle sonu tespiti, ara/kesin iş üretimi
├── transcriber.py   # Faster-Whisper (GPU→CPU yedekli), uydurma filtresi
├── recorder.py      # ses + metnin anında diske yazılması
├── session.py       # bağlantı başına oturum
├── server.py        # FastAPI + WebSocket
└── static/index.html  # altyazı ekranı
```

## Testler

```bash
pip install -e ".[dev]"
pytest
```

Testler model indirmeden çalışır (sahte transkripsiyon motoru kullanılır).

## Lisans

MIT
