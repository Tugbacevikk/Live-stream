# 0007 — Tek repo, tek sistem: toplu iş + canlı altyazı (Türkçe, Vietnamca)

**Tarih:** 15 Ekim 2026 · **Durum:** Kabul (aynı gün güncellendi: Vietnamca eklendi, aşağıda)

## Bağlam

Üç repo aynı işin farklı parçalarını yapıyordu:

- **fine-tuning** — ilk Türkçe sistem (dosya → metin + konuşmacı), ~18.500 satır.
- **Vietnamese-speech-to-text-models** — fine-tuning'in yeniden yazılmış hali; Türkçe +
  Vietnamca, ölçülmüş kararlar (ADR 0001–0006), ~9.500 satır.
- **Live-stream** — konferans için canlı altyazı, ~900 satır, kendi Whisper ve VAD kodu.

Üçünde de ayrı birer faster-whisper sarmalayıcısı, Silero VAD sarmalayıcısı ve uydurma
süzgeci vardı. Karar: **repo Live-stream**; ilk adımda yalnızca Türkçe, aynı gün
Vietnamca da eklendi.

## Karar

Vietnamese-speech-to-text-models çekirdeği Live-stream reposuna taşındı;
Live-stream'in canlı akışı bu çekirdeğin katmanlarına yerleşti. Paket adı `live_stream`.

### Bileşen bileşen karşılaştırma

| Bileşen | fine-tuning | Vietnamese-speech-to-text-models | Alınan | Gerekçe |
|---|---|---|---|---|
| Türkçe ASR | faster-whisper adaptörü; ayarlar gizli bağımlılık, toplu çözme hatası yakalanmıyor | Aynı model, 5 hata düzeltilmiş, medium varsayılan | Vietnamese + Live-stream'in GPU→CPU yedeği ve canlı metin kipi | Tek sınıf, tek model nesnesi iki iş için |
| VAD | Silero (PyTorch) | Silero ONNX (sherpa-onnx) + kenar payı | Vietnamese; canlı mod da aynı ONNX modeli | PyTorch bağımlılığı kalktı |
| Konuşmacı ayırma | pyannote 3.1 (yavaş), sherpa-onnx, SpeechBrain, yedekli yollar | CAM++ imzası + NME-SC spektral kümeleme | Vietnamese | Ölçüldü (ADR 0003); pyannote ONNX seçenek olarak kaldı |
| Konuşmacı sayısı | silhouette | NME-SC özdeğer aralığı | Vietnamese | Tek kişilik kayıtta 72/84 doğru (silhouette 34/84) |
| Kuyruk | Celery / Redis Streams | Redis Streams + yerel mod iş parçacığı kuyruğu | Vietnamese | Windows'ta çalışır, Redis'siz yerel mod |
| Veritabanı | PostgreSQL, SQLAlchemy + Alembic | Code-first, SQLite ve PostgreSQL aynı kod | Vietnamese | Yerel modda kurulum gerekmez |
| Webhook | Outbox + HMAC | + deneme bildirimi, ölü teslimler, yeniden deneme, iş başına olay | Vietnamese | Entegrasyon için gerekli |
| Güvenlik ağı (kaçan konuşma) | Var, her parçada sıcaklık geri çekilmesi (yavaş) | Yalnızca boşluklarda | Vietnamese | Aynı kazanç, daha hızlı |
| Gürültü giderme (DeepFilterNet) | Var | Yok | Alınmadı | PyTorch ister, "1 dk ≤ 1 dk" hedefine uymuyor (ADR 0006) |
| Üst üste konuşma, LLM düzeltme, S3 | Var | Yok | Alınmadı | ADR 0006'daki ölçüm ve gerekçeler |
| Rust çekirdeği | Deneysel | `native/` (birleştirme, enerji kesimi, parça gruplama) | Alınmadı | Python sürümü birebir aynı sonucu veriyor; derleme zorunluluğu kalktı |
| Dil tespiti (Whisper-tiny/small) | Yok | Var | Vietnamese | Toplu işte ve canlı modda "otomatik" seçimde |
| Vietnamca (Zipformer-68M, ince ayar hattı) | Yok | Var | Vietnamese | Toplu iş + canlı altyazı |
| Arayüz | Tek sayfa | Satır düzeltme/silme, RTTM, veri seti onayı | Vietnamese + canlı sayfa (`/canli`) | Live-stream'in sunum modu korundu |
| Canlı altyazı | Yok | Yok | Live-stream | Segmenter, oturum kaydı, WebSocket protokolü |

### Canlı altyazı çekirdeğe nasıl yerleşti

| Live-stream ilk sürüm | Şimdi | Değişen |
|---|---|---|
| `segmenter.py`, `session.py` | `application/live.py` | Yalnızca portları bilir; mantık aynı |
| `transcriber.py` (kendi Whisper'ı) | `FasterWhisperAsr.transcribe_text` | Toplu işle aynı model nesnesi, aynı uydurma ve tekrar süzgeci |
| `vad.py` (PyTorch Silero) | `infrastructure/live_vad.py` | sherpa-onnx, `models/silero_vad.onnx` |
| `recorder.py` | `infrastructure/recorder.py` | Ses yolunu döndürür |
| `server.py` (ayrı sunucu) | `interface/live_server.py`, yerel sunucuya eklenir | `/ws/live` (eski `/ws` de çalışır) |
| Kayıt elle başka projeye verilirdi | Oturum bitince otomatik toplu iş | `done` mesajında `job_id` |

## Sonuçlar

- Tek komut (`baslat.bat` ya da `python -m live_stream`): `/` dosya analizi, `/canli` canlı altyazı.
- Canlı ve toplu iş aynı süreçte aynı modeli paylaşır; `LIVE_MODEL` farklıysa (4 GB GPU'da
  `faster-whisper-small`) ikinci model ayrıca yüklenir.
- Sunucu tarafı (tam mod, 4000 mesaj, yalnızca CPU) değişmedi; `WHISPER_DEVICE=cpu`.
- `LANGUAGES=tr` ile tek dil: tespit adımı atlanır, iş doğrudan o dilin kuyruğuna gider,
  dil tespiti modeli yüklenmez ve indirilmez.

## Güncelleme: Vietnamca (15 Ekim 2026)

- Toplu iş: Vietnamese reposundaki hali olduğu gibi (Zipformer-68M, sesin tamamı en sessiz
  anlardan bölünür, Whisper-tiny → small dil tespiti, terim listesi, ince ayar hattı).
- Canlı altyazı: dil oturum başında seçilir (`language=tr|vi|auto`). `auto`: ilk yeterince uzun
  cümleden (≥1,5 sn ya da ilk kesin cümle) toplu işle aynı kademeli tespitle bulunur ve oturum
  boyunca kilitlenir (`LanguageLock`); her cümlede tespit hem gecikme ekler hem kısa cümlede
  yanılır. Desteklenmeyen dil ya da hata olursa ilk dil (tr) kullanılır.
- Zipformer canlıda da aynı model nesnesidir; çok hızlı olduğu için ara metin her seferinde
  baştan çözülür, `fast` kipi ve istem (prompt) ona uygulanmaz. Ekran biçimi: küçük harf.
- Oturum kaydı oturumun diliyle toplu işe gider; dil bulunamadıysa toplu iş kendisi bulur.
- Eski Vietnamca reposundaki bir hata düzeltildi: `sentencepiece` kurulu değilse model indirici
  çöküyordu (bpe.vocab). Paket bağımlılıklara eklendi; yine de yoksa indirme sürer, yalnızca
  Vietnamca terim listesi çalışmaz.
- fine-tuning ve Vietnamese-speech-to-text-models başvuru kaynağı olarak kalır; güncel kod burası.
