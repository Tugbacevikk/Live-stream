# 0006 — Fine-tuning projesinden alınanlar, düzeltilenler, alınmayanlar

**Tarih:** 14 Ekim 2026 · **Durum:** Kabul

Fine-tuning projesindeki (Tugbacevikk/fine-tuning) yapıların bu sistemdeki karşılığı.

## Alındı (düzeltilerek)

| Projede | Burada | Düzeltilen |
|---|---|---|
| FasterWhisperAdapter | `faster_whisper_asr.py` (Türkçe, medium) | Ayarlar kurucuda; tembel sonuç yüzünden yakalanamayan toplu çözme hatası; noktalamaya takılan tekrar süzgeci; `log_prob_threshold=None` ile sessiz konuşmanın atılması; istemin çıktıya sızması |
| Celery + Redis | Redis Streams (0001) | Windows'ta çalışır |
| SQLAlchemy + Alembic | code-first depo (0002) | SQLite için toplu değişiklik modu; `external_id` yarışı hata metni yerine kısıtla |
| `external_id` | aynı | — |
| Webhook outbox + HMAC | aynı (+ deneme bildirimi, ölü teslim listesi, yeniden deneme, iş başına olay seçimi, `job.running`) | — |
| SSRF denetimi | `security.callback_problem` | — |
| Rate limiter (Redis) | `rate_limit.py` (varsayılan kapalı) + bekleyen iş üst sınırı (503) | Tek entegre sistemin 4000'lik ani yükü reddedilmesin diye sınır kapalı, koruma kuyruk derinliğiyle |
| Sweeper + kilit | `bakim` + `RedisLock` | Kilit yenileme Lua ile atomik; QUEUED işi yalnızca kuyruk boşsa yeniden koyar (çift işleme yok) |
| Retention | `Maintenance` | Düzeltilmiş işin sesi ince ayar verisi olduğu için kayıtla birlikte silinir |
| `PIPELINE_PROFILE=feedback\|full` | aynı, istekte `profile=` ile iş başına da | — |
| Prometheus `/metrics` | `/metrics/prometheus` | Aşama süreleri veritabanından: ayrı işçi süreçlerindeki işler de görünür |
| `/health/ready` | aynı (modeller / canlı işçiler) | — |
| Arayüz: satır düzeltme/silme, RTTM | aynı + düzeltmelerden ince ayar verisi | Silme kalıcı değil (kayıtta kalır) |
| SemanticRefiner "sandwich" | `FusionConfig.sandwich_max` | Ölçüldü; varsayılan kapalı (olcumler.md) |
| Rust DSP (PyO3) | `native/` (birleştirme, enerji kesimi) | Python sürümüyle eşlik testi zorunlu (0005) |
| Speech gap detector + yeniden çözme | `application/coverage.py` + `asr.rescue` (güvenlik ağı) | Kenar payındaki kelimeler iki kez yazılıyordu (tekrar ayıklanıyor); sıcaklık yedeği (0 → 0,2 → 0,4 → 0,6) her bölüme uygulanıyordu, yalnızca boşluk kurtarmada kullanılıyor (hız); kapsama yüzdesi sonuçta `quality` |
| Kelime sıcaklığı / güveni | `confidence`, `low_confidence_words` | Vietnamca (Zipformer) için de: log olasılıktan; 0,5 altı kelimeler ~3 kat hatalı (olcumler.md) |
| Re-analyze | `POST /jobs/{id}/reanalyze` | Projede yeni iş açılıyor, dosyanın tamamı belleğe okunuyor ve sabit bir klasör yolu ("Yeni klasör") kullanılıyordu; burada aynı iş kuyruğa döner, ses diskteki yerinden okunur |
| "Gerçek test setine kaydet" | `POST /jobs/{id}/dataset` + `scripts/veri_seti.py` | Projede doğrudan repo içine (tests/fixtures) yazılıyordu: hasta sesi commit edilebilirdi. Burada repo dışı klasör, kontrol eden + açık onay zorunlu, onay kuyruğu, test/eğitim ayrımı ve ses özetiyle sızıntı denetimi; düzeltme dışa aktarımı yalnızca onaylı işlerden |
| Yabancı yazı süzgeci | `normalizer.latin_only` + Kiril benzer harf eşlemesi | Whisper'ın Türkçe seste ürettiği Devanagari/Kiril kelimeler ("कजतच", "nasыsыn") |

## Ölçüldü, alınmadı

- **SpeakerCountEstimator (silhouette ile kişi sayısı):** önbellekteki aynı gömmelerle doğru kişi
  sayısı — tek kişilik 34/84 (bizimki 72/84), gerçek diyalog 6/9 (7/9), kesit 11/56 (34/56),
  sentetik 14/28 (26/28). Ayrıca sklearn `silhouette_score(metric="precomputed")` köşegeni sıfır
  olmayan uzaklık matrisinde hata veriyor (kosinüs uzaklığı yuvarlamayla 0 çıkmıyor). Bizim
  NME-SC özdeğer aralığı yöntemi kaldı.
- **SpeakerMerger "küçük konuşmacıyı koru" koruması:** `DiarizerConfig.small_keep_distance` olarak
  eklendi, ölçüldü (hasta setinde 7 → 6, kötüleşti); varsayılan 0 (kapalı).
- **Sandwich düzeltmesi varsayılan açık:** DER 10,64 → 10,14 (1,5 sn) ama test diyaloglarında
  gerçek kısa onay sesleri ("ừ", "evet") yok; gerçek kayıtta onları silebilir. Kapalı, ayarla açılır.

## Alınmadı

- **Gürültü giderme (DeepFilterNet):** PyTorch ister; internetsiz CPU kurulumuna ve "1 dk ses ≤ 1 dk"
  hedefine uymaz. ASR modelleri gürültülü veriyle eğitildiği için ön temizlik çoğu zaman yarardan
  çok zarar verir; gerekirse ölçülerek eklenmeli.
- **Üst üste konuşma özeti (overlap):** buradaki konuşmacı ayırma çakışmasız aralık üretir; özet hep
  sıfır çıkardı. pyannote yolu (`DIARIZER=pyannote`) çakışma verebilir; ihtiyaç olursa onunla.
- **LLM ile düzeltme (Ollama):** projede de deneysel; internetsiz sunucuda ayrı model servisi ister.
- **S3 depolama:** tek makine, internetsiz kurulum; yükleme klasörü yeterli.
- **ASR ve konuşmacı ayırmayı aynı iş içinde paralel çalıştırma:** eşzamanlı iş sayısı zaten
  çekirdekleri dolduruyor; iş içi paralellik çekirdekleri aşırı paylaştırır.
- **Escalating/fallback diarizer, SpeechBrain:** spektral kümeleme gerçek veride daha iyi ölçüldü (0003).
