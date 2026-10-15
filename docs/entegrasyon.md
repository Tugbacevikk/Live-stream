# Entegrasyon rehberi

Bu sistemi başka bir uygulamaya (ör. mesajlaşma sistemi) bağlayan geliştirici için. Kısaca:
ses dosyasını `POST /jobs` ile gönderin, `job_id` alın; sonucu ya `GET /jobs/{id}` ile sorgulayın
ya da `callback_url` verip iş bitince size gelen bildirimi bekleyin (önerilen).

## 1. İş gönderme

```
POST /jobs            (multipart/form-data)
X-API-Key: <API_KEY>  (API_KEY tanımlıysa; production'da zorunlu)
```

| Alan | Zorunlu | Açıklama |
|---|---|---|
| `file` | evet | Ses dosyası. WAV, MP3, FLAC, OGG, OPUS; M4A/AAC/AMR/WEBM için sunucuda ffmpeg gerekir |
| `language` | hayır | `tr`, `vi` ya da `auto` (varsayılan). `auto`: dil sesten bulunur (yalnızca bir dil açıksa o dil) |
| `num_speakers` | hayır | Biliniyorsa konuşmacı sayısı (1–20). `1` verilirse konuşmacı ayırma hiç çalışmaz (daha hızlı) |
| `callback_url` | hayır | İş bitince sonucun POST edileceği adres |
| `notify` | hayır | Hangi olaylarda bildirim gitsin, virgülle: `running,done,failed,rejected`. Boşsa `NOTIFY_EVENTS` (varsayılan `done,failed,rejected`) |
| `external_id` | hayır | Sizin sistemdeki kimlik (ör. mesaj kimliği). Aynısı tekrar gelirse yeni iş açılmaz |
| `profile` | hayır | `full` (varsayılan) ya da `feedback`: kısa sesli mesajlar için hızlı mod (10 sn altında konuşmacı ayırma yok, Whisper tek ışın) |

Yanıtlar:

| Kod | Anlamı |
|---|---|
| 202 | İş kuyruğa girdi: `{"job_id", "status": "queued", "duration", "language"}` |
| 200 | Aynı `external_id` daha önce gelmiş: `{"job_id": <eski iş>, "status", "tekrar": true}`. Dosya ikinci kez saklanmaz |
| 400 | Ses okunamadı (bozuk dosya) |
| 401 | API anahtarı yok/yanlış |
| 413 | Dosya `MAX_UPLOAD_MB`'tan büyük ya da ses `MAX_AUDIO_MINUTES`'tan uzun |
| 415 | Desteklenmeyen dosya türü |
| 422 | Geçersiz alan (dil, profil, notify, callback adresi) |
| 429 | İstek sınırı (`RATE_LIMIT_PER_MINUTE`, varsayılan kapalı). `Retry-After` başlığına uyun |
| 503 | Sistem dolu (`MAX_PENDING_JOBS` iş bekliyor). `Retry-After` (60 sn) bekleyip tekrar gönderin |

**Ağ hatasında tekrar gönderme:** her isteğe `external_id` koyun. Yanıt gelmeden bağlantı koparsa
aynı `external_id` ile tekrar gönderin; iş iki kez açılmaz (veritabanı kısıtıyla garanti, yarış
durumunda da).

**Ani yük (4000 mesaj):** hepsini aynı anda gönderebilirsiniz; işler kuyrukta sırayla işlenir.
503 alan isteği `Retry-After` sonra yeniden deneyin.

## 2. İş durumları

```
queued ──► running ──► done
   │          │
   │          └──────► failed     (deneme hakkı bitti; error alanında neden)
   └─────────────────► rejected   (bilinçli ret: ör. desteklenmeyen dil tespit edildi)
```

`GET /jobs/{id}` her zaman şu alanları döner: `job_id, status, requested_language, num_speakers,
external_id, profile, notify, language, error, created_at, updated_at`; `done` ise ayrıca `result`.

## 3. Sonuç biçimi (`result`, ve `job.done` bildiriminin gövdesi)

```json
{
  "job_id": "6f26e5a7…",
  "status": "done",
  "duration": 60.0,
  "model": "faster-whisper-medium",
  "language": {"code": "tr", "confidence": 1.0, "source": "varsayilan", "uncertain": false},
  "speakers": 2,
  "text": "tüm metin",
  "utterances": [
    {"index": 0, "speaker": "speaker_00", "start": 0.0, "end": 4.3, "text": "…",
     "confidence": 0.91, "low_confidence_words": [3]}
  ],
  "quality": {"kapsama_yuzde": 98.1, "kurtarilan_kelime": 0}
}
```

- `speaker`: `speaker_00`, `speaker_01`… Kime ait olduğu bulunamayan kısa parçalar `BELIRSIZ`.
- `confidence`: satırın ortalama kelime güveni (0–1). `low_confidence_words`: güveni 0,5'in altındaki
  kelimelerin satır içindeki sıra numaraları. Bu kelimelerde hata oranı ~3 kat (olcumler.md);
  arayüzde altı çizili gösterilir, kontrol eden kişi önce bunlara bakmalı.
- `quality.kapsama_yuzde`: konuşma olan sürenin ne kadarında metin çıktı. Düşükse kayıp olabilir.
- `edited: true`: satır bir kişi tarafından düzeltilmiş.
- Noktalama yoktur, metin küçük harftir (istenen biçim).

## 4. Bildirimler (webhook)

İş durumu değişince `callback_url`'e POST gönderilir. Bildirimler veritabanında "outbox" tablosuna
iş durumuyla aynı işlemde yazılır: sunucu çökse de kaybolmaz.

| Olay (`X-Event`) | Ne zaman | Gövde |
|---|---|---|
| `job.running` | İşlenmeye başladı (varsayılanda kapalı) | `{"job_id", "status": "running", "error": null}` |
| `job.done` | Bitti | Bölüm 3'teki sonuç |
| `job.failed` | Hata, deneme hakkı bitti | `{"job_id", "status": "failed", "error"}` |
| `job.rejected` | Bilinçli ret | `{"job_id", "status": "rejected", "error"}` |
| `live.ended` | Canlı altyazı oturumu bitti (Bölüm 4.1) | Oturumun canlı metni, dili, süresi, `job_id` |

Başlıklar:

```
Content-Type: application/json; charset=utf-8
User-Agent: live-stream-webhook/1
X-Event: job.done
X-Job-Id: <job_id>                             (işe bağlı değilse boş)
X-Session-Id: <oturum kimliği>                 (yalnızca live.ended)
X-Delivery-Id: <bu teslimin kimliği>
X-Timestamp: <unix saniye>                     (WEBHOOK_SECRET tanımlıysa)
X-Signature: sha256=<hex HMAC>                 (WEBHOOK_SECRET tanımlıysa)
```

**Teslim kuralları**

- 2xx yanıt = teslim edildi. Başka her şey (hata kodu, zaman aşımı, bağlantı yok) = tekrar:
  10 sn, 20 sn, 40 sn … en fazla 1 saat aralıkla, `WEBHOOK_MAX_ATTEMPTS` (varsayılan 8) kez.
- Aynı bildirim birden fazla gelebilir (ağ koparsa). Alıcı `X-Delivery-Id`'yi saklayıp tekrarı atlamalı.
- Yeniden analiz edilen iş (Bölüm 6) aynı `job_id` ile yeni bir `job.done` gönderir (farklı
  `X-Delivery-Id`). Alıcı son geleni geçerli saymalı.
- Hızlı yanıt verin (10 sn içinde); ağır işi kendi kuyruğunuza bırakın.
- Deneme hakkı biten bildirimler "ölü" sayılır: `GET /webhooks/dead` listeler,
  `POST /webhooks/{id}/retry` yeniden kuyruğa koyar. `/metrics/prometheus`'ta
  `asr_webhook_deliveries{state="olu"}` izlenebilir.
- Kurulum sırasında: `POST /webhooks/test {"url": "..."}` imzalı bir `ping` gönderir ve alıcının
  yanıt kodunu döner.

### 4.1 Canlı oturum bitti (`live.ended`)

Canlı altyazı oturumu bittiğinde (konuşmacı **Durdur**'a bastığında ya da bağlantı koptuğunda)
iki bildirim gider, ikisi de aynı adrese ve aynı kurallarla (imza, tekrar, ölü teslim):

1. **`live.ended`** — hemen: ekranda görünen canlı metin.
2. **`job.done`** — kayıt konuşmacı ayrımından geçince (birkaç saniye ile birkaç dakika):
   Bölüm 3'teki sonuç, kimin ne söylediği satır satır. `job_id` iki bildirimde aynıdır;
   işin `external_id`'si `live:<session_id>`.

Adres: sunucuda `LIVE_WEBHOOK_URL` (bütün oturumlar) ya da oturum başına
`ws://<sunucu>/ws/live?key=<API_KEY>&callback_url=https://...` (öncelikli; sunucuda `API_KEY`
tanımlı olmalı, anahtarsız biri sunucuya istediği adrese istek attıramasın). Adres yoksa
bildirim gitmez.

Bağlantı kısa süre koparsa (`LIVE_RESUME_SEC`, varsayılan 30 sn) oturum bitmiş sayılmaz: istemci
`resume=<session_id>&token=<resume_token>&after=<son satır id>` ile döner, aynı oturum sürer ve
tek bir `live.ended` gider. Süre dolarsa oturum `reason: "disconnect"` ile kapanır.
Adres, `callback_url` ile aynı denetimden geçer (production'da iç ağ adresleri yalnızca
`WEBHOOK_ALLOWED_HOSTS` ile).

```json
{
  "event": "live.ended",
  "session_id": "3f9c1b2e8d4a4f0c9e6b7a1d2c3e4f50",
  "reason": "stop",
  "started_at": "2026-10-15T07:30:02+00:00",
  "ended_at": "2026-10-15T07:52:41+00:00",
  "duration": 1358.4,
  "language": {"code": "tr", "source": "secim", "confidence": 1.0},
  "lines": [
    {"id": 0, "start": 1.2, "end": 4.9, "text": "Herkese merhaba, ben Tuğba."},
    {"id": 1, "start": 5.6, "end": 11.0, "text": "Bugün canlı altyazı sistemini anlatacağım."}
  ],
  "text": "Herkese merhaba, ben Tuğba. Bugün canlı altyazı sistemini anlatacağım.",
  "recording": "2026-10-15_10-30-02",
  "job_id": "a1b2c3d4e5f6..."
}
```

| Alan | Açıklama |
|---|---|
| `reason` | `stop` (Durdur) ya da `disconnect` (bağlantı koptu; kalan cümleler yine de işlendi) |
| `duration` | Oturumda gelen sesin süresi (sn) |
| `language.source` | `secim` (kullanıcı seçti) ya da `tespit` (Otomatik; ilk cümleden bulundu) |
| `lines` | Kesinleşen satırlar; `start`/`end` oturum başından saniye. Türkçede büyük harf ve noktalama vardır, Vietnamcada yoktur |
| `job_id` | Konuşmacı ayrımı için açılan toplu iş; `null`: analiz kapalıydı ya da hiç konuşma yoktu (o zaman `job.done` gelmez) |

Canlı metin hızlı üretilir ve konuşmacı bilgisi taşımaz; kalıcı kayıt için `job.done`'daki
sonucu esas alın.

### İmza doğrulama

İmza `HMAC_SHA256(WEBHOOK_SECRET, "{X-Timestamp}.{ham gövde}")`. Gövdeyi JSON olarak çözüp yeniden
yazmadan, **gelen ham baytlar** üzerinden hesaplayın. Zaman damgası 5 dakikadan eskiyse reddedin
(eski isteğin tekrar oynatılmasına karşı).

Python (FastAPI):

```python
import hashlib, hmac, time
from fastapi import FastAPI, Request, HTTPException

SECRET = b"..."  # WEBHOOK_SECRET
app = FastAPI()
seen = set()     # gerçekte veritabanı

@app.post("/asr-callback")
async def callback(req: Request):
    body = await req.body()
    ts, sig = req.headers.get("X-Timestamp", ""), req.headers.get("X-Signature", "")
    good = "sha256=" + hmac.new(SECRET, ts.encode() + b"." + body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(good, sig) or abs(time.time() - int(ts or 0)) > 300:
        raise HTTPException(401)
    if req.headers["X-Delivery-Id"] in seen:
        return {"ok": True}                       # tekrar: yine 2xx dön
    seen.add(req.headers["X-Delivery-Id"])
    data = await req.json()
    ...                                           # kendi kuyruğunuza bırakın
    return {"ok": True}
```

Node.js (Express):

```js
const crypto = require("crypto");
app.post("/asr-callback", express.raw({ type: "application/json" }), (req, res) => {
  const ts = req.get("X-Timestamp") || "", sig = req.get("X-Signature") || "";
  const good = "sha256=" + crypto.createHmac("sha256", process.env.WEBHOOK_SECRET)
    .update(ts + ".").update(req.body).digest("hex");
  const ok = sig.length === good.length && crypto.timingSafeEqual(Buffer.from(sig), Buffer.from(good));
  if (!ok || Math.abs(Date.now() / 1000 - Number(ts)) > 300) return res.sendStatus(401);
  const data = JSON.parse(req.body.toString("utf8"));
  res.sendStatus(200);
});
```

C# (ASP.NET Core):

```csharp
app.MapPost("/asr-callback", async (HttpRequest req) => {
    using var ms = new MemoryStream(); await req.Body.CopyToAsync(ms); var body = ms.ToArray();
    var ts = req.Headers["X-Timestamp"].ToString(); var sig = req.Headers["X-Signature"].ToString();
    using var h = new HMACSHA256(Encoding.UTF8.GetBytes(secret));
    var good = "sha256=" + Convert.ToHexString(
        h.ComputeHash(Encoding.UTF8.GetBytes(ts + ".").Concat(body).ToArray())).ToLowerInvariant();
    if (!CryptographicOperations.FixedTimeEquals(Encoding.UTF8.GetBytes(good), Encoding.UTF8.GetBytes(sig))
        || Math.Abs(DateTimeOffset.UtcNow.ToUnixTimeSeconds() - long.Parse(ts)) > 300)
        return Results.Unauthorized();
    return Results.Ok();
});
```

## 5. Sorgulama ile çalışma (callback kullanmadan)

`GET /jobs/{id}`'yi 2–5 sn aralıkla sorgulayın; `status` `done`, `failed` ya da `rejected` olunca
durun. Çok sayıda iş için callback daha verimlidir.

## 6. Sonucu düzeltme ve yeniden analiz

| İstek | Ne yapar |
|---|---|
| `PATCH /jobs/{id}/utterances/{n}` `{"text"?, "speaker"?}` | Satırı düzeltir; modelin ilk hali saklanır (ince ayar verisi) |
| `DELETE /jobs/{id}/utterances/{n}` | Uydurma satırı sonuçtan kaldırır (kayıtta kalır) |
| `POST /jobs/{id}/reanalyze` `{"num_speakers": 0–20}` | Aynı sesi yeni konuşmacı sayısıyla yeniden analiz eder (0 = sesten bul). **Aynı iş** kuyruğa döner, yeni iş açılmaz; elle düzeltmeler yeni sonuçla değişir. 409: iş bitmemiş ya da sesi silinmiş |
| `DELETE /jobs/{id}` | Bitmiş işi, sonucunu ve sesini siler |

Sesler `AUDIO_RETENTION_HOURS` (varsayılan 24) sonra silinir (düzeltilmiş işlerinki hariç); sonrasında `reanalyze` ve
ses dinleme 409/410 döner.

## 7. Veri setine aktarma (isteğe bağlı)

Kontrol edilmiş sonuçlar modelin doğruluğunu ölçmek (test) ya da ince ayar (eğitim) için
kullanılabilir. Varsayılan **kapalı**; `ENABLE_DATASET_EXPORT=1` ile açılır.

1. Arayüzde "Veri setine ekle" ya da `POST /jobs/{id}/dataset`
   `{"reviewer": "Ad", "consent": true, "use": "test"|"egitim", "notes": "..."}`.
   `consent: true` ve kontrol eden kişinin adı zorunlu; kim, ne zaman, hangi amaçla onay verdiği
   veritabanına yazılır.
2. Kayıt `DATASET_DIR/bekleyen/<iş>/` klasörüne düşer (repo dışında; ses commit edilmez).
3. Sorumlu kişi `python scripts/veri_seti.py listele` ile bakar, `onayla <iş>` ya da `reddet <iş>`.
   - `test` → `test/<dil>/diyalog.tsv` (doğrudan `benchmarks/diyalog_testi.py` ile ölçülür)
   - `egitim` → `egitim/<dil>/liste.tsv` (1–20 sn satır kesitleri; `finetune/veri_hazirla.py` girdisi)
   - Aynı ses iki bölüme giremez (ses özeti `ozetler.tsv`'de tutulur): test sonucu şişmez.
4. `scripts/duzeltmeleri_disa_aktar.py` yalnızca onay verilmiş işlerin düzeltmelerini dışa aktarır.

## 8. İzleme

| Uç | Ne için |
|---|---|
| `GET /health` | Veritabanı/kuyruk erişimi, iş sayıları, açık özellikler (anahtar istemez) |
| `GET /health/ready` | Modeller yüklü ve her dil kuyruğunda çalışan işçi var mı (yük dengeleyici için) |
| `GET /metrics` | Gecikme, "1 dk ses ≤ 1 dk" hedefine uyum, kuyruk doluluğu (JSON) |
| `GET /metrics/prometheus` | Aynısı + aşama süreleri, webhook teslim/ölü sayıları |
