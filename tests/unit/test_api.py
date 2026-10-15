import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from live_stream.application.jobs import DetectLanguage, RunTranscription
from live_stream.application.fusion import FusionEngine
from live_stream.application.language import LanguageResolver
from live_stream.application.transcribe import TranscribeAndDiarize
from live_stream.domain import JobStatus, SpeakerSegment, VadSegment, Word
from live_stream.infrastructure.engines import LazyAsrEngines
from live_stream.infrastructure.memory_store import MemoryJobStore
from live_stream.interface.api import create_app
from live_stream.interface.settings import Settings


class Queue:
    def __init__(self):
        self.sent = []

    def enqueue_detection(self, job_id):
        self.sent.append(("lid", job_id))

    def enqueue_transcription(self, job_id, language):
        self.sent.append((f"asr.{language}", job_id))


def wav_bytes(sec=1.0):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(16000 * sec), dtype=np.float32), 16000, format="WAV")
    return buf.getvalue()


@pytest.fixture
def ctx(tmp_path):
    store, queue = MemoryJobStore(), Queue()
    s = Settings(data_dir=str(tmp_path), languages=["vi", "tr"], max_upload_mb=1)
    return store, queue, TestClient(create_app(store, queue, s, {"sahte": lambda: None}))


def post(client, data=None, content=None, name="a.wav"):
    files = {"file": (name, content if content is not None else wav_bytes(), "audio/wav")}
    return client.post("/jobs", files=files, data=data or {})


def test_dil_verilmezse_tespit_kuyruguna(ctx):
    store, queue, c = ctx
    r = post(c)
    assert r.status_code == 202, r.text
    jid = r.json()["job_id"]
    assert queue.sent == [("lid", jid)]
    assert store.get(jid).status == JobStatus.QUEUED and r.json()["duration"] == 1.0


def test_tek_dil_varken_tespit_atlanir(tmp_path):
    """Şimdilik yalnızca Türkçe: 'auto' iş doğrudan Türkçe kuyruğuna gider, tespit işçisi gerekmez."""
    store, queue = MemoryJobStore(), Queue()
    c = TestClient(create_app(store, queue, Settings(data_dir=str(tmp_path), languages=["tr"])))
    jid = post(c).json()["job_id"]
    assert queue.sent == [("asr.tr", jid)] and store.get(jid).job.language is None
    assert post(c, {"language": "vi"}).status_code == 422  # tanımlı olmayan dil


def test_dil_verilirse_o_dilin_kuyruguna(ctx):
    _, queue, c = ctx
    jid = post(c, {"language": "TR", "callback_url": "https://x/hook"}).json()["job_id"]
    assert queue.sent == [("asr.tr", jid)]


def test_ayni_external_id_tekrar_is_acmaz(ctx, tmp_path):
    _, queue, c = ctx
    first = post(c, {"external_id": "mesaj-42"})
    assert first.status_code == 202
    again = post(c, {"external_id": "mesaj-42"})
    assert again.status_code == 200 and again.json()["job_id"] == first.json()["job_id"]
    assert again.json()["tekrar"] is True and len(queue.sent) == 1   # kuyruğa bir kez girdi
    assert len(list(tmp_path.glob("*.wav"))) == 1                    # ikinci dosya silindi


def test_hatali_istekler(ctx):
    _, queue, c = ctx
    assert post(c, {"language": "en"}).status_code == 422          # desteklenmeyen dil
    assert post(c, name="a.xyz").status_code == 415                 # dosya türü
    assert post(c, content=b"ses degil").status_code == 400          # bozuk dosya
    assert post(c, content=wav_bytes(40)).status_code == 413         # 1 MB sınırı
    assert post(c, {"callback_url": "ftp://x"}).status_code == 422
    assert c.post("/jobs", data={"language": "vi"}).status_code == 422  # dosya yok
    assert queue.sent == []


def test_sonuc_sorgulama_ve_uctan_uca_akis(ctx):
    """API -> lid kuyruğu -> asr kuyruğu, sahte modellerle, gerçek iş akışı sınıflarıyla."""
    store, queue, c = ctx

    class Loader:
        def load(self, p):
            return sf.read(p, dtype="float32")[0]

    class Vad:
        def split(self, a):
            return [VadSegment(0.0, a)]

    class Det:
        def probabilities(self, a):
            return {"vi": 0.95, "tr": 0.03, "en": 0.02}

    class Asr:
        name = "sahte-vi"

        def transcribe(self, segs):
            return [Word("xin", 0.1, 0.3), Word("chào", 0.3, 0.6)]

    class Diar:
        def diarize(self, a, segments=None, num_speakers=None):
            return [SpeakerSegment("speaker_00", 0.0, 1.0)]

    jid = post(c).json()["job_id"]
    assert c.get(f"/jobs/{jid}").json()["status"] == "queued"
    resolver = LanguageResolver(["vi", "tr"], [Det()])
    DetectLanguage(store, queue, Loader(), Vad(), resolver)(jid)
    assert queue.sent[-1] == ("asr.vi", jid)
    assert c.get(f"/jobs/{jid}").json()["language"]["code"] == "vi"

    pipeline = TranscribeAndDiarize(Loader(), Vad(), LanguageResolver(["vi", "tr"], None),
                                    LazyAsrEngines({"vi": Asr}), Diar(), FusionEngine(), store)
    run = RunTranscription(store, pipeline)
    run(jid)
    body = c.get(f"/jobs/{jid}").json()
    assert body["status"] == "done"
    assert body["result"]["text"] == "xin chào"
    assert body["result"]["language"] == {"code": "vi", "confidence": 0.9694, "source": "tespit",
                                          "uncertain": False}
    assert body["result"]["utterances"][0]["speaker"] == "speaker_00"
    assert run(jid) is None  # aynı iş tekrar gelirse atlanır

    assert c.get("/jobs/yok").status_code == 404
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["jobs"] == {"done": 1}


def test_konusmaci_sayisi_ve_sure_siniri(tmp_path):
    store, queue = MemoryJobStore(), Queue()
    s = Settings(data_dir=str(tmp_path), languages=["vi", "tr"], max_audio_minutes=0.5)
    c = TestClient(create_app(store, queue, s))
    r = post(c, {"num_speakers": "3"})
    assert r.status_code == 202
    assert store.get(r.json()["job_id"]).job.num_speakers == 3
    assert c.get(f"/jobs/{r.json()['job_id']}").json()["num_speakers"] == 3
    assert post(c, {"num_speakers": "0"}).status_code == 422
    assert post(c, content=wav_bytes(31)).status_code == 413   # 0,5 dakikadan uzun
    assert len(list(tmp_path.iterdir())) == 1                   # reddedilenlerin dosyası silindi


def test_api_anahtari(tmp_path):
    store, queue = MemoryJobStore(), Queue()
    s = Settings(data_dir=str(tmp_path), languages=["vi", "tr"], api_key="gizli")
    c = TestClient(create_app(store, queue, s))
    assert post(c).status_code == 401
    ok = c.post("/jobs", files={"file": ("a.wav", wav_bytes(), "audio/wav")},
                headers={"X-API-Key": "gizli"})
    assert ok.status_code == 202
    assert c.get(f"/jobs/{ok.json()['job_id']}").status_code == 401
    assert c.get(f"/jobs/{ok.json()['job_id']}", headers={"X-API-Key": "gizli"}).status_code == 200
    assert c.get("/health").status_code == 200   # sağlık kontrolü anahtar istemez
    with pytest.raises(RuntimeError):            # production'da anahtarsız açılmaz
        create_app(store, queue, Settings(data_dir=str(tmp_path), app_env="production", api_key=""))


def test_callback_dogrulama(tmp_path):
    store, queue = MemoryJobStore(), Queue()
    prod = Settings(data_dir=str(tmp_path), app_env="production", api_key="k", languages=["vi"])
    c = TestClient(create_app(store, queue, prod))
    h = {"X-API-Key": "k"}

    def send(url):
        return c.post("/jobs", files={"file": ("a.wav", wav_bytes(), "audio/wav")},
                      data={"callback_url": url}, headers=h).status_code

    assert send("http://127.0.0.1:9000/hook") == 422      # iç adres: SSRF
    assert send("http://10.0.0.5/hook") == 422
    allow = Settings(data_dir=str(tmp_path), app_env="production", api_key="k", languages=["vi"],
                     webhook_allowed_hosts=["10.0.0.5"])
    c2 = TestClient(create_app(store, queue, allow))
    r = c2.post("/jobs", files={"file": ("a.wav", wav_bytes(), "audio/wav")},
                data={"callback_url": "http://10.0.0.5/hook"}, headers=h)
    assert r.status_code == 202                           # izin listesindeki iç ağ alıcısı
    r = c2.post("/jobs", files={"file": ("a.wav", wav_bytes(), "audio/wav")},
                data={"callback_url": "http://baska.local/hook"}, headers=h)
    assert r.status_code == 422


@pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None, reason="ffmpeg yok")
def test_m4a_ffmpeg_ile_okunur(tmp_path):
    import subprocess

    from live_stream.infrastructure.audio import SoundfileLoader

    wav, m4a = tmp_path / "a.wav", tmp_path / "a.m4a"
    t = np.arange(16000 * 2) / 16000
    sf.write(wav, (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 16000)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), str(m4a)], check=True)
    a = SoundfileLoader().load(str(m4a))
    assert abs(len(a) / 16000 - 2.0) < 0.1
    store, queue = MemoryJobStore(), Queue()
    c = TestClient(create_app(store, queue, Settings(data_dir=str(tmp_path / "up"), languages=["vi"])))
    assert post(c, content=m4a.read_bytes(), name="kayit.m4a").status_code == 202


def test_metrikler(ctx):
    store, queue, c = ctx
    m = c.get("/metrics").json()
    assert m["gecikme"] == {"n": 0}


def test_arayuz_gecmis_ses_ve_silme(ctx, tmp_path):
    store, queue, c = ctx
    assert c.get("/").status_code == 200 and "Ses Analizi" in c.get("/").text
    jid = post(c, {"language": "vi"}).json()["job_id"]
    assert [j["job_id"] for j in c.get("/jobs").json()] == [jid]
    audio = c.get(f"/jobs/{jid}/audio")
    assert audio.status_code == 200 and audio.content[:4] == b"RIFF"
    assert c.delete(f"/jobs/{jid}").status_code == 409          # kuyruktaki iş silinmez
    store.set_status(jid, JobStatus.FAILED, "x")
    assert c.delete(f"/jobs/{jid}").status_code == 200
    assert c.get(f"/jobs/{jid}").status_code == 404 and list(tmp_path.glob("*.wav")) == []
    assert c.get("/jobs/yok/audio").status_code == 404


def test_asiri_yuk_ve_istek_siniri(tmp_path):
    from live_stream.infrastructure.rate_limit import MemoryRateLimiter

    store, queue = MemoryJobStore(), Queue()
    s = Settings(data_dir=str(tmp_path), languages=["vi", "tr"], max_pending_jobs=2,
                 rate_limit_per_minute=0)
    c = TestClient(create_app(store, queue, s))
    assert post(c).status_code == 202 and post(c).status_code == 202
    r = post(c)
    assert r.status_code == 503 and r.headers["Retry-After"] == "60"     # 2 iş bekliyor: dolu
    assert len(list(tmp_path.glob("*.wav"))) == 2                         # reddedilen dosya yazılmadı
    for j in list(store.jobs):
        store.set_status(j, JobStatus.FAILED, "x")
    assert post(c).status_code == 202                                     # yer açıldı

    now = [120.0]
    lim = MemoryRateLimiter(2, clock=lambda: now[0])
    s2 = Settings(data_dir=str(tmp_path), languages=["vi"], max_pending_jobs=0)
    c2 = TestClient(create_app(MemoryJobStore(), Queue(), s2, rate_limiter=lim))
    assert [post(c2).status_code for _ in range(3)] == [202, 202, 429]
    assert c2.post("/jobs", files={"file": ("a.wav", wav_bytes(), "audio/wav")},
                   headers={"X-API-Key": "baska"}).status_code == 202     # her istemcinin kendi sayacı
    now[0] += 60                                                          # yeni dakika
    assert post(c2).status_code == 202


def test_satir_duzeltme(ctx):
    from live_stream.domain import LanguageDecision, LanguageSource, Transcript, Utterance

    store, _, c = ctx
    jid = post(c, {"language": "tr"}).json()["job_id"]
    assert c.patch(f"/jobs/{jid}/utterances/0", json={"text": "x"}).status_code == 404   # bitmedi
    store.save(Transcript(jid, 3.0, [Utterance("speaker_00", 0, 1, "merhba doktor")], "m",
                          LanguageDecision("tr", 1.0, LanguageSource.REQUEST)))
    r = c.patch(f"/jobs/{jid}/utterances/0", json={"text": "  merhaba   doktor ", "speaker": "hasta"})
    assert r.status_code == 200 and r.json()["text"] == "merhaba doktor" and r.json()["speaker"] == "hasta"
    assert c.get(f"/jobs/{jid}").json()["result"]["utterances"][0]["edited"] is True
    assert c.patch(f"/jobs/{jid}/utterances/0", json={}).status_code == 422
    assert c.patch(f"/jobs/{jid}/utterances/0", json={"speaker": "<script>"}).status_code == 422
    (k,) = c.get("/corrections").json()
    assert k["original_text"] == "merhba doktor" and k["text"] == "merhaba doktor" and k["language"] == "tr"


def test_profil_bildirim_ve_satir_silme(ctx):
    from live_stream.domain import LanguageDecision, LanguageSource, Transcript, Utterance

    store, _, c = ctx
    assert post(c, {"profile": "hizli"}).status_code == 422
    assert post(c, {"notify": "done,patladi"}).status_code == 422
    jid = post(c, {"language": "tr", "profile": "feedback", "notify": "running, done",
                   "callback_url": "https://x/h"}).json()["job_id"]
    rec = c.get(f"/jobs/{jid}").json()
    assert rec["profile"] == "feedback" and rec["notify"] == ["running", "done"]
    store.set_status(jid, JobStatus.RUNNING)
    store.save(Transcript(jid, 3.0, [Utterance("speaker_00", 0, 1, "a"), Utterance("speaker_00", 1, 2, "b"),
                                     Utterance("speaker_00", 2, 3, "c")], "m",
                          LanguageDecision("tr", 1.0, LanguageSource.REQUEST)))
    assert store.events == [(jid, "job.running"), (jid, "job.done")]
    assert c.delete(f"/jobs/{jid}/utterances/1").status_code == 200
    assert c.delete(f"/jobs/{jid}/utterances/1").status_code == 404           # zaten silindi
    utts = c.get(f"/jobs/{jid}").json()["result"]["utterances"]
    assert [(u["index"], u["text"]) for u in utts] == [(0, "a"), (2, "c")]    # sıra numarası korunur
    assert c.patch(f"/jobs/{jid}/utterances/2", json={"text": "c2"}).json()["text"] == "c2"
    assert c.patch(f"/jobs/{jid}/utterances/1", json={"text": "x"}).status_code == 404


def test_prometheus_metrikleri(ctx):
    from live_stream.domain import LanguageDecision, LanguageSource, Transcript, Utterance

    store, _, c = ctx
    jid = post(c, {"language": "vi"}).json()["job_id"]
    store.save(Transcript(jid, 60.0, [Utterance("speaker_00", 0, 1, "a")], "m",
                          LanguageDecision("vi", 1.0, LanguageSource.REQUEST),
                          timings={"asr": 3.0, "konusmaci": 1.2, "toplam": 6.0}))
    r = c.get("/metrics/prometheus")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    body = r.text
    assert 'asr_jobs{status="done"} 1' in body
    assert 'asr_stage_seconds_bucket{le="5",stage="asr"} 1' in body
    assert 'asr_stage_seconds_count{stage="toplam"} 1' in body
    assert 'asr_realtime_ratio_bucket{le="0.1",language="vi"} 1' in body      # 6 / 60 = 0,1
    assert 'asr_webhook_deliveries{state="olu"} 0' in body


def test_webhook_deneme_bildirimi(ctx, monkeypatch):
    import httpx

    import live_stream.infrastructure.webhook as wh

    seen = []

    def handler(req):
        seen.append((req.headers["X-Event"], req.headers["User-Agent"], "X-Signature" in req.headers))
        return httpx.Response(204)

    res = wh.send_test("http://alici/hook", "gizli", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert res["ok"] and res["status"] == 204 and res["imzali"]
    assert seen == [("ping", wh.USER_AGENT, True)]
    _, _, c = ctx
    monkeypatch.setattr(wh, "send_test", lambda url, secret: {"ok": True, "url": url})
    assert c.post("/webhooks/test", json={"url": "https://x/h"}).json() == {"ok": True, "url": "https://x/h"}
    assert c.get("/webhooks/dead").json() == []
    assert c.post("/webhooks/5/retry").status_code == 404


def test_hazir_olma(tmp_path):
    store, queue = MemoryJobStore(), Queue()
    s = Settings(data_dir=str(tmp_path), languages=["vi"])
    durum = {"ok": False}
    c = TestClient(create_app(store, queue, s, readiness={"modeller": lambda: (durum["ok"], "x")}))
    assert c.get("/health/ready").status_code == 503
    durum["ok"] = True
    r = c.get("/health/ready")
    assert r.status_code == 200 and r.json()["ready"] is True


def test_yeniden_analiz(ctx):
    from live_stream.domain import LanguageDecision, LanguageSource, Transcript, Utterance

    store, queue, c = ctx
    jid = post(c, {"language": "vi"}).json()["job_id"]
    assert c.post(f"/jobs/{jid}/reanalyze", json={"num_speakers": 2}).status_code == 409   # bitmedi
    store.save(Transcript(jid, 3.0, [Utterance("speaker_00", 0, 1, "a")], "m",
                          LanguageDecision("vi", 1.0, LanguageSource.REQUEST)))
    queue.sent.clear()
    r = c.post(f"/jobs/{jid}/reanalyze", json={"num_speakers": 2})
    assert r.status_code == 202 and queue.sent == [("asr.vi", jid)]
    rec = c.get(f"/jobs/{jid}").json()
    assert rec["status"] == "queued" and rec["num_speakers"] == 2
    assert c.post(f"/jobs/{jid}/reanalyze", json={"num_speakers": 50}).status_code == 422


def test_veri_setine_aktarma(tmp_path):
    import json

    from live_stream.domain import LanguageDecision, LanguageSource, Transcript, Utterance
    from live_stream.infrastructure import dataset

    body = {"reviewer": "Dr. A", "consent": True, "use": "test"}
    store, queue = MemoryJobStore(), Queue()
    off = TestClient(create_app(store, queue, Settings(data_dir=str(tmp_path / "up"), languages=["vi"])))
    jid = post(off, {"language": "vi"}).json()["job_id"]
    assert off.post(f"/jobs/{jid}/dataset", json=body).status_code == 403       # varsayılan kapalı
    assert off.get("/health").json()["features"]["dataset_export"] is False

    root = tmp_path / "vs"
    s = Settings(data_dir=str(tmp_path / "up"), languages=["vi"], enable_dataset_export=True, dataset_dir=str(root))
    c = TestClient(create_app(store, queue, s))
    assert c.get("/health").json()["features"]["dataset_export"] is True
    assert c.post(f"/jobs/{jid}/dataset", json=body).status_code == 409         # bitmedi
    assert c.post("/jobs/yok/dataset", json=body).status_code == 404
    store.save(Transcript(jid, 1.0, [Utterance("speaker_00", 0.0, 1.0, "xin chào")], "m",
                          LanguageDecision("vi", 1.0, LanguageSource.REQUEST)))
    assert c.post(f"/jobs/{jid}/dataset", json={**body, "consent": False}).status_code == 422
    assert c.post(f"/jobs/{jid}/dataset", json={**body, "use": "baska"}).status_code == 422
    assert store.corrections(consented_only=True) == []
    store.update_utterance(jid, 0, text="xin chào bác sĩ")
    assert store.corrections(consented_only=True) == []                          # onay yok
    r = c.post(f"/jobs/{jid}/dataset", json=body)
    assert r.status_code == 200, r.text
    assert len(store.corrections(consented_only=True)) == 1
    d = root / "bekleyen" / jid
    assert {p.name for p in d.iterdir()} == {"ses.wav", "konusmaci.rttm", "ref.json", "satirlar.tsv", "meta.json"}
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["kontrol_eden"] == "Dr. A" and meta["dil"] == "vi" and meta["duzeltilen_satir"] == 1
    assert json.loads((d / "ref.json").read_text(encoding="utf-8"))["konusmaci_metin"] == {"speaker_00": "xin chào bác sĩ"}
    assert [m["job_id"] for m in dataset.pending(root)] == [jid]

    target = dataset.approve(root, jid, approver="Dr. B")
    assert target == root / "test" / "vi" and not d.exists()
    assert (target / "audio" / f"{jid}.wav").exists() and (target / "diyalog.tsv").exists()

    # aynı ses yeniden aktarılıp eğitime onaylanamaz (test setine sızıntı)
    assert c.post(f"/jobs/{jid}/dataset", json={**body, "use": "egitim"}).status_code == 200
    with pytest.raises(ValueError, match="sızıntı"):
        dataset.approve(root, jid)
    dataset.reject(root, jid)
    assert dataset.pending(root) == []


def test_veri_seti_egitim_bolumu(tmp_path):
    import csv

    import numpy as np

    from live_stream.infrastructure import dataset

    audio = np.random.default_rng(0).standard_normal(16000 * 30).astype(np.float32) * 0.1
    utts = [{"speaker": "speaker_00", "start": 0.0, "end": 5.0, "text": "một"},
            {"speaker": "speaker_01", "start": 5.0, "end": 5.5, "text": "kısa"},          # 1 sn altı
            {"speaker": "speaker_00", "start": 6.0, "end": 29.0, "text": "uzun"},         # 20 sn üstü
            {"speaker": "BELIRSIZ", "start": 29.0, "end": 30.0, "text": "?"}]
    dataset.write_pending(tmp_path, "j1", audio, utts, {"dil": "vi", "kullanim": "egitim"})
    assert "BELIRSIZ" not in (tmp_path / "bekleyen" / "j1" / "konusmaci.rttm").read_text()
    target = dataset.approve(tmp_path, "j1")
    assert target == tmp_path / "egitim" / "vi"
    rows = list(csv.DictReader((target / "liste.tsv").open(encoding="utf-8"), delimiter="\t"))
    assert [(r["metin"], r["kayit"]) for r in rows] == [("một", "j1")]
    assert (target / rows[0]["dosya"]).exists()
    with pytest.raises(ValueError):
        dataset.write_pending(tmp_path, "j2", audio, utts, {"dil": "vi"})
        dataset.approve(tmp_path, "j2", use="hepsi")
