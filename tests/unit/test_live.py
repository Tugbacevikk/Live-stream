"""Canlı altyazı: cümle sonu tespiti, oturum kaydı, canlı metin süzgeçleri ve uçtan uca WebSocket.

Modeller gerekmez: konuşma algılayıcı olarak EnergyDetector, ASR olarak sahte nesneler kullanılır.
"""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from live_stream.application.config import AsrParams, ModelRef, WhisperParams
from live_stream.application.live import (
    FinalJob, LanguageLock, LiveSession, PartialJob, Segmenter, SegmenterConfig,
)
from live_stream.infrastructure.catalog import DROP_PHRASES
from live_stream.infrastructure.faster_whisper_asr import FasterWhisperAsr, device_candidates
from live_stream.infrastructure.live_vad import EnergyDetector, make_detector
from live_stream.infrastructure.normalizer import live_normalizer, normalizer_for
from live_stream.infrastructure.recorder import SessionRecorder

SR = 16000


def tone(sec, freq=220.0, amp=0.3):
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(sec):
    return np.zeros(int(sec * SR), dtype=np.float32)


def run(seg, audio, chunk=1600):
    jobs = []
    for i in range(0, len(audio), chunk):
        jobs += seg.feed(audio[i:i + chunk])
    return jobs


def finals_of(jobs):
    return [j for j in jobs if isinstance(j, FinalJob)]


# ------------------------------------------------------------------ cümle sonu tespiti
def test_duraklamada_boler_ve_zamanlar_dogru():
    seg = Segmenter(EnergyDetector(), SegmenterConfig(partial_interval_sec=0))
    audio = np.concatenate([silence(0.5), tone(1.5), silence(1.0), tone(2.0), silence(1.0)])
    f = finals_of(run(seg, audio))
    assert len(f) == 2
    assert f[0].start == pytest.approx(0.5, abs=0.35) and f[0].end == pytest.approx(2.0, abs=0.35)
    assert f[1].start == pytest.approx(3.0, abs=0.35) and f[1].end == pytest.approx(5.0, abs=0.35)


def test_kisa_duraklama_cumleyi_bolmez():
    seg = Segmenter(EnergyDetector(), SegmenterConfig(end_silence_ms=700, partial_interval_sec=0))
    audio = np.concatenate([tone(1.0), silence(0.3), tone(1.0), silence(1.0)])
    assert len(finals_of(run(seg, audio))) == 1


def test_konusurken_ara_metin():
    seg = Segmenter(EnergyDetector(), SegmenterConfig(partial_interval_sec=1.0))
    jobs = run(seg, np.concatenate([tone(3.5), silence(1.0)]))
    assert sum(isinstance(j, PartialJob) for j in jobs) >= 2
    assert isinstance(jobs[-1], FinalJob)


def test_uzun_konusma_zorla_bolunur():
    seg = Segmenter(EnergyDetector(), SegmenterConfig(max_utterance_sec=3.0, partial_interval_sec=0))
    f = finals_of(run(seg, tone(7.0))) + seg.flush()
    assert len(f) >= 2 and all(len(j.audio) / SR <= 3.05 for j in f)


def test_tik_sesi_yok_sayilir():
    seg = Segmenter(EnergyDetector(), SegmenterConfig(partial_interval_sec=0))
    assert run(seg, np.concatenate([silence(0.5), tone(0.12), silence(1.0)])) == []


def test_model_yoksa_enerji_algilayiciya_duser(tmp_path):
    assert isinstance(make_detector("silero", str(tmp_path / "yok.onnx")), EnergyDetector)
    assert isinstance(make_detector("energy", "x"), EnergyDetector)


# --------------------------------------------------------------- oturum kaydı
def test_oturum_ses_ve_metni_diske_yazar(tmp_path):
    rec = SessionRecorder(tmp_path, name="deneme")
    s = LiveSession(Segmenter(EnergyDetector(), SegmenterConfig(partial_interval_sec=0)),
                    lambda a, fast, prompt: "merhaba", rec)
    audio = np.concatenate([tone(1.5), silence(1.0)])
    pcm = (audio * 32767).astype("<i2").tobytes()
    out = [s.run_final(j) for i in range(0, len(pcm), 3200) for j in s.feed_pcm16(pcm[i:i + 3200])]
    path = s.close()
    assert out[0]["text"] == "merhaba" and s.lines == 1
    assert path.endswith("deneme.wav")
    data, sr = sf.read(tmp_path / "deneme.wav")
    assert sr == SR and len(data) == len(audio)
    assert "merhaba" in (tmp_path / "deneme.txt").read_text(encoding="utf-8")
    assert json.loads((tmp_path / "deneme.jsonl").read_text(encoding="utf-8").splitlines()[0])["id"] == 0


def test_onceki_cumleler_istem_olarak_verilir():
    prompts = []

    def tr(audio, fast, prompt):
        prompts.append(prompt)
        return f"cümle {len(prompts)}"

    s = LiveSession(Segmenter(EnergyDetector()), tr)
    job = FinalJob(audio=tone(1.0), start=0.0, end=1.0)
    s.run_final(job)
    s.run_final(job)
    assert prompts == [None, "cümle 1"]


# ------------------------------------------------- canlı metin (faster-whisper, sahte model)
REF = ModelRef("faster-whisper-small", "faster-whisper", "models/faster-whisper-small", {})


class TextModel:
    """transcribe_text'in çağırdığı model.transcribe yerine geçer."""

    def __init__(self, texts):
        self.texts, self.calls = texts, []

    def transcribe(self, samples, **opt):
        self.calls.append(opt)
        return iter([NS(text=t, words=None) for t in self.texts]), None


def live_engine(texts):
    params = AsrParams(drop_phrases=DROP_PHRASES["tr"], whisper=WhisperParams())
    m = TextModel(texts)
    return FasterWhisperAsr(REF, params, normalizer_for("tr"), "tr", whisper_model=m, batched=None), m


@pytest.mark.parametrize("text", ["Altyazı M.K.", " İzlediğiniz için teşekkürler.", "", "   "])
def test_canli_uydurma_kaliplar_gosterilmez(text):
    asr, _ = live_engine([text])
    assert asr.transcribe_text(tone(1.0), normalizer=live_normalizer("tr")) == ""


def test_canli_gercek_konusma_ekran_bicimiyle():
    asr, m = live_engine([" Teşekkür ederim.", " Bugün görüntü işleme anlatacağım."])
    out = asr.transcribe_text(tone(1.0), normalizer=live_normalizer("tr"))
    assert out == "Teşekkür ederim. Bugün görüntü işleme anlatacağım."  # büyük harf, noktalama
    assert m.calls[0]["without_timestamps"] and m.calls[0]["beam_size"] == 5


def test_canli_ara_metin_hizli_ve_tekrar_ayiklanir():
    asr, m = live_engine([" ve ve ve ve ve ve"])
    assert asr.transcribe_text(tone(1.0), fast=True, prompt="önceki cümle",
                               normalizer=live_normalizer("tr")) == "ve ve"
    assert m.calls[0]["beam_size"] == 1 and m.calls[0]["initial_prompt"] == "önceki cümle"


def test_cihaz_sirasi():
    assert device_candidates("auto", "auto") == [("cuda", "int8_float16"), ("cpu", "int8")]
    assert device_candidates("cpu", "auto") == [("cpu", "int8")]
    assert device_candidates("cuda", "float16") == [("cuda", "float16")]
    with pytest.raises(ValueError):
        device_candidates("tpu", "auto")


# ------------------------------------------------------------------- oturum dili
class Decision:
    def __init__(self, code, confidence=0.9):
        self.code, self.confidence = code, confidence


def test_dil_secildiyse_tespit_yapilmaz():
    calls = []
    lock = LanguageLock("vi", lambda a: calls.append(1), fallback="tr")
    assert lock.decide(tone(0.5), final=False) == "vi" and not lock.detected and calls == []


def test_otomatik_dil_ilk_uzun_cumleden_bulunur_ve_kilitlenir():
    calls = []

    def detect(audio):
        calls.append(len(audio))
        return Decision("vi", 0.97)

    lock = LanguageLock(None, detect, fallback="tr", min_sec=1.5)
    assert lock.decide(tone(0.8), final=False) is None       # kısa ara metin: henüz karar yok
    assert lock.decide(tone(2.0), final=False) == "vi"       # yeterince uzun: tespit
    assert lock.decide(tone(3.0), final=True) == "vi"        # kilitli: tekrar tespit yok
    assert len(calls) == 1 and lock.detected and lock.confidence == 0.97


def test_otomatik_dil_tespit_edilemezse_varsayilan():
    def boom(audio):
        raise RuntimeError("desteklenmeyen dil")

    lock = LanguageLock(None, boom, fallback="tr")
    assert lock.decide(tone(0.5), final=True) == "tr" and lock.confidence is None


# ------------------------------------------------------------------- uçtan uca WebSocket
class FakeLiveAsr:
    def __init__(self, tag=""):
        self.tag = tag
        self.device = "cpu"

    def transcribe_text(self, audio, fast=False, prompt=None, normalizer=None):
        return "ara metin" if fast else f"{self.tag}cümle {len(audio) // SR}"


def fake_engines():
    from live_stream.interface.live_server import LiveEngine

    return {"tr": LiveEngine(lambda: FakeLiveAsr(), "sahte-tr"),
            "vi": LiveEngine(lambda: FakeLiveAsr("vi:"), "sahte-vi")}


def make_local(tmp_path, monkeypatch, languages=("tr", "vi"), engines=None, detect=None, trackers=None,
               **settings):
    """Yerel sunucu (toplu iş + canlı altyazı), modeller yerine sahte nesnelerle."""
    from live_stream.interface import local_server
    from live_stream.interface.settings import Settings

    from live_stream.application.language import LanguageResolver

    monkeypatch.setattr(local_server, "build_pipeline", lambda cfg, store, **kw: NS(engines={}))
    # dil tespiti modeli (Whisper-tiny) yerine boş: canlı tespit testte 'detect' ile verilir
    monkeypatch.setattr(local_server, "build_resolver", lambda cfg: LanguageResolver(list(cfg.languages), []))
    s = Settings(data_dir=str(tmp_path / "up"), recordings_dir=str(tmp_path / "rec"),
                 languages=list(languages), live_language="tr", **settings)
    engines = engines or {k: v for k, v in fake_engines().items() if k in languages}
    app, store, queue, _ = local_server.build_local(s, workers=1, warm_up=False, live_engines=engines,
                                                    live_detector=EnergyDetector, live_detect=detect,
                                                    live_speakers=trackers)
    sent = []
    queue.enqueue_transcription = lambda job_id, lang: sent.append((job_id, lang))
    queue.enqueue_detection = lambda job_id: sent.append((job_id, "lid"))
    return app, store, sent


@pytest.fixture
def local(tmp_path, monkeypatch):
    return (*make_local(tmp_path, monkeypatch), tmp_path)


def speak(ws, audio):
    pcm = (audio * 32767).astype("<i2").tobytes()
    for i in range(0, len(pcm), 3200):
        ws.send_bytes(pcm[i:i + 3200])


def session(client, url, audio):
    with client.websocket_connect(url) as ws:
        ready = ws.receive_json()
        speak(ws, audio)
        ws.send_text('{"type": "stop"}')
        msgs = []
        while (m := ws.receive_json())["type"] != "done":
            msgs.append(m)
    return ready, msgs, m


TWO = np.concatenate([silence(0.3), tone(1.5), silence(1.0), tone(2.5), silence(0.2)])


def test_canli_oturum_uctan_uca_ve_toplu_ise_gonderilir(local):
    app, store, sent, tmp = local
    with TestClient(app) as client:
        assert "Canlı Altyazı" in client.get("/canli").text
        assert "/canli" in client.get("/").text                     # ana sayfadan bağlantı
        ready, msgs, done = session(client, "/ws/live?partial_sec=0.8", TWO)
        h = client.get("/live/health").json()

    assert ready["type"] == "ready" and ready["recording"] and ready["model"] == "sahte-tr"
    assert [x["code"] for x in h["languages"]] == ["tr", "vi"] and h["default"] == "tr"
    assert h["auto"] is False                         # dil tespiti verilmedi: otomatik seçenek yok
    finals = [m for m in msgs if m["type"] == "final"]
    assert [m["id"] for m in finals] == [0, 1]        # ikinci cümle "stop" ile kesinleşti
    assert not any(m["type"] == "language" for m in msgs)  # dil seçildi: duyuru yok
    assert any(m["type"] == "stats" for m in msgs)
    wavs = list((tmp / "rec").glob("*.wav"))
    assert len(wavs) == 1
    # oturum kaydı konuşmacı ayrımı için oturumun diliyle toplu işe gitti
    assert done["job_id"] and sent == [(done["job_id"], "tr")]
    rec = store.get(done["job_id"])
    assert rec.job.audio_path == str(wavs[0].resolve()) and rec.job.language == "tr"


def test_vietnamca_secilince_vietnamca_model(local):
    app, store, sent, _ = local
    with TestClient(app) as client:
        ready, msgs, done = session(client, "/ws/live?language=vi&partial_sec=0", TWO)
    assert ready["language"] == "vi" and ready["model"] == "sahte-vi"
    assert all(m["text"].startswith("vi:") for m in msgs if m["type"] == "final")
    assert sent == [(done["job_id"], "vi")] and store.get(done["job_id"]).job.language == "vi"


def test_otomatik_dil_ilk_cumleden_bulunur(tmp_path, monkeypatch):
    calls = []

    def detect(audio):
        calls.append(len(audio))
        return Decision("vi", 0.93)

    app, store, sent = make_local(tmp_path, monkeypatch, detect=detect)
    with TestClient(app) as client:
        assert client.get("/live/health").json()["auto"] is True
        ready, msgs, done = session(client, "/ws/live?language=auto&partial_sec=0", TWO)
    assert ready["language"] == "auto" and ready["model"] is None
    lang = [m for m in msgs if m["type"] == "language"]
    assert lang == [{"type": "language", "code": "vi", "confidence": 0.93, "model": "sahte-vi", "device": "cpu"}]
    assert msgs.index(lang[0]) < next(i for i, m in enumerate(msgs) if m["type"] == "final")
    assert [m["text"][:3] for m in msgs if m["type"] == "final"] == ["vi:", "vi:"]
    assert len(calls) == 1                                   # oturumda bir kez
    assert sent == [(done["job_id"], "vi")]


def test_tek_dil_acikken_otomatik_o_dildir(tmp_path, monkeypatch):
    app, _, sent = make_local(tmp_path, monkeypatch, languages=("tr",))
    with TestClient(app) as client:
        assert client.get("/live/health").json()["auto"] is False
        ready, _, done = session(client, "/ws/live?language=auto", TWO)
    assert ready["language"] == "tr" and sent == [(done["job_id"], "tr")]


def test_acik_olmayan_dil_reddedilir(tmp_path, monkeypatch):
    app, *_ = make_local(tmp_path, monkeypatch, languages=("tr",))
    with TestClient(app) as client, client.websocket_connect("/ws/live?language=vi") as ws:
        m = ws.receive_json()
    assert m["type"] == "error" and "vi" in m["message"]


def test_analiz_istenmezse_toplu_is_acilmaz(local):
    app, _, sent, _ = local
    with TestClient(app) as client:  # eski uç adı da çalışır
        _, _, done = session(client, "/ws?analyze=0", np.concatenate([tone(1.5), silence(1.0)]))
    assert done["job_id"] is None and sent == []


def test_konusma_yoksa_toplu_is_acilmaz(local):
    app, _, sent, _ = local
    with TestClient(app) as client:
        _, _, done = session(client, "/ws/live", silence(1.0))
    assert done["job_id"] is None and sent == []


def test_model_yuklenemezse_hata_mesaji(tmp_path, monkeypatch):
    from live_stream.interface.live_server import LiveEngine

    def bozuk():
        raise RuntimeError("CUDA yok")

    app, *_ = make_local(tmp_path, monkeypatch, languages=("tr",),
                         engines={"tr": LiveEngine(bozuk, "x")})
    with TestClient(app) as client, client.websocket_connect("/ws/live") as ws:
        m = ws.receive_json()
    assert m["type"] == "error" and "CUDA yok" in m["message"]


# ------------------------------------------------------------- oturum sonu bildirimi (webhook)
HOOK = "https://alici.ornek.com/hook"


def outbox(store, event=None):
    return [r for r in store.outbox_rows() if event is None or r["event"] == event]


def test_oturum_bitince_live_ended_ve_toplu_ise_ayni_adres(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE_WEBHOOK_URL", HOOK)
    app, store, sent = make_local(tmp_path, monkeypatch)
    with TestClient(app) as client:
        assert client.get("/live/health").json()["webhook"] is True
        _, msgs, done = session(client, "/ws/live?partial_sec=0", TWO)

    assert done["webhook"] is True and done["session_id"] and done["job_id"]
    [row] = outbox(store, "live.ended")
    p = row["payload"]
    assert row["url"] == HOOK and row["job_id"] == done["job_id"]
    assert p["event"] == "live.ended" and p["session_id"] == done["session_id"] and p["reason"] == "stop"
    assert p["language"] == {"code": "tr", "source": "secim", "confidence": 1.0}
    finals = [m for m in msgs if m["type"] == "final"]
    assert [ln["text"] for ln in p["lines"]] == [m["text"] for m in finals]
    assert p["lines"][0] == {"id": 0, "start": finals[0]["start"], "end": finals[0]["end"], "text": finals[0]["text"]}
    assert p["text"] == " ".join(m["text"] for m in finals) and p["job_id"] == done["job_id"]
    assert p["duration"] == pytest.approx(len(TWO) / SR, abs=0.1) and p["started_at"] <= p["ended_at"]
    assert p["recording"]
    # konuşmacı ayrımı aynı adrese 'job.done' gönderecek: iş aynı callback ve oturum kimliğiyle açıldı
    job = store.get(done["job_id"]).job
    assert job.callback_url == HOOK and job.external_id == f"live:{done['session_id']}"


def test_oturum_basina_callback_adresi_ve_analizsiz_oturum(tmp_path, monkeypatch):
    app, store, sent = make_local(tmp_path, monkeypatch, api_key="k")   # LIVE_WEBHOOK_URL yok
    with TestClient(app) as client:
        _, _, done = session(client, f"/ws/live?key=k&analyze=0&callback_url={HOOK}", TWO)
    assert done["webhook"] is True and done["job_id"] is None and sent == []
    [row] = outbox(store, "live.ended")
    assert row["url"] == HOOK and row["job_id"] is None and row["payload"]["job_id"] is None


def test_adres_yoksa_bildirim_yok(local):
    app, store, _, _ = local
    with TestClient(app) as client:
        assert client.get("/live/health").json()["webhook"] is False
        _, _, done = session(client, "/ws/live", TWO)
    assert done["webhook"] is False and outbox(store) == []


def test_gecersiz_callback_reddedilir(tmp_path, monkeypatch):
    app, store, _ = make_local(tmp_path, monkeypatch, api_key="k")
    with TestClient(app) as client, client.websocket_connect("/ws/live?key=k&callback_url=ftp://x/y") as ws:
        m = ws.receive_json()
    assert m["type"] == "error" and "http" in m["message"] and outbox(store) == []


def test_anahtarsiz_sunucuda_oturum_basina_adres_verilemez(local):
    """API_KEY yokken herkes bağlanabilir: sunucuya istediği adrese istek attıramasın."""
    app, store, _, _ = local
    with TestClient(app) as client, client.websocket_connect(f"/ws/live?callback_url={HOOK}") as ws:
        m = ws.receive_json()
    assert m["type"] == "error" and "API_KEY" in m["message"] and outbox(store) == []


def test_baglanti_koparsa_da_bildirilir(tmp_path, monkeypatch):
    import time

    monkeypatch.setenv("LIVE_WEBHOOK_URL", HOOK)
    app, store, sent = make_local(tmp_path, monkeypatch, live_resume_sec=0.3)  # geri dönülmezse kapanır
    with TestClient(app) as client:
        with client.websocket_connect("/ws/live?partial_sec=0") as ws:
            ws.receive_json()
            speak(ws, np.concatenate([tone(1.5), silence(1.0)]))
            time.sleep(0.3)                           # son paket işlensin; "stop" gönderilmeden çık
        for _ in range(50):                           # sunucu kopmayı arka planda işler
            if outbox(store, "live.ended"):
                break
            time.sleep(0.1)
    [row] = outbox(store, "live.ended")
    assert row["payload"]["reason"] == "disconnect" and row["payload"]["lines"]
    assert row["job_id"] and len(sent) == 1           # kayıt yine de konuşmacı ayrımına gitti


def test_otomatik_dilde_bildirimde_tespit_bilgisi(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE_WEBHOOK_URL", HOOK)
    app, store, _ = make_local(tmp_path, monkeypatch, detect=lambda a: Decision("vi", 0.91))
    with TestClient(app) as client:
        session(client, "/ws/live?language=auto&partial_sec=0", TWO)
    [row] = outbox(store, "live.ended")
    assert row["payload"]["language"] == {"code": "vi", "source": "tespit", "confidence": 0.91}


# ------------------------------------------------------------------- erişim anahtarı
def test_anahtar_tanimliysa_canli_baglanti_ister(tmp_path, monkeypatch):
    app, _, _ = make_local(tmp_path, monkeypatch, api_key="gizli")
    with TestClient(app) as client:
        assert client.get("/live/health").json()["auth"] is True
        for url in ("/ws/live", "/ws/live?key=yanlis"):
            with client.websocket_connect(url) as ws:
                m = ws.receive_json()
            assert m["type"] == "error" and "anahtar" in m["message"]
        with client.websocket_connect("/ws/live?key=gizli") as ws:          # sorgu parametresi
            assert ws.receive_json()["type"] == "ready"
        with client.websocket_connect("/ws/live", headers={"X-API-Key": "gizli"}) as ws:  # başlık
            assert ws.receive_json()["type"] == "ready"


# ------------------------------------------------------------------- kopunca devam
def wait_for(cond, sec=5.0):
    import time

    for _ in range(int(sec / 0.05)):
        if cond():
            return True
        time.sleep(0.05)
    return cond()


def test_kopan_oturum_ayni_kayit_ve_satirlarla_devam_eder(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE_WEBHOOK_URL", HOOK)
    app, store, sent = make_local(tmp_path, monkeypatch, live_resume_sec=5)
    first = np.concatenate([tone(1.5), silence(1.0)])
    second = np.concatenate([tone(2.5), silence(1.0)])
    with TestClient(app) as client:
        with client.websocket_connect("/ws/live?partial_sec=0") as ws:
            ready = ws.receive_json()
            assert ready["resumed"] is False and ready["resume_token"]
            speak(ws, first)
            got = [ws.receive_json() for _ in range(2)]            # final + stats
        sid, tok = ready["session_id"], ready["resume_token"]
        assert [m["type"] for m in got] == ["final", "stats"] and got[0]["id"] == 0
        assert wait_for(lambda: sid in app.state.live_runs and app.state.live_runs[sid].ws is None)
        assert outbox(store) == [] and sent == []                   # henüz kapanmadı

        _, msgs, done = session(client, f"/ws/live?resume={sid}&token={tok}&after=0&partial_sec=0", second)
    finals = [m for m in msgs if m["type"] == "final"]
    assert [m["id"] for m in finals] == [1]                         # numaralar sürüyor, tekrar yok
    assert finals[0]["start"] == pytest.approx(len(first) / SR, abs=0.35)  # zaman da sürüyor
    assert done["session_id"] == sid
    wavs = list((tmp_path / "rec").glob("*.wav"))
    assert len(wavs) == 1                                           # tek kayıt
    data, _ = sf.read(wavs[0])
    assert len(data) == len(first) + len(second)
    [row] = outbox(store, "live.ended")                             # tek bildirim, tek toplu iş
    assert [ln["id"] for ln in row["payload"]["lines"]] == [0, 1] and row["payload"]["reason"] == "stop"
    assert len(sent) == 1


def test_kopukken_kesinlesen_satir_donunce_gonderilir(tmp_path, monkeypatch):
    app, _, _ = make_local(tmp_path, monkeypatch, live_resume_sec=5)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/live?partial_sec=0") as ws:
            ready = ws.receive_json()
            speak(ws, np.concatenate([tone(1.5), silence(1.0)]))
        sid, tok = ready["session_id"], ready["resume_token"]
        run = app.state.live_runs[sid]
        assert wait_for(lambda: run.session.lines == 1)            # istemci bu satırı almadı
        with client.websocket_connect(f"/ws/live?resume={sid}&token={tok}&after=-1") as ws:
            assert ws.receive_json()["resumed"] is True
            m = ws.receive_json()
            ws.send_text('{"type": "stop"}')
            while ws.receive_json()["type"] != "done":
                pass
    assert m["type"] == "final" and m["id"] == 0


def test_yanlis_belirtecle_devam_edilemez_yeni_oturum_acilir(tmp_path, monkeypatch):
    app, _, _ = make_local(tmp_path, monkeypatch, live_resume_sec=5)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/live") as ws:
            sid = ws.receive_json()["session_id"]
        with client.websocket_connect(f"/ws/live?resume={sid}&token=baskasi") as ws:
            r = ws.receive_json()
        assert r["resumed"] is False and r["session_id"] != sid
        with client.websocket_connect("/ws/live?resume=yok&token=x") as ws:
            assert ws.receive_json()["resumed"] is False


def test_sure_dolunca_oturum_kapanir_ve_bildirilir(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE_WEBHOOK_URL", HOOK)
    app, store, sent = make_local(tmp_path, monkeypatch, live_resume_sec=0.3)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/live?partial_sec=0") as ws:
            ready = ws.receive_json()
            speak(ws, np.concatenate([tone(1.5), silence(1.0)]))
            ws.receive_json()
        assert wait_for(lambda: outbox(store, "live.ended"))
        with client.websocket_connect(f"/ws/live?resume={ready['session_id']}&token={ready['resume_token']}") as ws:
            assert ws.receive_json()["resumed"] is False          # süre doldu: yeni oturum
    [row] = outbox(store, "live.ended")
    assert row["payload"]["reason"] == "disconnect" and len(sent) == 1


def test_eski_canli_kayitlar_silinir(tmp_path):
    import os
    import time

    from live_stream.infrastructure.recorder import delete_old_recordings

    now = time.time()
    for name, age_days in (("eski.wav", 40), ("eski.txt", 40), ("eski.jsonl", 40),
                           ("yeni.wav", 1), ("not.md", 40)):
        f = tmp_path / name
        f.write_text("x")
        os.utime(f, (now - age_days * 86400,) * 2)
    assert delete_old_recordings(tmp_path, 0, now) == 0                  # 0 = silme
    assert delete_old_recordings(tmp_path, 30, now) == 3
    assert sorted(f.name for f in tmp_path.iterdir()) == ["not.md", "yeni.wav"]
    assert delete_old_recordings(tmp_path / "yok", 30, now) == 0


# ------------------------------------------------------------------- canlı konuşmacı etiketi
def unit(*v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def test_konusmaci_izleyici_yakin_imza_ayni_kisi_uzak_yeni_kisi():
    from live_stream.application.live import SpeakerTracker

    seq = iter([unit(1, 0, 0), unit(0.95, 0.1, 0), unit(0, 1, 0), unit(0.9, 0.2, 0), unit(0, 0.9, 0.2)])
    t = SpeakerTracker(lambda a: next(seq), threshold=0.5, min_sec=1.0)
    labels = [t.assign(tone(2.0)) for _ in range(5)]
    assert labels == ["speaker_00", "speaker_00", "speaker_01", "speaker_00", "speaker_01"]
    assert t.count == 2


def test_kisa_cumle_onceki_konusmaciya_ve_ust_sinir():
    from live_stream.application.live import SpeakerTracker

    calls = []

    def embed(a):
        calls.append(1)
        return unit(*np.eye(3)[len(calls) % 3])

    t = SpeakerTracker(embed, threshold=0.5, min_sec=1.0, max_speakers=2)
    assert t.assign(tone(0.5)) is None                   # henüz kimse yok, kısa: etiket yok
    assert t.assign(tone(2.0)) == "speaker_00"
    assert t.assign(tone(0.4)) == "speaker_00" and len(calls) == 1   # kısa: imza çıkarılmaz
    assert t.assign(tone(2.0)) == "speaker_01"
    assert t.assign(tone(2.0)) in ("speaker_00", "speaker_01") and t.count == 2  # üst sınır


def test_oturum_satirinda_konusmaci_ve_kayitta_etiket(tmp_path):
    rec = SessionRecorder(tmp_path, name="k")
    who = iter(["speaker_00", "speaker_01"])
    s = LiveSession(Segmenter(EnergyDetector()), lambda a, f, p: "merhaba", rec, speaker_of=lambda a: next(who))
    job = FinalJob(audio=tone(1.5), start=0.0, end=1.5)
    assert s.run_final(job)["speaker"] == "speaker_00" and s.run_final(job)["speaker"] == "speaker_01"
    s.close()
    assert s.items[1] == {"id": 1, "start": 0.0, "end": 1.5, "text": "merhaba", "speaker": "speaker_01"}
    assert "Konuşmacı 2: merhaba" in (tmp_path / "k.txt").read_text(encoding="utf-8")


def test_imza_hatasi_metni_engellemez():
    def boom(a):
        raise RuntimeError("model yok")

    s = LiveSession(Segmenter(EnergyDetector()), lambda a, f, p: "selam", speaker_of=boom)
    m = s.run_final(FinalJob(audio=tone(1.5), start=0.0, end=1.5))
    assert m["text"] == "selam" and m["speaker"] is None


def test_canli_konusmaci_etiketi_uctan_uca(tmp_path, monkeypatch):
    from live_stream.application.live import SpeakerTracker

    monkeypatch.setenv("LIVE_WEBHOOK_URL", HOOK)
    # sahte imza: sesin perdesine göre (220 Hz ve 440 Hz iki "kişi")
    def embed(a):
        f = np.abs(np.fft.rfft(a[:16000]))
        return unit(f[200:240].sum(), f[420:460].sum(), 1e-3)

    app, store, _ = make_local(tmp_path, monkeypatch, trackers=lambda: SpeakerTracker(embed))
    audio = np.concatenate([tone(1.5, 220), silence(1.0), tone(1.5, 440), silence(1.0),
                            tone(1.5, 220), silence(1.0)])
    with TestClient(app) as client:
        assert client.get("/live/health").json()["speakers"] is True
        _, msgs, _ = session(client, "/ws/live?partial_sec=0", audio)
        _, msgs_off, _ = session(client, "/ws/live?partial_sec=0&speakers=0", audio[:SR * 3])
    assert [m["speaker"] for m in msgs if m["type"] == "final"] == ["speaker_00", "speaker_01", "speaker_00"]
    assert all("speaker" not in m for m in msgs_off if m["type"] == "final")
    row = outbox(store, "live.ended")[0]
    assert [ln["speaker"] for ln in row["payload"]["lines"]] == ["speaker_00", "speaker_01", "speaker_00"]
