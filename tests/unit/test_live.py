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
from live_stream.application.live import FinalJob, LiveSession, PartialJob, Segmenter, SegmenterConfig
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


# ------------------------------------------------------------------- uçtan uca WebSocket
class FakeLiveAsr:
    device = "cpu"

    def transcribe_text(self, audio, fast=False, prompt=None, normalizer=None):
        return "ara metin" if fast else f"cümle {len(audio) // SR}"


@pytest.fixture
def local(tmp_path, monkeypatch):
    """Yerel sunucu (toplu iş + canlı altyazı), modeller yerine sahte nesnelerle."""
    from live_stream.interface import local_server
    from live_stream.interface.live_server import LiveEngine
    from live_stream.interface.settings import Settings

    monkeypatch.setattr(local_server, "build_pipeline", lambda cfg, store, **kw: NS(engines={}))
    s = Settings(data_dir=str(tmp_path / "up"), recordings_dir=str(tmp_path / "rec"), languages=["tr"])
    engine = LiveEngine(lambda: FakeLiveAsr(), "sahte")
    app, store, queue, _ = local_server.build_local(s, workers=1, warm_up=False, live_engine=engine,
                                                    live_detector=EnergyDetector)
    sent = []
    queue.enqueue_transcription = lambda job_id, lang: sent.append((job_id, lang))
    return app, store, sent, tmp_path


def speak(ws, audio):
    pcm = (audio * 32767).astype("<i2").tobytes()
    for i in range(0, len(pcm), 3200):
        ws.send_bytes(pcm[i:i + 3200])


def test_canli_oturum_uctan_uca_ve_toplu_ise_gonderilir(local):
    app, store, sent, tmp = local
    audio = np.concatenate([silence(0.3), tone(1.5), silence(1.0), tone(2.5), silence(0.2)])
    with TestClient(app) as client:
        assert "Canlı Altyazı" in client.get("/canli").text
        assert "/canli" in client.get("/").text                     # ana sayfadan bağlantı
        with client.websocket_connect("/ws/live?partial_sec=0.8") as ws:
            ready = ws.receive_json()
            assert ready["type"] == "ready" and ready["recording"] and ready["model"] == "sahte"
            speak(ws, audio)
            ws.send_text('{"type": "stop"}')
            msgs = []
            while (m := ws.receive_json())["type"] != "done":
                msgs.append(m)
        done = m
        assert client.get("/live/health").json()["ready"] is True

    finals = [m for m in msgs if m["type"] == "final"]
    assert [m["id"] for m in finals] == [0, 1]        # ikinci cümle "stop" ile kesinleşti
    assert any(m["type"] == "stats" for m in msgs)
    wavs = list((tmp / "rec").glob("*.wav"))
    assert len(wavs) == 1
    # oturum kaydı konuşmacı ayrımı için toplu işe gitti
    assert done["job_id"] and sent == [(done["job_id"], "tr")]
    rec = store.get(done["job_id"])
    assert rec.job.audio_path == str(wavs[0].resolve()) and rec.job.language == "tr"


def test_analiz_istenmezse_toplu_is_acilmaz(local):
    app, _, sent, _ = local
    with TestClient(app) as client, client.websocket_connect("/ws?analyze=0") as ws:  # eski uç adı
        ws.receive_json()
        speak(ws, np.concatenate([tone(1.5), silence(1.0)]))
        ws.send_text('{"type": "stop"}')
        while (m := ws.receive_json())["type"] != "done":
            pass
    assert m["job_id"] is None and sent == []


def test_konusma_yoksa_toplu_is_acilmaz(local):
    app, _, sent, _ = local
    with TestClient(app) as client, client.websocket_connect("/ws/live") as ws:
        ws.receive_json()
        speak(ws, silence(1.0))
        ws.send_text('{"type": "stop"}')
        while (m := ws.receive_json())["type"] != "done":
            pass
    assert m["job_id"] is None and sent == []


def test_model_yuklenemezse_hata_mesaji(tmp_path, monkeypatch):
    from live_stream.interface import local_server
    from live_stream.interface.live_server import LiveEngine
    from live_stream.interface.settings import Settings

    def bozuk():
        raise RuntimeError("CUDA yok")

    monkeypatch.setattr(local_server, "build_pipeline", lambda cfg, store, **kw: NS(engines={}))
    s = Settings(data_dir=str(tmp_path / "up"), recordings_dir=str(tmp_path / "rec"), languages=["tr"])
    app, *_ = local_server.build_local(s, workers=1, warm_up=False,
                                       live_engine=LiveEngine(bozuk, "x"), live_detector=EnergyDetector)
    with TestClient(app) as client, client.websocket_connect("/ws/live") as ws:
        m = ws.receive_json()
    assert m["type"] == "error" and "CUDA yok" in m["message"]
