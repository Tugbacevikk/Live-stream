import json

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from live_stream.config import Settings
from live_stream.recorder import SessionRecorder
from live_stream.segmenter import FinalJob, PartialJob, Segmenter, SegmenterConfig
from live_stream.server import create_app
from live_stream.session import LiveSession
from live_stream.transcriber import is_hallucination
from live_stream.vad import EnergyDetector

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


# ------------------------------------------------------------------ segmenter
def test_splits_on_pause_with_correct_times():
    seg = Segmenter(EnergyDetector(), SegmenterConfig(partial_interval_sec=0))
    audio = np.concatenate([silence(0.5), tone(1.5), silence(1.0), tone(2.0), silence(1.0)])
    f = finals_of(run(seg, audio))
    assert len(f) == 2
    assert f[0].start == pytest.approx(0.5, abs=0.35) and f[0].end == pytest.approx(2.0, abs=0.35)
    assert f[1].start == pytest.approx(3.0, abs=0.35) and f[1].end == pytest.approx(5.0, abs=0.35)


def test_short_pause_does_not_split_sentence():
    seg = Segmenter(EnergyDetector(), SegmenterConfig(end_silence_ms=700, partial_interval_sec=0))
    audio = np.concatenate([tone(1.0), silence(0.3), tone(1.0), silence(1.0)])
    assert len(finals_of(run(seg, audio))) == 1


def test_partials_while_speaking():
    seg = Segmenter(EnergyDetector(), SegmenterConfig(partial_interval_sec=1.0))
    jobs = run(seg, np.concatenate([tone(3.5), silence(1.0)]))
    assert sum(isinstance(j, PartialJob) for j in jobs) >= 2
    assert isinstance(jobs[-1], FinalJob)


def test_long_speech_force_cut():
    seg = Segmenter(EnergyDetector(), SegmenterConfig(max_utterance_sec=3.0, partial_interval_sec=0))
    f = finals_of(run(seg, tone(7.0))) + seg.flush()
    assert len(f) >= 2 and all(len(j.audio) / SR <= 3.05 for j in f)


def test_click_ignored():
    seg = Segmenter(EnergyDetector(), SegmenterConfig(partial_interval_sec=0))
    assert run(seg, np.concatenate([silence(0.5), tone(0.12), silence(1.0)])) == []


# -------------------------------------------------------------- hallucination
@pytest.mark.parametrize("text", ["Altyazı M.K.", "İzlediğiniz için teşekkürler.", "", "  ", "ve ve ve ve ve ve ve"])
def test_hallucinations_filtered(text):
    assert is_hallucination(text)


@pytest.mark.parametrize("text", ["Teşekkür ederim.", "Bugün görüntü işleme üzerine konuşacağım."])
def test_real_speech_kept(text):
    assert not is_hallucination(text)


# --------------------------------------------------------------- recorder
def test_session_records_audio_and_text(tmp_path):
    rec = SessionRecorder(tmp_path, name="deneme")
    s = LiveSession(Segmenter(EnergyDetector(), SegmenterConfig(partial_interval_sec=0)),
                    lambda a, p, pr: "merhaba", rec)
    audio = np.concatenate([tone(1.5), silence(1.0)])
    pcm = (audio * 32767).astype("<i2").tobytes()
    out = [s.run_final(j) for i in range(0, len(pcm), 3200) for j in s.feed_pcm16(pcm[i:i + 3200])]
    s.close()
    assert out[0]["text"] == "merhaba"
    data, sr = sf.read(tmp_path / "deneme.wav")
    assert sr == SR and len(data) == len(audio)
    assert "merhaba" in (tmp_path / "deneme.txt").read_text(encoding="utf-8")
    assert json.loads((tmp_path / "deneme.jsonl").read_text(encoding="utf-8").splitlines()[0])["id"] == 0


# --------------------------------------------------------------- websocket
class FakeTranscriber:
    device = "cpu"
    ready = True

    def load(self):
        pass

    def transcribe(self, audio, partial, prompt=None):
        return "ara metin" if partial else f"cümle {len(audio) // SR}"


def test_websocket_end_to_end(tmp_path):
    settings = Settings(recordings_dir=str(tmp_path))
    app = create_app(settings, transcriber=FakeTranscriber(), detector_factory=EnergyDetector)
    audio = np.concatenate([silence(0.3), tone(1.5), silence(1.0), tone(2.5), silence(0.2)])
    pcm = (audio * 32767).astype("<i2").tobytes()

    with TestClient(app) as client:
        assert client.get("/health").json()["ready"] is True
        assert "Canlı Altyazı" in client.get("/").text
        with client.websocket_connect("/ws?partial_sec=0.8") as ws:
            ready = ws.receive_json()
            assert ready["type"] == "ready" and ready["recording"]
            for i in range(0, len(pcm), 3200):
                ws.send_bytes(pcm[i:i + 3200])
            ws.send_text('{"type": "stop"}')
            msgs = []
            while (m := ws.receive_json())["type"] != "done":
                msgs.append(m)

    finals = [m for m in msgs if m["type"] == "final"]
    assert [m["id"] for m in finals] == [0, 1]        # ikinci cümle "stop" ile kesinleşti
    assert any(m["type"] == "stats" for m in msgs)
    assert len(list(tmp_path.glob("*.wav"))) == 1
