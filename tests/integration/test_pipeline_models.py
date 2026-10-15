"""Gerçek modellerle uçtan uca testler (Türkçe, Vietnamca). Gereken model yoksa test atlanır.

    python scripts/download_models.py                      # faster-whisper-medium + Zipformer + yardımcılar
    python scripts/download_models.py --tr all             # tüm Türkçe adaylar
    pytest -m models
    MODELS_DIR=D:/modeller pytest -m models                # modeller başka klasördeyse
"""
import os
from pathlib import Path

import numpy as np
import pytest

M = Path(os.environ.get("MODELS_DIR") or Path(__file__).resolve().parents[2] / "models")
LIDW = M / "spoken-language-identification-test-wavs"
Z68 = M / "sherpa-onnx-zipformer-vi-int8-2025-04-20"
TINY = M / "sherpa-onnx-whisper-tiny"
SMALL = M / "sherpa-onnx-whisper-small"
TURBO = M / "sherpa-onnx-whisper-turbo"
OMNI = M / "sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12"
VAD = M / "silero_vad.onnx"
CAMPP = M / "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
TR_WAV = LIDW / "tr-turkish.wav"
FW_MODELS = [m for m in ("faster-whisper-medium", "faster-whisper-small") if (M / m / "model.bin").exists()]


def need(*paths):
    missing = [p for p in paths if not Path(p).exists()]
    return pytest.mark.skipif(bool(missing), reason=f"model yok: {[str(p) for p in missing]}")


pytestmark = pytest.mark.models
VI_OK = need(Z68 / "tokens.txt", M / "silero_vad.onnx")
LID_OK = need(TINY / "tiny-encoder.int8.onnx", SMALL / "small-encoder.int8.onnx",
              LIDW / "tr-turkish.wav", Z68 / "test_wavs")
DIAR = [M / "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"]


def load(p):
    from live_stream.infrastructure.audio import SoundfileLoader

    return SoundfileLoader().load(str(p))


def vi_wavs():
    return [load(Z68 / "test_wavs" / f"{i}.wav") for i in range(3)]


def tr_asr_model():
    """Elde olan en iyi Türkçe model: faster-whisper, yoksa Omnilingual."""
    if FW_MODELS:
        return FW_MODELS[0]
    return "omnilingual-300m" if (OMNI / "model.int8.onnx").exists() else None


TR_MODEL = tr_asr_model()
HAS_TR = pytest.mark.skipif(TR_MODEL is None, reason="Türkçe ASR modeli yok")


@need(VAD, TR_WAV)
def test_vad_ofseti():
    from live_stream.application.config import VadConfig
    from live_stream.infrastructure.sherpa_vad import SileroVad

    audio = np.concatenate([np.zeros(16000 * 3, dtype=np.float32), load(TR_WAV)])
    segs = SileroVad(VadConfig(model=str(VAD))).split(audio)
    assert segs and 2.5 < segs[0].start < 3.6


@need(TURBO / "turbo-encoder.int8.onnx", TR_WAV, VAD)
def test_turkce_whisper_ve_uydurma_filtresi():
    from live_stream.domain import VadSegment
    from live_stream.infrastructure.catalog import language_profile
    from live_stream.infrastructure.normalizer import TrNormalizer
    from live_stream.infrastructure.sherpa_asr import build_asr

    asr = build_asr(language_profile("tr", M, "whisper-turbo"), TrNormalizer())
    words = asr.transcribe([VadSegment(4.0, load(TR_WAV))])
    text = " ".join(w.text for w in words).lower()
    assert "hayal" in text and "yapabilirsin" in text
    assert words[0].start >= 4.0 and words[-1].end <= 4.0 + len(load(TR_WAV)) / 16000 + 1e-6
    # 10 sn sessizlikte Whisper "İzlediğiniz için teşekkür ederim." uyduruyor -> atılmalı
    assert asr.transcribe([VadSegment(0.0, np.zeros(16000 * 10, dtype=np.float32))]) == []


@need(OMNI / "model.int8.onnx", TR_WAV)
def test_turkce_omnilingual_gercek_zaman_damgasi():
    from live_stream.domain import VadSegment
    from live_stream.infrastructure.catalog import language_profile
    from live_stream.infrastructure.normalizer import TrNormalizer
    from live_stream.infrastructure.sherpa_asr import build_asr

    a = load(TR_WAV)
    words = build_asr(language_profile("tr", M, "omnilingual-300m"), TrNormalizer()).transcribe([VadSegment(4.0, a)])
    assert [w.text for w in words] == ["hayal", "edebiliyorsan", "yapabilirsin"]
    assert 4.0 <= words[0].start < words[1].start < words[2].start < 4.0 + len(a) / 16000


@pytest.mark.skipif(not FW_MODELS, reason="faster-whisper modeli yok")
@need(TR_WAV)
@pytest.mark.parametrize("fw", FW_MODELS or ["yok"])
def test_turkce_faster_whisper_kelime_zamani_ve_canli_metin(fw):
    """Toplu iş (kelime zamanlı) ve canlı altyazı (ekran biçimi) aynı model nesnesiyle."""
    from live_stream.domain import VadSegment
    from live_stream.infrastructure.catalog import language_profile
    from live_stream.infrastructure.normalizer import TrNormalizer, live_normalizer
    from live_stream.infrastructure.sherpa_asr import build_asr

    a = load(TR_WAV)
    asr = build_asr(language_profile("tr", M, fw), TrNormalizer(lowercase=True, strip_punct=True))
    for eng in (asr, _unbatched(asr)):
        words = eng.transcribe([VadSegment(4.0, a)])
        assert " ".join(w.text for w in words) == "hayal edebiliyorsan yapabilirsin"
        assert 4.0 <= words[0].start < words[1].start < words[2].start < 4.0 + len(a) / 16000
    live = asr.transcribe_text(a, normalizer=live_normalizer("tr"))
    assert "edebiliyorsan" in live.lower() and live[0].isupper()          # ekran biçimi
    assert asr.transcribe_text(a, fast=True, normalizer=live_normalizer("tr"))  # ara metin
    assert asr.transcribe_text(np.zeros(16000 * 5, np.float32), normalizer=live_normalizer("tr")) == ""


def _unbatched(asr):
    import copy

    e = copy.copy(asr)
    e.batched = None  # tek tek çözme yolu da aynı sonucu vermeli
    return e


@HAS_TR
@need(VAD, CAMPP, TR_WAV)
def test_uctan_uca_toplu_is(tmp_path):
    import soundfile as sf

    from live_stream.domain import AudioJob, LanguageSource
    from live_stream.interface.cli import MemoryRepo, build, default_config

    tr = load(TR_WAV)
    gap = np.zeros(8000, dtype=np.float32)
    p = tmp_path / "konusma.wav"
    sf.write(p, np.concatenate([tr, gap, tr, gap, tr]), 16000)

    uc = build(default_config(M, ("tr",), models={"tr": TR_MODEL}, device="cpu"), MemoryRepo())
    t = uc.run(AudioJob("e2e", str(p)))  # dil verilmedi, yalnızca Türkçe açık: tespit yok
    assert t.language.code == "tr" and t.language.source == LanguageSource.DEFAULT
    assert t.text.count("yapabilirsin") == 3
    assert {u.speaker for u in t.utterances} == {"speaker_00"}  # aynı kişi üç kez


@VI_OK
def test_vad_ofseti_ve_asr_zamanlari():
    from live_stream.application.config import VadConfig
    from live_stream.infrastructure.catalog import language_profile
    from live_stream.infrastructure.normalizer import ViNormalizer
    from live_stream.infrastructure.sherpa_asr import build_asr
    from live_stream.infrastructure.sherpa_vad import SileroVad

    audio = np.concatenate([np.zeros(16000 * 3, dtype=np.float32), vi_wavs()[0]])
    segs = SileroVad(VadConfig(model=str(M / "silero_vad.onnx"))).split(audio)
    assert segs and 2.5 < segs[0].start < 3.6

    words = build_asr(language_profile("vi", M), ViNormalizer()).transcribe(segs)
    assert len(words) > 3
    assert words[0].start >= 2.5  # ofset eklenmiş olmalı
    assert all(a.start <= b.start for a, b in zip(words, words[1:]))
    assert all(w.text == w.text.lower() for w in words)


@VI_OK
def test_hotword_bpe_vocab_olmadan_hata_verir(tmp_path):
    from dataclasses import replace

    from live_stream.infrastructure.catalog import language_profile
    from live_stream.infrastructure.normalizer import ViNormalizer
    from live_stream.infrastructure.sherpa_asr import build_asr

    hw = tmp_path / "h.txt"
    hw.write_text("BÁC SĨ\n", encoding="utf-8")
    prof = language_profile("vi", M, hotwords_file=str(hw))
    files = {**prof.asr_model.files, "bpe_vocab": "yok.vocab"}
    prof = replace(prof, asr_model=replace(prof.asr_model, files=files))
    with pytest.raises(FileNotFoundError):
        build_asr(prof, ViNormalizer())


@LID_OK
def test_dil_tespiti_gercek_seslerde():
    from live_stream.application.config import LidConfig
    from live_stream.application.language import LanguageResolver
    from live_stream.domain import UnsupportedLanguageError, VadSegment
    from live_stream.infrastructure.catalog import lid_stages
    from live_stream.infrastructure.whisper_lid import WhisperLanguageDetector

    tiny, small = (WhisperLanguageDetector(e, d) for e, d in lid_stages(M))
    p = tiny.probabilities(load(LIDW / "tr-turkish.wav"))
    assert abs(sum(p.values()) - 1) < 1e-6 and len(p) == 99

    # 2.wav yalnızca 1,6 sn: tiny tek başına reddediyor, kademe small'a devredip doğru buluyor
    with pytest.raises(UnsupportedLanguageError):
        LanguageResolver(["vi", "tr"], [tiny]).resolve(None, [VadSegment(0.0, vi_wavs()[2])])

    res = LanguageResolver(["vi", "tr"], [tiny, small], LidConfig())
    seg = lambda a: [VadSegment(0.0, a)]  # noqa: E731
    assert res.resolve(None, seg(load(LIDW / "tr-turkish.wav"))).code == "tr"
    for a in vi_wavs():
        assert res.resolve(None, seg(a)).code == "vi"
    for f in ("en-english.wav", "de-german.wav", "zh-chinese.wav"):
        with pytest.raises(UnsupportedLanguageError):
            res.resolve(None, seg(load(LIDW / f)))


@VI_OK
@LID_OK
@need(*DIAR)
def test_uctan_uca_dil_tespitli(tmp_path):
    import soundfile as sf

    from live_stream.domain import AudioJob, LanguageSource
    from live_stream.interface.cli import MemoryRepo, build, default_config

    w = vi_wavs()
    gap = np.zeros(8000, dtype=np.float32)
    p = tmp_path / "konusma.wav"
    sf.write(p, np.concatenate([w[0], gap, w[1], gap, w[2], gap, w[0]]), 16000)

    uc = build(default_config(M), MemoryRepo())
    t = uc.run(AudioJob("e2e", str(p)))
    assert t.language.code == "vi" and t.language.source == LanguageSource.DETECTED
    assert t.utterances and len(t.text.split()) > 10
    # 0.wav, 1.wav, 2.wav üç farklı kişi (3D-Speaker uzaklığı 0,84-1,07); sıra 0,1,2,0
    assert [u.speaker for u in t.utterances] == ["speaker_00", "speaker_01", "speaker_02", "speaker_00"]
    assert uc.engines.loaded() == ["vi"]  # Türkçe model yüklenmedi


@need(CAMPP, TR_WAV, LIDW / "en-english.wav", VAD)
def test_konusmaci_sayisi_otomatik_ve_verilen():
    from live_stream.application.config import VadConfig
    from live_stream.infrastructure.sherpa_vad import SileroVad
    from live_stream.interface.composition import build_diarizer, default_config

    tr, en = load(TR_WAV), load(LIDW / "en-english.wav")
    gap = np.zeros(8000, dtype=np.float32)
    iki = np.concatenate([tr, gap, en, gap, tr, gap, en])
    tek = np.concatenate([tr, gap, tr, gap, tr])
    vad = SileroVad(VadConfig(model=str(VAD)))
    d = build_diarizer(default_config(M).diarizer)

    segs = d.diarize(iki, vad.split(iki))
    assert len({s.speaker for s in segs}) == 2
    assert [s.speaker for s in segs][:2] == ["speaker_00", "speaker_01"]
    assert len({s.speaker for s in d.diarize(tek, vad.split(tek))}) == 1
    assert len({s.speaker for s in d.diarize(iki, vad.split(iki), num_speakers=1)}) == 1
    assert d.diarize(iki, []) == []


@need(VAD, TR_WAV)
def test_canli_silero_algilayici_cumle_bulur():
    """Canlı mod, toplu işin Silero modelini (ONNX) çerçeve çerçeve kullanır."""
    from live_stream.application.live import FinalJob, Segmenter, SegmenterConfig
    from live_stream.infrastructure.live_vad import SileroFrameDetector

    tr = load(TR_WAV)
    sil = np.zeros(16000, dtype=np.float32)
    audio = np.concatenate([sil, tr, sil, sil, tr, sil])
    det = SileroFrameDetector(str(VAD))
    assert not det(np.zeros(512, np.float32))
    seg = Segmenter(det, SegmenterConfig(partial_interval_sec=0))
    finals = [j for i in range(0, len(audio), 1600) for j in seg.feed(audio[i:i + 1600])
              if isinstance(j, FinalJob)] + seg.flush()
    assert len(finals) == 2
    assert finals[0].start == pytest.approx(1.0, abs=0.4)
    dur = len(tr) / 16000
    assert all(abs((f.end - f.start) - dur) < 0.8 for f in finals)


@HAS_TR
@need(VAD, CAMPP, TR_WAV)
def test_canli_oturum_gercek_modelle_ve_kayit_toplu_ise(tmp_path, monkeypatch):
    """Yerel sunucu gerçek modellerle: WebSocket -> canlı metin -> kayıt -> konuşmacılı toplu iş."""
    import time

    from fastapi.testclient import TestClient

    from live_stream.domain import JobStatus
    from live_stream.interface.local_server import build_local
    from live_stream.interface.settings import Settings

    monkeypatch.setenv("CONCURRENCY", "1")
    s = Settings(models_dir=str(M), data_dir=str(tmp_path / "up"), recordings_dir=str(tmp_path / "rec"),
                 languages=["tr"], tr_model=TR_MODEL, whisper_device="cpu", verify_models=False)
    # (Türkçe; iki dilli canlı oturum aşağıda)
    app, store, queue, _ = build_local(s, workers=1)
    tr = load(TR_WAV)
    sil = np.zeros(16000, dtype=np.float32)
    pcm = (np.concatenate([sil, tr, sil, tr, sil]) * 32767).astype("<i2").tobytes()
    try:
        with TestClient(app) as c, c.websocket_connect("/ws/live?partial_sec=0") as ws:
            assert ws.receive_json()["type"] == "ready"
            for i in range(0, len(pcm), 3200):
                ws.send_bytes(pcm[i:i + 3200])
            ws.send_text('{"type": "stop"}')
            msgs = []
            while (m := ws.receive_json())["type"] != "done":
                msgs.append(m)
        finals = [x["text"].lower() for x in msgs if x["type"] == "final"]
        assert len(finals) == 2 and all("yapabilirsin" in f for f in finals)
        job_id = m["job_id"]
        assert job_id
        for _ in range(600):  # toplu iş arka planda bitsin
            rec = store.get(job_id)
            if rec.status in (JobStatus.DONE, JobStatus.FAILED):
                break
            time.sleep(0.1)
        assert rec.status == JobStatus.DONE, rec.error
        assert store.get_transcript(job_id).text.count("yapabilirsin") == 2
    finally:
        queue.shutdown(wait=False)


@VI_OK
@LID_OK
@need(*DIAR)
@HAS_TR
@pytest.mark.parametrize("secim", ["vi", "auto"])
def test_canli_vietnamca_ve_otomatik_dil(tmp_path, monkeypatch, secim):
    """İki dil açık: Vietnamca seçilince Zipformer; 'auto' seçilince ilk cümleden Vietnamca bulunur.
    Oturum kaydı Vietnamca toplu işe gider ve konuşmacı ayrımlı sonuç çıkar."""
    import time

    from fastapi.testclient import TestClient

    from live_stream.domain import JobStatus
    from live_stream.interface.local_server import build_local
    from live_stream.interface.settings import Settings

    monkeypatch.setenv("CONCURRENCY", "1")
    s = Settings(models_dir=str(M), data_dir=str(tmp_path / "up"), recordings_dir=str(tmp_path / "rec"),
                 languages=["tr", "vi"], tr_model=TR_MODEL, whisper_device="cpu", verify_models=False)
    app, store, queue, _ = build_local(s, workers=1)
    w = vi_wavs()
    sil = np.zeros(16000, dtype=np.float32)
    pcm = (np.concatenate([sil, w[0], sil, w[1], sil]) * 32767).astype("<i2").tobytes()
    try:
        with TestClient(app) as c, c.websocket_connect(f"/ws/live?language={secim}&partial_sec=0") as ws:
            ready = ws.receive_json()
            assert ready["type"] == "ready" and ready["language"] == secim
            for i in range(0, len(pcm), 3200):
                ws.send_bytes(pcm[i:i + 3200])
            ws.send_text('{"type": "stop"}')
            msgs = []
            while (m := ws.receive_json())["type"] != "done":
                msgs.append(m)
        finals = [x["text"] for x in msgs if x["type"] == "final"]
        assert len(finals) == 2 and all(len(f.split()) > 3 and f == f.lower() for f in finals)
        langs = [x for x in msgs if x["type"] == "language"]
        assert langs == ([] if secim == "vi" else [langs[0]]) and all(x["code"] == "vi" for x in langs)
        for _ in range(600):
            rec = store.get(m["job_id"])
            if rec.status in (JobStatus.DONE, JobStatus.FAILED):
                break
            time.sleep(0.1)
        assert rec.status == JobStatus.DONE, rec.error
        t = store.get_transcript(m["job_id"])
        assert t.language.code == "vi" and t.model == "zipformer-68m"
        assert len({u.speaker for u in t.utterances}) == 2  # 0.wav ve 1.wav iki farklı kişi
    finally:
        queue.shutdown(wait=False)


@VI_OK
@need(*DIAR)
def test_canli_oturum_sonu_webhooklari_gercek_alici(tmp_path, monkeypatch):
    """Konuşma bitince: önce 'live.ended' (canlı metin), konuşmacı ayrımı bitince 'job.done';
    ikisi de outbox'tan, imzalı, gerçek bir HTTP alıcısına."""
    import json
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from fastapi.testclient import TestClient

    from live_stream.domain import JobStatus
    from live_stream.infrastructure.webhook import verify_signature
    from live_stream.interface.local_server import build_local
    from live_stream.interface.settings import Settings

    got = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            got.append((dict(self.headers), body))
            self.send_response(204)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    hook = f"http://127.0.0.1:{srv.server_port}/hook"
    monkeypatch.setenv("CONCURRENCY", "1")
    s = Settings(models_dir=str(M), data_dir=str(tmp_path / "up"), recordings_dir=str(tmp_path / "rec"),
                 languages=["vi"], verify_models=False, live_webhook_url=hook, webhook_secret="gizli")
    app, store, queue, dispatcher = build_local(s, workers=1)
    w = vi_wavs()
    sil = np.zeros(16000, dtype=np.float32)
    pcm = (np.concatenate([sil, w[0], sil, w[1], sil]) * 32767).astype("<i2").tobytes()
    try:
        with TestClient(app) as c, c.websocket_connect("/ws/live?partial_sec=0") as ws:
            ws.receive_json()
            for i in range(0, len(pcm), 3200):
                ws.send_bytes(pcm[i:i + 3200])
            ws.send_text('{"type": "stop"}')
            while (m := ws.receive_json())["type"] != "done":
                pass
        assert m["webhook"] is True and m["job_id"]
        for _ in range(600):
            dispatcher.run_once()
            if store.get(m["job_id"]).status == JobStatus.DONE and len(got) >= 2:
                break
            time.sleep(0.1)
        dispatcher.run_once()
    finally:
        queue.shutdown(wait=False)
        srv.shutdown()

    events = [h["X-Event"] for h, _ in got]
    assert events == ["live.ended", "job.done"]
    for h, body in got:
        assert verify_signature("gizli", h["X-Timestamp"], body, h["X-Signature"])
    ended, done = (json.loads(b) for _, b in got)
    assert ended["session_id"] == m["session_id"] and len(ended["lines"]) == 2
    assert ended["job_id"] == done["job_id"] == m["job_id"] and got[0][0]["X-Session-Id"] == m["session_id"]
    assert done["status"] == "done" and len({u["speaker"] for u in done["utterances"]}) == 2


@VI_OK
@need(*DIAR)
def test_canli_konusmaci_etiketi_gercek_imza(tmp_path, monkeypatch):
    """Gerçek CAM++ imzası: iki farklı kişi (0.wav, 1.wav) farklı, aynı kişi aynı etiket alır."""
    from fastapi.testclient import TestClient

    from live_stream.interface.local_server import build_local
    from live_stream.interface.settings import Settings

    monkeypatch.setenv("CONCURRENCY", "1")
    s = Settings(models_dir=str(M), data_dir=str(tmp_path / "up"), recordings_dir=str(tmp_path / "rec"),
                 languages=["vi"], verify_models=False, live_auto_analyze=False)
    app, _, queue, _ = build_local(s, workers=1)
    w = vi_wavs()
    sil = np.zeros(16000, dtype=np.float32)
    pcm = (np.concatenate([sil, w[0], sil, w[1], sil, w[0], sil]) * 32767).astype("<i2").tobytes()
    try:
        with TestClient(app) as c:
            assert c.get("/live/health").json()["speakers"] is True
            with c.websocket_connect("/ws/live?partial_sec=0") as ws:
                ws.receive_json()
                for i in range(0, len(pcm), 3200):
                    ws.send_bytes(pcm[i:i + 3200])
                ws.send_text('{"type": "stop"}')
                msgs = []
                while (m := ws.receive_json())["type"] != "done":
                    msgs.append(m)
    finally:
        queue.shutdown(wait=False)
    assert [x["speaker"] for x in msgs if x["type"] == "final"] == ["speaker_00", "speaker_01", "speaker_00"]
