import numpy as np
import pytest

from live_stream.application.fusion import FusionEngine
from live_stream.application.language import LanguageResolver
from live_stream.application.transcribe import TranscribeAndDiarize
from live_stream.domain import (
    AudioJob, JobStatus, LanguageSource, SpeakerSegment, UnsupportedLanguageError, VadSegment, Word,
)
from live_stream.infrastructure.engines import LazyAsrEngines


class Loader:
    def load(self, path):
        return np.zeros(16000 * 3, dtype=np.float32)


class Vad:
    def split(self, audio):
        return [VadSegment(start=0.5, samples=np.zeros(16000, dtype=np.float32))]


class Asr:
    def __init__(self, name, words):
        self.name, self.words = name, words

    def transcribe(self, segs):
        return [Word(t, 0.5 + i * 0.3, 0.8 + i * 0.3) for i, t in enumerate(self.words)]


class Det:
    def __init__(self, probs):
        self.p = probs

    def probabilities(self, audio):
        return self.p


class Diar:
    def diarize(self, audio, segments=None, num_speakers=None):
        return [SpeakerSegment("speaker_00", 0.4, 1.5)]


class BozukDiar:
    def diarize(self, audio, segments=None, num_speakers=None):
        raise RuntimeError("model yok")


class Repo:
    def __init__(self):
        self.history, self.saved = [], None

    def set_status(self, job_id, status, error=None):
        self.history.append((status, error))

    def save(self, t):
        self.saved = t
        self.history.append((JobStatus.DONE, None))


def make(probs, diar=None, loaded=None):
    loaded = loaded if loaded is not None else []

    def f(code, words):
        def build():
            loaded.append(code)
            return Asr(f"asr-{code}", words)
        return build

    engines = LazyAsrEngines({"vi": f("vi", ["xin", "chào"]), "tr": f("tr", ["merhaba"])})
    repo = Repo()
    uc = TranscribeAndDiarize(Loader(), Vad(), LanguageResolver(["vi", "tr"], Det(probs)),
                              engines, diar or Diar(), FusionEngine(), repo)
    return uc, repo, loaded


def test_tespit_edilen_dilin_motoru_kullanilir_digeri_yuklenmez():
    uc, repo, loaded = make({"vi": 0.9, "tr": 0.05, "en": 0.05})
    t = uc.run(AudioJob("j1", "x.wav"))
    assert t.text == "xin chào" and t.model == "asr-vi"
    assert t.language.code == "vi" and t.language.source == LanguageSource.DETECTED
    assert loaded == ["vi"]
    assert [s for s, _ in repo.history] == [JobStatus.RUNNING, JobStatus.DONE]
    assert repo.saved is t and t.utterances[0].speaker == "speaker_00"


def test_istekteki_dil_tespiti_gecersiz_kilar():
    uc, _, loaded = make({"vi": 1.0})
    t = uc.run(AudioJob("j2", "x.wav", language="tr"))
    assert t.text == "merhaba" and t.language.source == LanguageSource.REQUEST and loaded == ["tr"]


def test_desteklenmeyen_dil_reddedilir_motor_yuklenmez():
    uc, repo, loaded = make({"en": 0.99, "vi": 0.01})
    with pytest.raises(UnsupportedLanguageError):
        uc.run(AudioJob("j3", "x.wav"))
    assert repo.history[-1][0] == JobStatus.REJECTED and "en" in repo.history[-1][1]
    assert loaded == [] and repo.saved is None


def test_hata_durumu_kaydedilir():
    uc, repo, _ = make({"vi": 1.0}, diar=BozukDiar())
    with pytest.raises(RuntimeError):
        uc.run(AudioJob("j4", "x.wav"))
    assert repo.history[-1][0] == JobStatus.FAILED and "model yok" in repo.history[-1][1]
    assert repo.saved is None


def test_konusma_yoksa_asr_calismaz():
    """Sessizlik / test sinyali: VAD konuşma bulmadıysa 'full' bölmeli motor da çalışmaz
    (Zipformer 1 sn'lik bip sesinde "ừ", Whisper "Altyazı M.K." uyduruyordu)."""
    class BosVad:
        def split(self, audio):
            return []

    class Chunker:
        def split(self, audio):
            return [VadSegment(start=0.0, samples=audio)]

    calls = []

    class FullAsr(Asr):
        segmentation = "full"

        def transcribe(self, segs):
            calls.append(segs)
            return super().transcribe(segs)

    engines = LazyAsrEngines({"vi": lambda: FullAsr("asr-vi", ["ừ"]), "tr": lambda: FullAsr("asr-tr", [])})
    uc = TranscribeAndDiarize(Loader(), BosVad(), LanguageResolver(["vi", "tr"], Det({"vi": 1.0})),
                              engines, Diar(), FusionEngine(), Repo(), chunker=Chunker())
    t = uc.run(AudioJob("bos", "x.wav", language="tr"))
    assert t.utterances == [] and calls == []


def test_feedback_profili_kisa_seste_konusmaci_ayirmayi_atlar():
    from live_stream.application.config import ProfileConfig

    class SayanDiar:
        calls = 0

        def diarize(self, audio, segments=None, num_speakers=None):
            SayanDiar.calls += 1
            return [SpeakerSegment("speaker_00", 0.4, 1.0), SpeakerSegment("speaker_01", 1.0, 1.5)]

    class HizliAsr(Asr):
        supports_fast = True
        fast_calls = []

        def transcribe(self, segs, fast=False):
            HizliAsr.fast_calls.append(fast)
            return super().transcribe(segs)

    engines = LazyAsrEngines({"tr": lambda: HizliAsr("asr-tr", ["iyi", "günler"])})
    uc = TranscribeAndDiarize(Loader(), Vad(), LanguageResolver(["tr"], None), engines, SayanDiar(),
                              FusionEngine(), Repo(), profiles=ProfileConfig(min_diarize_sec=10))
    t = uc.run(AudioJob("f", "x.wav", "tr", profile="feedback"))           # 3 sn < 10 sn
    assert SayanDiar.calls == 0 and {u.speaker for u in t.utterances} == {"speaker_00"}
    assert HizliAsr.fast_calls == [True] and t.profile == "feedback"
    assert set(t.timings) >= {"yukleme", "vad", "dil", "asr", "konusmaci", "birlestirme", "toplam"}
    t2 = uc.run(AudioJob("g", "x.wav", "tr"))                               # varsayılan full
    assert SayanDiar.calls == 1 and HizliAsr.fast_calls == [True, False] and t2.profile == "full"
    uc.run(AudioJob("h", "x.wav", "tr", num_speakers=1))                    # tek kişi: ayırma yok
    assert SayanDiar.calls == 1
