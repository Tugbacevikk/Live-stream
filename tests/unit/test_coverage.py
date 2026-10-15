import numpy as np

from live_stream.application.coverage import add_rescued, coverage
from live_stream.domain import VadSegment, Word


def seg(a, b):
    return VadSegment(a, np.zeros(int((b - a) * 16000), np.float32))


def test_kapsama_ve_yazisiz_araliklar():
    speech = [seg(0, 10), seg(12, 14)]
    words = [Word("a", 0.5, 1.0), Word("b", 1.2, 2.0), Word("c", 6.0, 7.0)]
    gaps, q = coverage(speech, words, min_gap=2.0)
    assert gaps == [(2.0, 6.0), (7.0, 10.0), (12.0, 14.0)]
    assert q["konusma_sn"] == 12.0 and q["yazisiz_sn"] == 0.5 + 0.2 + 4 + 3 + 2
    assert q["kapsama_yuzde"] == round((12 - 9.7) / 12 * 100, 1) and q["yazisiz_aralik"] == 3
    assert coverage([], words)[1]["kapsama_yuzde"] == 100.0


def test_kurtarilan_kelime_var_olanla_cakisirsa_atilir():
    words = [Word("a", 0, 1), Word("d", 5, 6)]
    merged, n = add_rescued(words, [Word("a", 0.1, 0.9), Word("b", 2, 3), Word("c", 3, 4)])
    assert n == 2 and [w.text for w in merged] == ["a", "b", "c", "d"]


def test_guvenlik_agi_yalnizca_destekleyen_motorda_ve_tam_profilde():
    from live_stream.application.config import SafetyNetConfig
    from live_stream.application.fusion import FusionEngine
    from live_stream.application.language import LanguageResolver
    from live_stream.application.transcribe import TranscribeAndDiarize
    from live_stream.domain import AudioJob, SpeakerSegment
    from live_stream.infrastructure.engines import LazyAsrEngines

    class L:
        def load(self, p):
            return np.zeros(16000 * 12, np.float32)

    class V:
        def split(self, a):
            return [seg(0, 12)]

    class D:
        def diarize(self, audio, segments=None, num_speakers=None):
            return [SpeakerSegment("speaker_00", 0, 12)]

    class Whisper:
        name, supports_rescue = "w", True
        rescued = []

        def transcribe(self, segs, fast=False):
            return [Word("bir", 0.5, 1.0, 0.9)]            # 1-12 sn arası atlandı

        def rescue(self, clips):
            Whisper.rescued.append([(round(c.start, 2), round(c.end, 2)) for c in clips])
            return [Word("bir", 0.6, 0.9, 0.9), Word("iki", 5, 5.5, 0.3), Word("üç", 9, 9.5, 0.8)]

    class Repo:
        def set_status(self, *a, **k): pass
        def save(self, t): pass

    uc = TranscribeAndDiarize(L(), V(), LanguageResolver(["tr"], None), LazyAsrEngines({"tr": Whisper}),
                              D(), FusionEngine(), Repo(), safety=SafetyNetConfig(min_gap=2.0, margin=0.3))
    t = uc.run(AudioJob("x", "x.wav", "tr"))
    assert Whisper.rescued == [[(0.7, 12.0)]]                 # kenar payıyla
    assert t.text == "bir iki üç" and t.quality["kurtarilan_kelime"] == 2
    # aralar 1,5 sn'den uzun: üç ayrı satır; ortadaki düşük güvenli
    assert [u.low_conf for u in t.utterances] == [(), (0,), ()]
    assert [u.conf for u in t.utterances] == [0.9, 0.3, 0.8] and t.quality["dusuk_guvenli_kelime"] == 1
    assert "guvenlik_agi" in t.timings
    uc.run(AudioJob("y", "x.wav", "tr", profile="feedback"))  # hızlı profilde kurtarma yok
    assert len(Whisper.rescued) == 1
