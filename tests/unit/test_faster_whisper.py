"""faster-whisper adaptörü (Türkçe): gerçek model yerine sahte nesnelerle.

Gerçek modelle doğrulama: tests/integration (models/faster-whisper-medium ya da -small varsa).
"""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from live_stream.application.config import AsrParams, ModelRef, WhisperParams
from live_stream.domain import VadSegment, Word
from live_stream.infrastructure.faster_whisper_asr import FasterWhisperAsr, drop_repeats
from live_stream.infrastructure.normalizer import normalizer_for

REF = ModelRef("faster-whisper-small", "faster-whisper", "models/faster-whisper-small", {})


def seg(start, end):
    return VadSegment(start, np.full(int(round((end - start) * 16000)), 0.1, np.float32))


def w(text, a, b):
    return NS(word=text, start=a, end=b)


class FakeBatched:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    def transcribe(self, audio, clip_timestamps, batch_size, **opt):
        self.calls.append((audio, clip_timestamps, batch_size, opt))

        def gen():  # gerçek kütüphane gibi tembel: hata döngüde çıkar
            if self.fail:
                raise RuntimeError("bellek yetmedi")
            yield NS(text=" Merhaba, nasılsınız?", words=[w(" Merhaba,", 1.0, 1.4), w(" nasılsınız?", 1.5, 2.1)])
            yield NS(text=" İzlediğiniz için teşekkür ederim.", words=[w(" İzlediğiniz", 5.0, 5.5)])
            yield NS(text=" evet evet evet evet", words=[w(" evet", 6.0, 6.2), w(" evet,", 6.2, 6.4),
                                                       w(" evet", 6.4, 6.6), w(" Evet", 6.6, 6.8)])
        return gen(), NS(language="tr")


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, samples, vad_filter, **opt):
        self.calls.append(len(samples))
        return iter([NS(text=" iyi günler", words=[w(" İyi", 0.1, 0.3), w(" günler", 0.4, 0.8)])]), None


def engine(batched=None, model=None, **wp):
    params = AsrParams(drop_phrases=("İzlediğiniz için teşekkür ederim.",), whisper=WhisperParams(**wp))
    return FasterWhisperAsr(REF, params, normalizer_for("tr"), "tr",
                            whisper_model=model or FakeModel(), batched=batched)


def test_toplu_cozme_mutlak_zaman_ve_suzgecler():
    b = FakeBatched()
    words = engine(b).transcribe([seg(1.0, 2.2), seg(5.0, 7.0)])
    # sistem çıktısı küçük harf, noktalamasız (iki dilde aynı); tekrar süzgeci noktalamaya bakmaz
    assert [x.text for x in words] == ["merhaba", "nasılsınız", "evet", "evet"]
    assert words[0].start == 1.0 and words[1].end == 2.1
    audio, clips, bs, opt = b.calls[0]
    assert clips == [{"start": 1.0, "end": 2.2}, {"start": 5.0, "end": 7.0}] and bs == 8
    assert audio[int(1.5 * 16000)] == pytest.approx(0.1) and audio[int(3 * 16000)] == 0  # boşluk sessiz
    assert opt["language"] == "tr" and opt["word_timestamps"] is True
    assert opt["condition_on_previous_text"] is False and opt["log_prob_threshold"] == -1.0


def test_toplu_cozme_hatasinda_tek_tek_cozer():
    m = FakeModel()
    words = engine(FakeBatched(fail=True), m).transcribe([seg(10.0, 11.0), seg(20.0, 21.5)])
    assert m.calls == [16000, 24000]
    # parçaya göreli zaman mutlak zamana çevrilir
    assert [(x.text, x.start) for x in words] == [("iyi", 10.1), ("günler", 10.4), ("iyi", 20.1), ("günler", 20.4)]


def test_istem_ciktida_tekrar_edilirse_atilir():
    e = engine(None, initial_prompt="Türkçe ses kaydı.")
    e.model = NS(transcribe=lambda s, vad_filter, **o: (iter([NS(text=" Türkçe ses kaydı.",
                                                                    words=[w(" Türkçe", 0, 1)])]), None))
    assert e.transcribe([seg(0, 1)]) == []


def test_tekrar_suzgeci():
    ws = [Word(t, i, i + 1) for i, t in enumerate(["a", "a", "a", "b", "a", "a", "a", "a"])]
    assert [x.text for x in drop_repeats(ws, 2)] == ["a", "a", "b", "a", "a"]
    assert drop_repeats([], 2) == []


def test_katalog_ve_profil():
    from live_stream.infrastructure.catalog import DEFAULT_MODELS, language_profile

    assert DEFAULT_MODELS["tr"] == "faster-whisper-medium"
    p = language_profile("tr", "models")
    assert p.segmentation == "vad" and p.asr_model.kind == "faster-whisper"
    assert p.asr_model.path("model").replace("\\", "/").endswith("faster-whisper-medium/model.bin")
