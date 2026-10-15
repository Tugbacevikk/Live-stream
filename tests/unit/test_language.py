import numpy as np
import pytest

from live_stream.application.config import LidConfig
from live_stream.application.language import LanguageResolver
from live_stream.domain import LanguageSource, UnsupportedLanguageError, VadSegment


class Det:
    def __init__(self, probs):
        self.probs, self.calls = probs, []

    def probabilities(self, audio):
        self.calls.append(len(audio))
        return self.probs


def seg(sec, start=0.0):
    return VadSegment(start=start, samples=np.ones(int(sec * 16000), dtype=np.float32))


SEGS = [seg(2.0)]


def test_istekteki_dil_kullanilir_tespit_yapilmaz():
    d = Det({"vi": 1.0})
    r = LanguageResolver(["vi", "tr"], d).resolve("TR", SEGS)
    assert (r.code, r.source, r.confidence) == ("tr", LanguageSource.REQUEST, 1.0)
    assert d.calls == []


def test_istekte_desteklenmeyen_dil_reddedilir():
    with pytest.raises(UnsupportedLanguageError, match="desteklenmiyor"):
        LanguageResolver(["vi", "tr"], Det({})).resolve("en", SEGS)


def test_kisitli_tespit_desteklenenler_arasinda_secer():
    # 99 dil arasında en olası 'th' olsa bile vi/tr arasında vi çok daha olası
    d = Det({"th": 0.40, "vi": 0.55, "tr": 0.02, "en": 0.03})
    r = LanguageResolver(["vi", "tr"], d, LidConfig(min_supported=0.5)).resolve(None, SEGS)
    assert r.code == "vi" and r.source == LanguageSource.DETECTED
    assert r.confidence == pytest.approx(0.55 / 0.57, abs=1e-3) and not r.uncertain


def test_desteklenmeyen_dildeki_ses_reddedilir():
    d = Det({"en": 0.97, "vi": 0.02, "tr": 0.01})
    with pytest.raises(UnsupportedLanguageError) as e:
        LanguageResolver(["vi", "tr"], d).resolve(None, SEGS)
    assert e.value.detected == "en"


def test_dusuk_guven_belirsiz_isaretlenir():
    d = Det({"vi": 0.5, "tr": 0.4, "en": 0.1})
    r = LanguageResolver(["vi", "tr"], d, LidConfig(min_confidence=0.8)).resolve(None, SEGS)
    assert r.code == "vi" and r.uncertain


def test_tespite_en_fazla_max_seconds_konusma_gider():
    d = Det({"vi": 1.0})
    LanguageResolver(["vi", "tr"], d, LidConfig(max_seconds=3)).resolve(None, [seg(2), seg(2, 5)])
    assert d.calls == [3 * 16000]


def test_konusma_yoksa_tespit_yapilmaz():
    d = Det({"vi": 1.0})
    r = LanguageResolver(["vi", "tr"], d).resolve(None, [])
    assert r.source == LanguageSource.DEFAULT and d.calls == []


def test_tek_dil_varsa_tespit_gereksiz():
    r = LanguageResolver(["vi"], None).resolve(None, SEGS)
    assert r.code == "vi" and r.source == LanguageSource.DEFAULT


def test_tespit_kapali_ve_dil_yoksa_hata():
    with pytest.raises(UnsupportedLanguageError):
        LanguageResolver(["vi", "tr"], None).resolve(None, SEGS)


def test_kademe_ucuz_model_emin_degilse_guclu_modele_gecer():
    tiny = Det({"ar": 0.9, "vi": 0.03, "tr": 0.01})   # kısa seste yanlış ret
    small = Det({"vi": 0.9, "en": 0.1})
    r = LanguageResolver(["vi", "tr"], [tiny, small]).resolve(None, SEGS)
    assert r.code == "vi" and not r.uncertain
    assert len(tiny.calls) == 1 and len(small.calls) == 1


def test_kademe_ucuz_model_eminse_guclu_model_calismaz():
    tiny, small = Det({"tr": 0.95, "en": 0.05}), Det({"vi": 1.0})
    r = LanguageResolver(["vi", "tr"], [tiny, small]).resolve(None, SEGS)
    assert r.code == "tr" and small.calls == []


def test_kademe_son_model_de_reddederse_ret():
    dets = [Det({"en": 0.99, "vi": 0.01}), Det({"en": 0.98, "tr": 0.02})]
    with pytest.raises(UnsupportedLanguageError) as e:
        LanguageResolver(["vi", "tr"], dets).resolve(None, SEGS)
    assert e.value.detected == "en"
