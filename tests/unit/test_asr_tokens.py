import unicodedata

from live_stream.infrastructure.normalizer import TrNormalizer
from live_stream.infrastructure.sherpa_asr import tokens_to_words

N = TrNormalizer(lowercase=True)


def test_bpe_parcalari_kelimeye_birlesir_ve_ofset_eklenir():
    tokens = ["▁İYİ", "▁GÜN", "LER", "▁HOCAM"]
    ts = [0.0, 0.4, 0.5, 0.9]
    s = tokens_to_words(tokens, ts, seg_start=10.0, seg_end=11.5, normalizer=N)
    assert [x.text for x in s] == ["iyi", "günler", "hocam"]
    assert [x.start for x in s] == [10.0, 10.4, 10.9]
    assert [round(x.end, 2) for x in s] == [10.4, 10.9, 11.5]  # son kelime parça sonuna kadar


def test_kelime_suresi_sinirlanir():
    s = tokens_to_words(["▁A", "▁B"], [0.0, 3.0], 0.0, 5.0, N, max_duration=1.0)
    assert s[0].end == 1.0 and s[1].end == 4.0


def test_bosluk_ile_baslayan_parca_da_yeni_kelime():
    s = tokens_to_words([" İYİ", " AKŞAMLAR"], [0.0, 0.3], 0.0, 1.0, N)
    assert [x.text for x in s] == ["iyi", "akşamlar"]


def test_bos_cikti():
    assert tokens_to_words([], [], 0.0, 1.0, N) == []


def test_normalizer_nfc_ve_kucuk_harf():
    ayrik = unicodedata.normalize("NFD", "ÇİĞDEM  ÖĞRETMEN")
    assert N.normalize(ayrik) == unicodedata.normalize("NFC", "çiğdem öğretmen")
    assert TrNormalizer(lowercase=True, strip_punct=True).normalize("Merhaba, hocam!") == "merhaba hocam"


# --- Türkçe ---
from live_stream.infrastructure.normalizer import turkish_lower  # noqa: E402
from live_stream.infrastructure.sherpa_asr import spread_words  # noqa: E402


def test_harf_tabanli_ctc_ciktisi_kelimeye_birlesir():
    # Omnilingual: her harf ayrı token, " " ayırıcı
    toks = list("iki gün")
    ts = [0.1 * i for i in range(len(toks))]
    w = tokens_to_words(toks, ts, 5.0, 6.0, TrNormalizer())
    assert [x.text for x in w] == ["iki", "gün"]
    assert w[0].start == 5.0 and round(w[1].start, 2) == 5.4


def test_zaman_bilgisi_yoksa_harf_sayisiyla_yayilir():
    w = spread_words("Merhaba doktor bey.", 2.0, 4.0, TrNormalizer())
    assert [x.text for x in w] == ["Merhaba", "doktor", "bey."]
    assert w[0].start == 2.0 and abs(w[-1].end - 4.0) < 1e-9
    assert w[0].end - w[0].start > w[2].end - w[2].start  # uzun kelime daha uzun sürer
    assert spread_words("   ", 0, 1, TrNormalizer()) == []


def test_turkce_kucuk_harf():
    assert turkish_lower("IŞIK İLAÇ") == "ışık ilaç"
    assert TrNormalizer().normalize("İzmir'de  doktor.") == "İzmir'de doktor."  # büyük harf korunur
    assert TrNormalizer(lowercase=True, strip_punct=True).normalize("İLAÇ, IŞIK!") == "ilaç ışık"


def test_parcalar_sureye_gore_gruplanir():
    import numpy as np

    from live_stream.domain import VadSegment
    from live_stream.infrastructure.sherpa_asr import batches

    segs = [VadSegment(i * 30.0, np.zeros(int(d * 16000), np.float32)) for i, d in enumerate([20, 25, 20, 70, 5])]
    g = batches(segs, 60)
    assert [[round(s.end - s.start) for s in b] for b in g] == [[20, 25], [20], [70], [5]]
    assert [s for b in g for s in b] == segs and batches([], 60) == []


def test_baska_alfabedeki_uydurma_kelime_atilir():
    from live_stream.infrastructure.normalizer import normalizer_for

    assert normalizer_for("tr").normalize("hoşçakalın कजतच 38,5") == "hoşçakalın 38,5"
    assert normalizer_for("tr").normalize("Merhaba hocam Привет") == "merhaba hocam"
    assert normalizer_for("tr").normalize("कजतच") == ""
    assert normalizer_for("tr").normalize("nasыsыn") == "nasısın"            # karışık: düzeltilir


def test_kelime_guveni_parca_olasiliklarindan():
    import math

    w = tokens_to_words(["▁İYİ", "▁GÜN", "LER"], [0.0, 0.5, 0.7], 0.0, 1.0, N,
                        log_probs=[-0.1, -1.0, -2.0])
    assert [x.text for x in w] == ["iyi", "günler"]
    assert w[0].conf == round(math.exp(-0.1), 3) and w[1].conf == round(math.exp(-1.5), 3)
    assert tokens_to_words(["▁A"], [0.0], 0, 1, N)[0].conf is None


def test_canli_altyazi_bicimi_buyuk_harf_ve_noktalamayi_korur():
    from live_stream.infrastructure.normalizer import live_normalizer

    assert live_normalizer("tr").normalize("Merhaba, ben Tuğba. Привет") == "Merhaba, ben Tuğba."
