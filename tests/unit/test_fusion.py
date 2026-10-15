from live_stream.application.config import FusionConfig
from live_stream.application.fusion import FusionEngine
from live_stream.domain import BELIRSIZ, SpeakerSegment as S, Word as Y

A, B = "speaker_00", "speaker_01"


def eng(**kw):
    return FusionEngine(FusionConfig(**kw))


def test_orta_nokta_tek_konusmaciya_duser():
    segs = [S(A, 0, 2), S(B, 2, 4)]
    syl = [Y("chào", 0.2, 0.5), Y("bác", 1.8, 2.4), Y("sĩ", 2.5, 2.8)]
    # "bác" orta noktası 2.1 -> B
    assert eng().assign(syl, segs) == [A, B, B]


def test_ust_uste_konusmada_baskin_konusmaci():
    segs = [S(A, 0, 3), S(B, 1.0, 1.25)]
    # hece 0.9-1.3, orta 1.1 iki aralıkta; A örtüşmesi 0.4, B 0.25 -> A payı 0.62
    assert eng(min_dominance=0.6).assign([Y("x", 0.9, 1.3)], segs) == [A]
    assert eng(min_dominance=0.7).assign([Y("x", 0.9, 1.3)], segs) == [BELIRSIZ]


def test_bosluga_dusen_hece_en_yakina_ya_da_belirsiz():
    segs = [S(A, 0, 1), S(B, 3, 4)]
    assert eng(max_gap=0.5).assign([Y("a", 1.1, 1.3)], segs) == [A]      # 0.2 uzak
    assert eng(max_gap=0.5).assign([Y("b", 2.6, 2.8)], segs) == [B]      # 0.3 uzak
    assert eng(max_gap=0.5).assign([Y("c", 1.9, 2.1)], segs) == [BELIRSIZ]



def test_uzun_aralik_icindeki_hece_bulunur():
    # A'nın uzun aralığı, B'nin kısa aralıklarından önce başlıyor: geriye bakma doğru çalışmalı
    segs = [S(A, 0, 10), S(B, 1, 1.5), S(B, 2, 2.5)]
    assert eng().assign([Y("x", 5, 5.2)], segs) == [A]


def test_satirlara_toplama():
    segs = [S(A, 0, 2), S(B, 2, 5)]
    syl = [Y("xin", 0.1, 0.3), Y("chào", 0.3, 0.6), Y("dạ", 2.1, 2.3), Y("vâng", 4.5, 4.8)]
    u = eng(new_utterance_gap=1.5).fuse(syl, segs)
    assert [(x.speaker, x.text) for x in u] == [(A, "xin chào"), (B, "dạ"), (B, "vâng")]
    assert u[0].start == 0.1 and u[0].end == 0.6


def test_sirasiz_heceler_siralanir():
    segs = [S(A, 0, 5)]
    u = eng().fuse([Y("b", 1, 1.2), Y("a", 0, 0.2)], segs)
    assert u[0].text == "a b"


def test_tek_konusmaci_varsa_tum_kelimeler_ona():
    # VAD'ın kaçırdığı (aralık dışı) kelimeler bile BELIRSIZ değil, tek konuşmacıya gider
    w = [Y("a", 0.1, 0.3), Y("b", 10.0, 10.2)]
    assert eng(max_gap=0.5).assign(w, [S(A, 0, 1)]) == [A, A]
    assert eng().assign(w, []) == ["speaker_00", "speaker_00"]


def test_uzun_satir_en_uzun_duraklamadan_bolunur():
    # ~60 sn, kelimeler 0,5 sn arayla; iki yerde 1 sn'lik duraklama
    ws, t = [], 0.0
    while t < 60:
        ws.append(Y("x", t, t + 0.4))
        t += 1.4 if len(ws) in (37, 83) else 0.5   # 18. ve 41,9. sn'den sonra duraklama
    utts = eng(max_utterance=25).fuse(ws, [S(A, 0, 61)])
    assert [round(u.start, 1) for u in utts] == [0, 19.4, 43.3]   # iki duraklamadan
    assert all(u.end - u.start <= 25 for u in utts) and {u.speaker for u in utts} == {A}
    assert sum(len(u.text.split()) for u in utts) == len(ws)       # kelime kaybı yok
    assert len(eng(max_utterance=0).fuse(ws, [S(A, 0, 61)])) == 1  # 0 = bölme yok


def test_duraklamasiz_uzun_satir_sinirdan_bolunur():
    ws = [Y("x", i * 0.5, i * 0.5 + 0.5) for i in range(120)]       # 60 sn, hiç boşluk yok
    utts = eng(max_utterance=25).fuse(ws, [S(A, 0, 61)])
    assert all(u.end - u.start <= 25 for u in utts) and len(utts) == 3


def test_araya_sikismis_kisa_konusmaci_duzeltmesi():
    ws = [Y("a", 0, .5), Y("b", .6, 1), Y("x", 1.1, 1.4), Y("c", 1.5, 2), Y("d", 2.1, 2.5),
          Y("u", 2.6, 4.5), Y("e", 4.6, 5)]
    segs = [S(A, 0, 1.05), S(B, 1.05, 1.45), S(A, 1.45, 2.55), S(B, 2.55, 4.55), S(A, 4.55, 6)]
    off = eng().fuse(ws, segs)
    assert [u.speaker for u in off] == [A, B, A, B, A]                       # kapalıyken olduğu gibi
    on = eng(sandwich_max=1.0).fuse(ws, segs)
    assert [(u.speaker, u.text) for u in on] == [(A, "a b x c d"), (B, "u"), (A, "e")]  # 1,9 sn'lik B kalır
