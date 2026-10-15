import numpy as np

from live_stream.infrastructure.chunker import EnergyChunker

SR = 16000


def test_tum_sesi_kapsar_ve_sessizlikten_keser():
    rng = np.random.default_rng(0)
    parts = []
    for i in range(6):  # 7 sn konuşma + 0,5 sn sessizlik
        parts += [rng.normal(0, 0.1, 7 * SR).astype(np.float32), np.zeros(SR // 2, np.float32)]
    a = np.concatenate(parts)
    segs = EnergyChunker(max_len=15, min_len=5).split(a)
    assert sum(len(s.samples) for s in segs) == len(a)            # boşluk yok
    assert all(len(s.samples) <= 15 * SR for s in segs)
    for s in segs[:-1]:                                            # kesimler sessizliğe denk gelir
        end = int(round(s.end * SR))
        assert np.abs(a[end - 400:end + 400]).max() == 0


def test_sessiz_parcalar_atlanir_kisa_ses_tek_parca():
    ch = EnergyChunker(max_len=15, min_len=5)
    assert ch.split(np.zeros(40 * SR, np.float32)) == []
    x = np.full(3 * SR, 0.1, np.float32)
    assert len(ch.split(x)) == 1 and ch.split(np.zeros(0, np.float32)) == []
