import live_stream.interface.settings as st
from live_stream.interface.settings import Settings


def test_eszamanlilik_elle(monkeypatch):
    monkeypatch.setenv("CONCURRENCY", "5")
    assert Settings().concurrency == 5
    monkeypatch.setenv("CONCURRENCY", "0")
    assert Settings().concurrency == 1


def plan(monkeypatch, cpus, mem, **kw):
    monkeypatch.delenv("CONCURRENCY", raising=False)
    monkeypatch.setattr(st, "available_cpus", lambda: cpus)
    monkeypatch.setattr(st, "total_memory_gb", lambda: mem)
    s = Settings(threads=2, languages=["vi", "tr"], **kw)
    return s.concurrency_plan(), s.concurrency_plan("tr"), s.concurrency_plan("vi")


def test_otomatik_cekirdek_ve_bellegin_kucugu(monkeypatch):
    # kullanıcının bilgisayarı: 12 mantıksal çekirdek, 16 GB, Türkçe medium
    (yerel, why), (tr, _), (vi, _) = plan(monkeypatch, 12, 16, tr_model="faster-whisper-medium")
    assert yerel == 6 and "bellek 16 GB" in why          # çekirdek sınırlıyor
    assert tr == 4 and vi == 6                            # tam modda Türkçe süreçlerini bellek sınırlıyor
    # 8 GB'lık makinede bellek sınırlar
    (yerel8, _), (tr8, _), _ = plan(monkeypatch, 12, 8, tr_model="faster-whisper-medium")
    assert yerel8 == 4 and tr8 == 2
    # small daha az bellek: daha çok süreç
    (_, _), (tr_small, _), _ = plan(monkeypatch, 12, 8, tr_model="faster-whisper-small")
    assert tr_small > tr8


def test_bellek_okunamazsa_cekirdek(monkeypatch):
    (n, why), _, _ = plan(monkeypatch, 8, None)
    assert n == 4 and "okunamadı" in why


def test_bellek_okunur():
    m = st.total_memory_gb()
    assert m is None or m > 0.5
