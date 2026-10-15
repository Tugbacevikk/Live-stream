import live_stream.interface.settings as st
from live_stream.interface.settings import Settings


def test_eszamanlilik_elle(monkeypatch):
    monkeypatch.setenv("CONCURRENCY", "5")
    assert Settings().concurrency == 5
    monkeypatch.setenv("CONCURRENCY", "0")
    assert Settings().concurrency == 1


def plan(monkeypatch, cpus, mem, phys=6, **kw):
    monkeypatch.delenv("CONCURRENCY", raising=False)
    monkeypatch.setattr(st, "available_cpus", lambda: cpus)
    monkeypatch.setattr(st, "physical_cpus", lambda: phys)  # makineden bağımsız
    monkeypatch.setattr(st, "total_memory_gb", lambda: mem)
    s = Settings(threads=2, languages=["vi", "tr"], **kw)
    return s.concurrency_plan(), s.concurrency_plan("tr"), s.concurrency_plan("vi")


def test_otomatik_cekirdek_ve_bellegin_kucugu(monkeypatch):
    # kullanıcının bilgisayarı: 12 mantıksal / 6 fiziksel çekirdek, 16 GB, Türkçe medium
    monkeypatch.delenv("WHISPER_THREADS", raising=False)
    (yerel, why), (tr, _), (vi, _) = plan(monkeypatch, 12, 16, tr_model="faster-whisper-medium")
    assert yerel == 6 and "bellek 16 GB" in why          # çekirdek sınırlıyor
    assert vi == 6                                        # Zipformer: 12 / ONNX_THREADS 2
    assert tr == 2                                        # Whisper süreci 6 iş parçacığı: 12 / 6
    # WHISPER_THREADS=2 iken (eski davranış) Türkçe süreçlerini bellek sınırlar
    monkeypatch.setenv("WHISPER_THREADS", "2")
    (_, _), (tr2, _), _ = plan(monkeypatch, 12, 16, tr_model="faster-whisper-medium")
    assert tr2 == 4
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


def test_whisper_is_parcacigi_butcesi(monkeypatch):
    from live_stream.interface.composition import config_from_settings

    monkeypatch.setattr(st, "available_cpus", lambda: 12)
    monkeypatch.setattr(st, "physical_cpus", lambda: 6)
    monkeypatch.delenv("WHISPER_THREADS", raising=False)
    s = st.Settings()
    assert s.whisper_threads == 6                      # tek iş de 6 çekirdeği kullanır (önce 2)
    assert s.whisper_workers(6) == 2 and s.whisper_workers(1) == 1
    monkeypatch.setenv("WHISPER_THREADS", "4")
    s = st.Settings()
    assert s.whisper_threads == 4 and s.whisper_workers(6) == 3
    monkeypatch.setattr(st, "physical_cpus", lambda: 1)
    monkeypatch.setenv("WHISPER_THREADS", "auto")
    assert st.Settings().whisper_threads == 2          # alt sınır
    cfg = config_from_settings(st.Settings(languages=["tr"]), workers=6)
    wp = cfg.languages["tr"].asr.whisper
    assert (wp.cpu_threads, wp.num_workers) == (2, 6)
