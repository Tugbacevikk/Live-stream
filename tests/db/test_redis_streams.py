"""Gerçek Redis ile kuyruk testleri. TEST_REDIS_URL yoksa atlanır.

    TEST_REDIS_URL=redis://localhost:6379/15 pytest tests/db/test_redis_streams.py
"""
import os
import time
import uuid

import pytest

URL = os.environ.get("TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_REDIS_URL yok")


@pytest.fixture
def r():
    import redis

    c = redis.Redis.from_url(URL)
    c.flushdb()
    yield c
    c.flushdb()


def consumer(r, q, handler, gave_up, **kw):
    from live_stream.infrastructure.redis_streams import StreamConsumer

    c = StreamConsumer(r, q, handler, on_give_up=lambda j, e: gave_up.append((j, e)),
                       block_ms=100, **kw)
    c.ensure_group()
    return c


def test_yonlendirme_ve_onay(r):
    from live_stream.infrastructure.redis_streams import RedisStreamJobQueue

    q = RedisStreamJobQueue(r)
    q.enqueue_detection("a")
    q.enqueue_transcription("b", "vi")
    done, gave = [], []
    for name in ("lid", "vi"):
        c = consumer(r, name, done.append, gave)
        c.run_once()
    assert done == ["a", "b"] and gave == []
    st = q.stats(["lid", "vi", "tr"])
    assert {k: (v["islenmekte"], v["bekleyen"], v["en_eski_sn"]) for k, v in st.items() if k != "dlq"} == {
        "lid": (0, 0, 0.0), "vi": (0, 0, 0.0), "tr": (0, 0, 0.0)}


def test_hata_tekrar_sonra_birakma(r):
    from live_stream.infrastructure.redis_streams import DLQ, RedisStreamJobQueue

    RedisStreamJobQueue(r).enqueue_transcription("x", "vi")
    calls, gave = [], []

    def boom(job_id):
        calls.append(job_id)
        raise RuntimeError("kalıcı hata")

    c = consumer(r, "vi", boom, gave, max_attempts=3)
    for _ in range(4):
        c.run_once()
    assert calls == ["x", "x", "x"] and len(gave) == 1 and "kalıcı" in gave[0][1]
    assert r.xlen(DLQ) == 1


def test_coken_worker_isini_baskasi_devralir(r):
    from live_stream.infrastructure.redis_streams import GROUP, RedisStreamJobQueue, stream_name

    RedisStreamJobQueue(r).enqueue_transcription("y", "vi")
    c1 = consumer(r, "vi", lambda j: None, [], consumer="olu-" + uuid.uuid4().hex[:4],
                  claim_idle_ms=200)
    # c1 mesajı alır ama işlemeden "çöker" (onay yok)
    assert c1._read_new()
    assert r.xpending(stream_name("vi"), GROUP)["pending"] == 1
    time.sleep(0.3)
    done = []
    c2 = consumer(r, "vi", done.append, [], claim_idle_ms=200)
    c2.run_once()
    assert done == ["y"] and r.xpending(stream_name("vi"), GROUP)["pending"] == 0


def test_redis_istek_siniri():
    import os

    import redis

    from live_stream.infrastructure.rate_limit import RedisRateLimiter

    r = redis.Redis.from_url(os.environ["TEST_REDIS_URL"])
    now = [600.0]
    lim = RedisRateLimiter(r, 2, clock=lambda: now[0], prefix="test:rl")
    r.delete(*[k for k in r.scan_iter("test:rl:*")] or ["yok"])
    assert [lim.check("a") for _ in range(3)][:2] == [None, None] and lim.check("a") == 61
    assert lim.check("b") is None
    now[0] += 60
    assert lim.check("a") is None


def test_kuyruk_istatistikleri_en_eski_ve_dlq():
    import redis

    from live_stream.infrastructure.redis_streams import RedisStreamJobQueue

    r = redis.Redis.from_url(URL)
    r.flushdb()
    q = RedisStreamJobQueue(r)
    q.enqueue_transcription("a", "vi")
    s = q.stats(["vi"])
    assert s["vi"]["bekleyen"] == 1 and s["vi"]["en_eski_sn"] >= 0 and s["dlq"]["toplam"] == 0


def test_kilit_tek_sahip_ve_bosta_kontrolu(r):
    from live_stream.infrastructure.redis_streams import RedisLock, RedisStreamJobQueue

    a, b = RedisLock(r, "test:kilit", ttl_s=5), RedisLock(r, "test:kilit", ttl_s=5)
    r.delete("test:kilit")
    assert a.acquire() and not b.acquire() and a.acquire()        # a yeniler, b alamaz
    r.set("test:kilit", "baskasi")                               # süre doldu, başkası aldı
    assert not a.acquire()                                       # eski sahip yenisininkini uzatamaz
    assert r.get("test:kilit") == b"baskasi"
    r.delete("test:kilit")
    assert b.acquire()
    b.release()
    assert r.get("test:kilit") is None
    q = RedisStreamJobQueue(r)
    assert q.is_idle("tr") and q.live_consumers("tr") == 0


def test_isci_varligi(r):
    import threading
    import time

    from live_stream.infrastructure.redis_streams import RedisStreamJobQueue, StreamConsumer

    q = RedisStreamJobQueue(r)
    assert q.live_consumers("vi") == 0
    c = StreamConsumer(r, "vi", lambda j: None, lambda j, e: None, block_ms=100, consumer="t1")
    t = threading.Thread(target=c.run_forever, daemon=True)
    t.start()
    for _ in range(50):
        if q.live_consumers("vi") == 1:
            break
        time.sleep(0.05)
    assert q.live_consumers("vi") == 1                 # mesaj almadan da görünür
    c.stop()
    t.join(timeout=5)
    assert q.live_consumers("vi") == 0                 # kapanınca kaybolur
