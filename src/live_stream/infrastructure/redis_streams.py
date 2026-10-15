"""Redis Streams kuyruğu (Celery yerine; Windows'ta da çalışır).

Desen (fine-tuning reposundaki RedisStreamAdapter'dan uyarlandı):
  - her kuyruk bir akış (stream): asr:lid, asr:vi, asr:tr
  - worker'lar bir tüketici grubunda (consumer group) mesaj paylaşır (XREADGROUP)
  - iş bitince XACK; bitmeden süreç ölürse mesaj "bekleyen" listesinde kalır ve
    claim_idle süresinden sonra başka bir worker XAUTOCLAIM ile devralır
  - uzun işlerde her heartbeat saniyede bir XCLAIM(min_idle=0) ile mesaj
    "canlı" tutulur; böylece claim_idle kısa olabilir (çöken iş çabuk devralınır)
  - hata: deneme hakkı varsa mesaj yeniden kuyruğa eklenir; yoksa iş FAILED olur
    ve mesaj asr:dlq (ölü mektup) akışına yazılır
"""
from __future__ import annotations

import logging
import os
import socket
import threading
import time
from collections.abc import Callable

import redis

log = logging.getLogger(__name__)

GROUP = "asr-workers"
DLQ = "asr:dlq"


def presence_key(queue: str, consumer: str) -> str:
    return f"asr:isci:{queue}:{consumer}"


def stream_name(queue: str) -> str:
    """queue: 'lid' ya da dil kodu ('vi', 'tr')."""
    return f"asr:{queue}"


def consumer_name() -> str:
    return f"{socket.gethostname()}-{os.getpid()}"


class RedisStreamJobQueue:
    """JobQueue portunun Redis Streams gerçeklemesi (API ve dil tespiti worker'ı kullanır)."""

    def __init__(self, client: redis.Redis, maxlen: int = 100_000):
        self.r, self.maxlen = client, maxlen

    def _add(self, queue: str, job_id: str, attempt: int = 1) -> None:
        self.r.xadd(stream_name(queue), {"job_id": job_id, "attempt": str(attempt)},
                    maxlen=self.maxlen, approximate=True)

    def enqueue_detection(self, job_id: str) -> None:
        self._add("lid", job_id)

    def enqueue_transcription(self, job_id: str, language: str) -> None:
        self._add(language, job_id)

    def stats(self, queues: list[str]) -> dict[str, dict[str, int]]:
        """Kuyruk başına bekleyen (henüz alınmamış + işlenmekte) mesaj sayısı."""
        out = {}
        for q in queues:
            s = stream_name(q)
            try:
                groups = {_s(g["name"]): g for g in self.r.xinfo_groups(s)}
                g = groups.get(GROUP)
                out[q] = {"islenmekte": int(g["pending"]) if g else 0,
                          "bekleyen": int(g.get("lag") or 0) if g else int(self.r.xlen(s)),
                          "en_eski_sn": self._oldest_age(s, g)}
            except redis.ResponseError:  # akış henüz yok
                out[q] = {"islenmekte": 0, "bekleyen": 0, "en_eski_sn": 0.0}
        out["dlq"] = {"toplam": int(self.r.xlen(DLQ)) if self.r.exists(DLQ) else 0}
        return out

    def is_idle(self, queue: str) -> bool:
        """Kuyrukta ne bekleyen ne işlenmekte olan mesaj var mı (bakımın yeniden kuyruğa koyma
        kararı için)."""
        st = self.stats([queue])[queue]
        return st["bekleyen"] == 0 and st["islenmekte"] == 0

    def live_consumers(self, queue: str) -> int:
        """Çalışan işçi sayısı: son 30 sn'de varlık anahtarını yenileyenler."""
        return sum(1 for _ in self.r.scan_iter(match=presence_key(queue, "*"), count=100))

    def _oldest_age(self, stream: str, group) -> float:
        """Henüz hiçbir işçiye verilmemiş en eski mesajın yaşı (sn): kuyruk tıkanmasını gösterir.
        Mesaj kimliğinin ilk kısmı eklenme zamanıdır (ms)."""
        start = "-"
        if group is not None:
            last = _s(group.get("last-delivered-id") or "0-0")
            start = "(" + last
        first = self.r.xrange(stream, min=start, max="+", count=1)
        if not first:
            return 0.0
        ms = int(_s(first[0][0]).split("-")[0])
        return round(max(0.0, time.time() - ms / 1000), 1)


class StreamConsumer:
    """Bir kuyruğu tüketen döngü. handler(job_id) işi yapar; hata fırlatırsa yeniden denenir."""

    def __init__(self, client: redis.Redis, queue: str, handler: Callable[[str], object],
                 on_give_up: Callable[[str, str], None], max_attempts: int = 3,
                 claim_idle_ms: int = 120_000, heartbeat_s: float = 30.0, block_ms: int = 2000,
                 consumer: str | None = None):
        self.r, self.queue, self.stream = client, queue, stream_name(queue)
        self.handler, self.on_give_up = handler, on_give_up
        self.max_attempts, self.claim_idle_ms = max_attempts, claim_idle_ms
        self.heartbeat_s, self.block_ms = heartbeat_s, block_ms
        self.consumer = consumer or consumer_name()
        self._stop = threading.Event()

    def ensure_group(self) -> None:
        try:
            self.r.xgroup_create(self.stream, GROUP, id="0", mkstream=True)
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    def stop(self) -> None:
        self._stop.set()

    # --- mesaj alma ---
    def _claim_orphans(self) -> list[tuple[str, dict]]:
        """Çöken worker'lardan kalan (claim_idle'dan uzun süredir onaysız) mesajlar."""
        res = self.r.xautoclaim(self.stream, GROUP, self.consumer, self.claim_idle_ms,
                                start_id="0-0", count=1)
        msgs = res[1] if res and len(res) > 1 else []
        return [(m[0], m[1]) for m in msgs if m and m[1]]

    def _read_new(self) -> list[tuple[str, dict]]:
        res = self.r.xreadgroup(GROUP, self.consumer, {self.stream: ">"}, count=1,
                                block=self.block_ms)
        return [(m[0], m[1]) for _, msgs in (res or []) for m in msgs]

    def _deliveries(self, msg_id: str) -> int:
        p = self.r.xpending_range(self.stream, GROUP, min=msg_id, max=msg_id, count=1)
        return int(p[0]["times_delivered"]) if p else 1

    # --- işleme ---
    def _heartbeat(self, msg_id: str, done: threading.Event) -> None:
        while not done.wait(self.heartbeat_s):
            try:
                self.r.xclaim(self.stream, GROUP, self.consumer, 0, [msg_id], justid=True)
            except redis.RedisError:
                log.warning("heartbeat başarısız: %s", msg_id)

    def process(self, msg_id: str, fields: dict) -> None:
        job_id = _s(fields.get("job_id") or fields.get(b"job_id"))
        attempt = int(_s(fields.get("attempt") or fields.get(b"attempt") or "1"))
        # bir mesajın kaç kez teslim edildiği + daha önce yeniden kuyruğa eklenme sayısı
        tries = attempt + self._deliveries(msg_id) - 1
        if tries > self.max_attempts:
            self._give_up(msg_id, job_id, f"en fazla deneme ({self.max_attempts}) aşıldı")
            return
        done = threading.Event()
        hb = threading.Thread(target=self._heartbeat, args=(msg_id, done), daemon=True)
        hb.start()
        try:
            self.handler(job_id)
        except Exception as e:
            log.exception("iş hata verdi job=%s deneme=%d", job_id, tries)
            if tries < self.max_attempts:
                # hemen yeniden dene: yeni mesaj, eskisi onaylanır
                self.r.xadd(self.stream, {"job_id": job_id, "attempt": str(tries + 1)})
            else:
                self._give_up(msg_id, job_id, f"{type(e).__name__}: {e}", ack=False)
        finally:
            done.set()
            hb.join(timeout=1)
        self.r.xack(self.stream, GROUP, msg_id)

    def _give_up(self, msg_id: str, job_id: str, reason: str, ack: bool = True) -> None:
        log.error("iş bırakıldı job=%s: %s", job_id, reason)
        self.r.xadd(DLQ, {"job_id": job_id, "queue": self.queue, "reason": reason[:500],
                          "at": str(int(time.time()))}, maxlen=100_000, approximate=True)
        self.on_give_up(job_id, reason)
        if ack:
            self.r.xack(self.stream, GROUP, msg_id)

    def run_once(self) -> int:
        msgs = self._claim_orphans() or self._read_new()
        for msg_id, fields in msgs:
            self.process(msg_id, fields)
        return len(msgs)

    def _presence(self, every: float = 10.0, ttl: int = 30) -> None:
        """'Bu işçi çalışıyor' anahtarı (hazır olma kontrolü). Ayrı iş parçacığında: uzun bir
        iş işlenirken de yenilenir. Redis 7.0'da XINFO CONSUMERS bunun yerine geçemiyor
        (tüketici ancak ilk mesajı alınca görünüyor, idle da son teslimden sayılıyor)."""
        key = presence_key(self.queue, self.consumer)
        while not self._stop.is_set():
            try:
                self.r.set(key, str(int(time.time())), ex=ttl)
            except redis.ConnectionError:
                pass
            self._stop.wait(every)
        try:
            self.r.delete(key)
        except redis.ConnectionError:
            pass

    def run_forever(self) -> None:
        self.ensure_group()
        presence = threading.Thread(target=self._presence, daemon=True, name="varlik")
        presence.start()
        log.info("kuyruk dinleniyor: %s (tüketici %s)", self.stream, self.consumer)
        while not self._stop.is_set():
            try:
                self.run_once()
            except redis.ConnectionError:
                log.warning("Redis'e ulaşılamıyor, 5 sn sonra tekrar")
                self._stop.wait(5)
        presence.join(timeout=2)  # varlık anahtarı silinsin


def _s(v) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


class RedisLock:
    """Tek sahipli kilit (ör. bakım süreci birden çok kopyada çalışırsa yalnız biri çalışsın).

    Yenileme ve bırakma Lua ile atomik: "kilit hâlâ bende mi" kontrolü ile süre uzatma tek
    adımdır. Projedeki süpürücüde GET ile EXPIRE ayrıydı; arada kilidin süresi dolup başkası
    alırsa, eski sahip yenisinin kilidini uzatabiliyordu.
    """

    _RENEW = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('pexpire', KEYS[1], ARGV[2]) else return 0 end"
    _RELEASE = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end"

    def __init__(self, client, name: str, ttl_s: float = 120.0):
        import uuid

        self.r, self.name, self.ttl_ms = client, name, int(ttl_s * 1000)
        self.token = uuid.uuid4().hex
        self.held = False

    def acquire(self) -> bool:
        """Kilidi al ya da (zaten bendeyse) süresini uzat. Sahibi başkasıysa False."""
        if self.held and self.r.eval(self._RENEW, 1, self.name, self.token, self.ttl_ms):
            return True
        self.held = bool(self.r.set(self.name, self.token, nx=True, px=self.ttl_ms))
        return self.held

    def release(self) -> None:
        self.r.eval(self._RELEASE, 1, self.name, self.token)
        self.held = False
