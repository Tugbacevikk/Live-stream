"""Yerel mod kuyruğu: Redis/Celery yerine aynı süreçte iş parçacığı havuzları.

İki havuz: dil tespiti (1 iş parçacığı) ve metne çevirme (çekirdek / ONNX_THREADS).
Modeller süreç içinde bir kez yüklenir ve iş parçacıkları arasında paylaşılır
(ONNX Runtime oturumları aynı anda çağrılabilir).
Süreç kapanırsa kuyruktaki işler kaybolmaz: kayıtları veritabanında QUEUED/RUNNING
olarak kalır ve yerel sunucu açılışta onları yeniden kuyruğa alır.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger(__name__)


class ThreadJobQueue:
    def __init__(self, asr_workers: int, lid_workers: int = 1, max_attempts: int = 3):
        self._lid = ThreadPoolExecutor(lid_workers, thread_name_prefix="lid")
        self._asr = ThreadPoolExecutor(asr_workers, thread_name_prefix="asr")
        self._detect: Callable[[str], object] | None = None
        self._run: Callable[[str], object] | None = None
        self._give_up: Callable[[str, str], None] = lambda job_id, err: None
        self.max_attempts = max_attempts
        # ölçüm (/metrics): kuyrukta bekleyen işin sıraya giriş zamanı, işlenen sayısı, bırakılan
        self._lock = threading.Lock()
        self._waiting: dict[str, dict[str, float]] = {"lid": {}, "asr": {}}
        self._running = {"lid": 0, "asr": 0}
        self.given_up = 0

    def bind(self, detect: Callable[[str], object], run: Callable[[str], object],
             give_up: Callable[[str, str], None] | None = None) -> None:
        """İş akışı nesneleri kuyruğa ihtiyaç duyduğu için kurulumdan sonra bağlanır.
        give_up(job_id, hata): son deneme de başarısız olunca (ör. store.fail)."""
        self._detect, self._run = detect, run
        if give_up is not None:
            self._give_up = give_up

    def _safe(self, kind, fn, job_id):
        with self._lock:
            self._waiting[kind].pop(job_id, None)
            self._running[kind] += 1
        try:
            self._attempts(fn, job_id)
        finally:
            with self._lock:
                self._running[kind] -= 1

    def _attempts(self, fn, job_id):
        for attempt in range(1, self.max_attempts + 1):
            try:
                fn(job_id)
                return
            except Exception as e:  # havuz ölmesin; tekrar dene, sonra bırak
                log.exception("iş hata verdi: %s (deneme %d/%d)", job_id, attempt, self.max_attempts)
                err = f"{type(e).__name__}: {e}"
        self.given_up += 1
        self._give_up(job_id, f"en fazla deneme ({self.max_attempts}) aşıldı; son hata: {err}")

    def _submit(self, kind, pool, fn, job_id):
        with self._lock:
            self._waiting[kind].setdefault(job_id, time.time())
        pool.submit(self._safe, kind, fn, job_id)

    def enqueue_detection(self, job_id: str) -> None:
        self._submit("lid", self._lid, self._detect, job_id)

    def enqueue_transcription(self, job_id: str, language: str) -> None:
        self._submit("asr", self._asr, self._run, job_id)

    def stats(self, queues=None) -> dict[str, dict[str, float]]:
        """/metrics için: kuyruk başına bekleyen, işlenmekte, en eski bekleyenin yaşı (sn)."""
        now = time.time()
        with self._lock:
            out = {k: {"bekleyen": len(w), "islenmekte": self._running[k],
                       "en_eski_sn": round(now - min(w.values()), 1) if w else 0.0}
                   for k, w in self._waiting.items()}
        out["birakilan"] = {"toplam": self.given_up}
        return out

    def shutdown(self, wait: bool = True) -> None:
        self._lid.shutdown(wait=wait, cancel_futures=not wait)
        self._asr.shutdown(wait=wait, cancel_futures=not wait)
