"""İstemci başına dakikalık istek sınırı (fine-tuning projesindeki Redis sınırlayıcının karşılığı).

Sabit pencere: her istemci için "bu dakikada kaç istek" sayılır; sınır aşılırsa dakikanın
bitmesine kalan süre döner (HTTP 429 + Retry-After). Tam modda sayaç Redis'te (birden çok API
süreci aynı sayacı görür), yerel modda bellekte.
"""
from __future__ import annotations

import threading
import time


class MemoryRateLimiter:
    def __init__(self, per_minute: int, clock=time.time):
        self.per_minute, self.clock = per_minute, clock
        self._counts: dict[tuple[str, int], int] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> int | None:
        """İzin varsa None; yoksa beklenecek saniye."""
        now = self.clock()
        window = int(now // 60)
        with self._lock:
            if len(self._counts) > 10_000:  # eski pencereleri at
                self._counts = {k: v for k, v in self._counts.items() if k[1] == window}
            n = self._counts.get((key, window), 0) + 1
            self._counts[(key, window)] = n
        return None if n <= self.per_minute else int(60 - now % 60) + 1


class RedisRateLimiter:
    def __init__(self, client, per_minute: int, clock=time.time, prefix: str = "asr:rl"):
        self.r, self.per_minute, self.clock, self.prefix = client, per_minute, clock, prefix

    def check(self, key: str) -> int | None:
        now = self.clock()
        k = f"{self.prefix}:{key}:{int(now // 60)}"
        pipe = self.r.pipeline()
        pipe.incr(k)
        pipe.expire(k, 120)
        n = pipe.execute()[0]
        return None if n <= self.per_minute else int(60 - now % 60) + 1
