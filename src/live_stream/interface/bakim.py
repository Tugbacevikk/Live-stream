"""Bakım süreci (tam modda; yerel modda local_server bunu kendi içinde çalıştırır).

    python -m live_stream.interface.bakim            # her 60 sn

Görevler: saklama süreleri, askıda kalan işler, kuyruğa girememiş işlerin yeniden kuyruğa
konması. Birden çok kopya çalıştırılırsa Redis kilidi sayesinde aynı anda yalnız biri çalışır
(diğerleri bekler, sahibi çökerse kilit 2 dk sonra boşalır ve biri devralır).
"""
from __future__ import annotations

import logging
import time

import redis

from live_stream.infrastructure.redis_streams import RedisLock, RedisStreamJobQueue
from live_stream.interface.composition import build_maintenance, store_from_settings
from live_stream.interface.settings import Settings

log = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s = Settings()
    store = store_from_settings(s, pool_size=1)
    store.migrate()
    r = redis.Redis.from_url(s.redis_url)
    queue = RedisStreamJobQueue(r)
    m = build_maintenance(store, s, queue=queue, queue_idle=queue.is_idle)
    lock = RedisLock(r, "asr:bakim:kilit", ttl_s=120)
    while True:
        try:
            if lock.acquire():
                m.run_once()
            else:
                log.debug("bakımı başka bir kopya yürütüyor")
        except Exception:
            log.exception("bakım turu başarısız")
        time.sleep(60)


if __name__ == "__main__":
    main()
