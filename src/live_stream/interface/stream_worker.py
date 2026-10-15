"""Kuyruk worker'ı (Redis Streams; Windows, Linux ve Docker'da aynı komut).

    python -m live_stream.interface.stream_worker --kuyruk lid          # dil tespiti
    python -m live_stream.interface.stream_worker --kuyruk vi           # Vietnamca
    python -m live_stream.interface.stream_worker --kuyruk tr --surec 1 # Türkçe, 1 süreç

--surec verilmezse: lid için 1, ASR kuyrukları için kullanılabilir çekirdek / ONNX_THREADS.
Her süreç modeli bir kez yükler ve kuyruktan tek tek iş alır. Süreçler Windows'ta da
çalışan "spawn" yöntemiyle açılır. Ctrl+C ya da durdur.bat ile düzgün kapanır: elindeki
işi bitirmeden çıkan süreç olursa iş, Redis'te onaysız kalır ve 2 dk sonra başka bir
süreç devralır.
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import signal

import redis

from live_stream.application.jobs import DetectLanguage, RunTranscription
from live_stream.infrastructure.audio import SoundfileLoader
from live_stream.infrastructure.redis_streams import RedisStreamJobQueue, StreamConsumer
from live_stream.infrastructure.sherpa_vad import SileroVad
from live_stream.interface.composition import (
    build_pipeline, build_resolver, config_from_settings, store_from_settings, verify_models_or_exit,
)
from live_stream.interface.settings import Settings

log = logging.getLogger(__name__)


def _config(s: Settings):
    return config_from_settings(s)


def build_consumer(queue: str, s: Settings, client: redis.Redis | None = None,
                   store=None) -> StreamConsumer:
    r = client or redis.Redis.from_url(s.redis_url)
    if store is None:
        store = store_from_settings(s, pool_size=2)
        if store.dialect == "sqlite":  # birden çok süreç aynı SQLite dosyasına güvenle yazamaz
            raise SystemExit("tam mod PostgreSQL ister (DATABASE_URL); tek bilgisayar için baslat.bat")
    cfg = _config(s)
    if queue == "lid":
        handler = DetectLanguage(store, RedisStreamJobQueue(r), SoundfileLoader(),
                                 SileroVad(cfg.vad), build_resolver(cfg))
        for det in handler.resolver.detectors:  # Whisper-small'ı da baştan yükle
            getattr(det, "load", lambda: None)()
    else:
        if queue not in cfg.languages:
            raise SystemExit(f"'{queue}' dili LANGUAGES içinde değil: {list(cfg.languages)}")
        pipeline = build_pipeline(cfg, store, with_detection=False)
        pipeline.engines[queue]  # modeli baştan yükle: ilk işte gecikme olmasın
        handler = RunTranscription(store, pipeline)
    return StreamConsumer(r, queue, handler, on_give_up=store.fail,
                          max_attempts=s.max_job_attempts)


def _run(queue: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(processName)s %(message)s")
    consumer = build_consumer(queue, Settings())

    def stop(*_):
        log.info("kapanıyor (elindeki iş bitince)")
        consumer.stop()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    consumer.run_forever()


def main(argv=None):
    p = argparse.ArgumentParser(description="Kuyruk worker'ı (Redis Streams)")
    p.add_argument("--kuyruk", required=True, help="lid, vi ya da tr")
    p.add_argument("--surec", type=int, default=None, help="süreç sayısı (varsayılan: otomatik)")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(processName)s %(message)s")
    s = Settings()
    verify_models_or_exit(s)
    if a.surec:
        n, why = a.surec, "--surec"
    elif a.kuyruk == "lid":
        n, why = 1, "dil tespiti kısa sürer"
    else:
        n, why = s.concurrency_plan(a.kuyruk)
    log.info("kuyruk=%s süreç=%d ONNX_THREADS=%d (%s)", a.kuyruk, n, s.threads, why)
    if n == 1:
        _run(a.kuyruk)
        return
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_run, args=(a.kuyruk,), name=f"{a.kuyruk}-{i}") for i in range(n)]
    for pr in procs:
        pr.start()
    try:
        for pr in procs:
            pr.join()
    except KeyboardInterrupt:
        for pr in procs:
            pr.join(timeout=60)


if __name__ == "__main__":
    main()
