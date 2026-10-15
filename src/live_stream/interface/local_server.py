"""Yerel mod: tek bilgisayarda, tek komutla tüm sistem (Redis, PostgreSQL, Celery GEREKMEZ).

    python -m live_stream                                       # http://localhost:8000
    python -m live_stream --port 9000 --workers 2

    http://localhost:8000/        dosya yükle -> konuşmacılı transkript (toplu iş)
    http://localhost:8000/canli   canlı altyazı (mikrofon); oturum bitince kayıt toplu işe gider

Aynı API (POST /jobs, GET /jobs/{id}, /health) ve aynı iş akışı; farkı:
  - veritabanı: SQLite dosyası (DATA_DIR/jobs.db)
  - kuyruk: süreç içi iş parçacığı havuzları
  - webhook göndericisi: arka planda bir iş parçacığı
Açılışta yarım kalmış işler (QUEUED/RUNNING) yeniden kuyruğa alınır.
Windows'ta da çalışır (Celery Windows'u desteklemiyor, bu mod onun yerine).
"""
from __future__ import annotations

import argparse
import logging
import threading
import time

import uvicorn

import uuid

from live_stream.application.jobs import DetectLanguage, RunTranscription, SubmitJob
from live_stream.domain import AudioJob
from live_stream.infrastructure.audio import SoundfileLoader
from live_stream.infrastructure.sherpa_vad import SileroVad
from live_stream.infrastructure.thread_queue import ThreadJobQueue
from live_stream.infrastructure.webhook import OutboxDispatcher
from live_stream.interface.api import create_app
from live_stream.interface.composition import (
    build_live, build_maintenance, build_pipeline, build_resolver, config_from_settings,
    store_from_settings, verify_models_or_exit,
)
from live_stream.interface.live_server import add_live_routes
from live_stream.interface.settings import Settings

log = logging.getLogger(__name__)


def build_local(settings: Settings, workers: int | None = None, warm_up: bool = True,
                live_engine=None, live_detector=None):
    """Yerel sistemi kurar; (app, store, queue, dispatcher) döndürür (testler de kullanır).
    live_engine/live_detector: testler için sahte canlı motor ve algılayıcı."""
    store = store_from_settings(settings, url=settings.local_db)
    store.migrate()
    n = workers or settings.concurrency
    cfg = config_from_settings(settings, workers=n)
    queue = ThreadJobQueue(asr_workers=n,
                           max_attempts=settings.max_job_attempts)
    pipeline = build_pipeline(cfg, store, with_detection=False)
    detect = DetectLanguage(store, queue, SoundfileLoader(), SileroVad(cfg.vad), build_resolver(cfg))
    queue.bind(detect, RunTranscription(store, pipeline), give_up=store.fail)
    if warm_up:  # ilk işte model yükleme gecikmesi olmasın
        for lang in settings.languages:
            pipeline.engines[lang]

    for rec in store.unfinished():  # önceki çalıştırmadan yarım kalanlar
        code = rec.decision.code if rec.decision else rec.job.language
        if not code and len(settings.languages) == 1:
            code = settings.languages[0]
        if code:
            queue.enqueue_transcription(rec.job.job_id, code)
        else:
            queue.enqueue_detection(rec.job.job_id)
        log.info("yarım kalan iş yeniden kuyrukta: %s", rec.job.job_id)

    dispatcher = OutboxDispatcher(store, secret=settings.webhook_secret,
                                  max_attempts=settings.webhook_max_attempts)
    def models_ready():
        loaded = set(pipeline.engines.loaded())
        missing = [lang for lang in settings.languages if lang not in loaded]
        return not missing, (f"yüklenmemiş: {missing}" if missing else f"yüklü: {sorted(loaded)}")

    app = create_app(store, queue, settings, {"veritabani": store.ping}, queue_stats=queue.stats,
                     readiness={"modeller": models_ready})
    app.state.maintenance = build_maintenance(store, settings)

    # canlı altyazı: aynı süreç, aynı model; oturum kaydı bitince toplu işe gider
    engine, detector = build_live(settings, cfg, pipeline.engines)
    if live_engine is not None:
        engine = live_engine
    submit = SubmitJob(store, queue, settings.languages)

    def submit_recording(path: str) -> str:
        return submit(AudioJob(uuid.uuid4().hex, path, settings.languages[0]))

    add_live_routes(app, settings, engine, live_detector or detector, submit_recording)
    if warm_up:  # canlı model açılışta arka planda yüklenir (ilk konuşmada beklenmesin)
        engine.start_loading()

    return app, store, queue, dispatcher


def main(argv=None):
    p = argparse.ArgumentParser(description="Yerel mod: tek komutla tüm sistem")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--workers", type=int, default=None, help="aynı anda işlenen ses sayısı")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    s = Settings()
    verify_models_or_exit(s)
    app, store, queue, dispatcher = build_local(s, a.workers)
    threading.Thread(target=dispatcher.run_forever, daemon=True, name="webhook").start()

    def bakim():
        while True:
            try:
                app.state.maintenance.run_once()
            except Exception:
                log.exception("bakım turu başarısız")
            time.sleep(60)

    threading.Thread(target=bakim, daemon=True, name="bakim").start()
    n, why = (a.workers, "--workers") if a.workers else s.concurrency_plan()
    log.info("hazır: http://%s:%d  (diller: %s, eşzamanlı iş: %d — %s)", a.host, a.port,
             ",".join(s.languages), n, why)
    log.info("canlı altyazı: http://%s:%d/canli", a.host, a.port)
    try:
        uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    finally:
        dispatcher.stop()
        queue.shutdown(wait=False)


if __name__ == "__main__":
    main()
