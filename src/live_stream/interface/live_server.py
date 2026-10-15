"""Canlı altyazı uçları (yerel sunucuya eklenir; aynı süreç, toplu işle aynı modeller).

    GET /canli        -> altyazı ekranı (sunum modu: F, yazı boyu: + / -)
    GET /live/health  -> diller ve modellerin durumu
    WS  /ws/live      -> canlı ses akışı  (/ws: ilk sürümün istemcileri için aynı uç)

WebSocket protokolü
  İstemci -> Sunucu : ikili mesaj = Int16 little-endian PCM, 16 kHz, mono
                      metin mesajı {"type": "stop"} = son parçayı işle ve kapat
                      sorgu parametreleri (isteğe bağlı): language (tr | vi | auto),
                      silence_ms, partial_sec, max_sec, save (1/0),
                      analyze (1/0: oturum sonu toplu işe gönder),
                      callback_url (oturum bitince bildirim; yoksa LIVE_WEBHOOK_URL)
  Sunucu -> İstemci : {"type": "ready", "language": "tr"|"auto", "device": "cuda", "model": "...",
                       "recording": "..."}
                      {"type": "language", "code": "vi", "confidence": 0.97, "model": "...",
                       "device": "cpu"}   ('auto' seçildiyse ilk cümleden sonra bir kez)
                      {"type": "partial", "start": 3.1, "text": "..."}
                      {"type": "final", "id": 0, "start": 3.1, "end": 5.4, "text": "..."}
                      {"type": "discard", "start": 3.1}
                      {"type": "stats", "lag_sec": 0.4}
                      {"type": "error", "message": "..."}
                      {"type": "done", "session_id": "...", "job_id": "..." | null,
                       "webhook": true}  job_id: konuşmacı ayrımlı transkript için açılan toplu
                      iş (GET /jobs/{job_id}); webhook: oturum sonu bildirimi kuyruğa yazıldı

Oturum sonu bildirimi (webhook): oturum bitince (Durdur ya da bağlantı kopması) callback
adresine 'live.ended' gider (canlı metin, dil, süre, job_id); konuşmacı ayrımı bitince aynı
adrese toplu işin 'job.done' bildirimi gider. İkisi de outbox'tan, imzalı ve tekrar denemeli
(docs/entegrasyon.md).
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from live_stream.application.live import (
    FinalJob, LanguageLock, LiveSession, Segmenter, SegmenterConfig, ended_payload,
)
from live_stream.infrastructure.recorder import SessionRecorder
from live_stream.interface.security import callback_problem

log = logging.getLogger(__name__)
_PAGE = Path(__file__).parent / "static" / "canli.html"
LANG_NAMES = {"tr": "Türkçe", "vi": "Vietnamca"}
_background: set[asyncio.Task] = set()  # iptalden korunan oturum sonu görevleri


class LiveEngine:
    """Bir dilin canlı ASR motoru: arka planda yüklenir (ilk konuşmada beklenmesin)."""

    def __init__(self, load: Callable[[], object], model_name: str, normalizer=None):
        self._load_fn = load
        self.model_name = model_name
        self.normalizer = normalizer
        self.asr = None
        self.load_error: str | None = None
        self._thread: threading.Thread | None = None
        self._start_lock = threading.Lock()

    @property
    def device(self) -> str | None:
        return getattr(self.asr, "device", None) or ("cpu" if self.asr is not None else None)

    def start_loading(self) -> None:
        with self._start_lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._load, daemon=True, name="canli-model")
                self._thread.start()

    def _load(self) -> None:
        try:
            self.asr = self._load_fn()
        except Exception as e:
            self.load_error = f"{type(e).__name__}: {e}"
            log.exception("canlı altyazı modeli yüklenemedi (%s)", self.model_name)

    def ensure_loaded(self) -> None:
        """Yüklenene kadar bekler (iş parçacığında çağrılır); yüklenemediyse hata."""
        self.start_loading()
        self._thread.join()
        if self.load_error:
            raise RuntimeError(self.load_error)

    async def wait_ready(self) -> None:
        await asyncio.to_thread(self.ensure_loaded)

    def transcribe(self, audio, fast: bool, prompt: str | None) -> str:
        self.ensure_loaded()
        return self.asr.transcribe_text(audio, fast=fast, prompt=prompt, normalizer=self.normalizer)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def add_live_routes(app: FastAPI, settings, engines: dict[str, LiveEngine], detector_factory: Callable,
                    submit_recording: Callable[..., str | None] | None = None,
                    detect_language: Callable | None = None,
                    notify_ended: Callable[[str, dict, str | None], None] | None = None) -> None:
    """engines: dil kodu -> motor. detect_language(ses) -> LanguageDecision ('auto' için).
    submit_recording(ses yolu, dil, callback_url, session_id) -> toplu iş kimliği: oturum sonu
    konuşmacı ayrımı için. notify_ended(url, gövde, job_id): 'live.ended' bildirimini outbox'a yazar."""
    languages = list(engines)
    default = settings.live_language if settings.live_language in (*languages, "auto") else languages[0]
    infer_lock = asyncio.Lock()  # çıkarımlar sırayla: birden çok canlı bağlantı GPU'yu boğmasın

    async def run(fn, job):
        async with infer_lock:
            return await asyncio.to_thread(fn, job)

    @app.get("/canli", include_in_schema=False)
    def live_page():
        return FileResponse(_PAGE, media_type="text/html; charset=utf-8")

    @app.get("/live/health")
    def live_health():
        return {
            "languages": [{"code": c, "name": LANG_NAMES.get(c, c), "model": e.model_name,
                           "ready": e.asr is not None, "error": e.load_error, "device": e.device}
                          for c, e in engines.items()],
            "default": default,
            "auto": len(languages) > 1 and detect_language is not None,
            "webhook": bool(settings.live_webhook_url and notify_ended),
            "ready": all(e.asr is not None for e in engines.values()),
        }

    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        q = ws.query_params
        try:
            seg_cfg = SegmenterConfig(
                end_silence_ms=int(q.get("silence_ms", settings.end_silence_ms)),
                max_utterance_sec=float(q.get("max_sec", settings.max_utterance_sec)),
                partial_interval_sec=float(q.get("partial_sec", settings.partial_interval_sec)),
            )
            save = q.get("save", "1" if settings.save_recordings else "0") == "1"
            analyze = save and q.get("analyze", "1" if settings.live_auto_analyze else "0") == "1"
        except ValueError:
            await ws.send_json({"type": "error", "message": "Geçersiz ayar değeri."})
            await ws.close()
            return
        callback = (q.get("callback_url") or settings.live_webhook_url or "").strip() or None
        problem = callback_problem(callback, settings)
        if problem:
            await ws.send_json({"type": "error", "message": problem})
            await ws.close()
            return
        lang = q.get("language", default).strip().lower()
        if lang == "auto" and (len(languages) == 1 or detect_language is None):
            lang = languages[0]  # tek dil açık: tespit gereksiz
        if lang != "auto" and lang not in engines:
            await ws.send_json({"type": "error", "message":
                                f"'{lang}' dili açık değil; seçenekler: {', '.join(languages)}"})
            await ws.close()
            return

        try:  # dil belliyse onun modeli, 'auto' ise hepsi hazır olsun
            for code in ([lang] if lang != "auto" else languages):
                await engines[code].wait_ready()
        except Exception as e:
            await ws.send_json({"type": "error", "message": f"Model yüklenemedi: {e}"})
            await ws.close()
            return

        lock = LanguageLock(None if lang == "auto" else lang, detect_language, fallback=languages[0])
        session_id, started_at = uuid.uuid4().hex, _now()

        def transcribe(audio, fast, prompt):
            code = lock.decide(audio, final=not fast)
            return "" if code is None else engines[code].transcribe(audio, fast, prompt)

        detector = await asyncio.to_thread(detector_factory)
        recorder = SessionRecorder(settings.recordings_dir) if save else None
        session = LiveSession(Segmenter(detector, seg_cfg), transcribe, recorder)

        finals: asyncio.Queue[FinalJob | None] = asyncio.Queue()
        send_lock = asyncio.Lock()
        partial_busy = False
        announced = False
        t_start: float | None = None  # ilk ses paketinin geldiği an

        async def send(msg: dict | None):
            if msg is None:
                return
            async with send_lock:
                try:
                    await ws.send_json(msg)
                except Exception:
                    pass

        async def announce():
            """'auto' seçildiyse dil bulununca bir kez bildirir."""
            nonlocal announced
            if lock.detected and not announced:
                announced = True
                e = engines[lock.code]
                await send({"type": "language", "code": lock.code, "confidence": lock.confidence,
                            "model": e.model_name, "device": e.device})

        async def final_worker():
            while (job := await finals.get()) is not None:
                try:
                    msg = await run(session.run_final, job)
                    await announce()
                    await send(msg)
                    # gecikme: konuşmanın bittiği an ile metnin hazır olduğu an arası
                    lag = (time.monotonic() - (t_start or time.monotonic())) - job.end
                    await send({"type": "stats", "lag_sec": round(max(0.0, lag), 2)})
                except Exception as e:
                    log.exception("kesin metin üretilemedi")
                    await send({"type": "error", "message": str(e)})

        async def partial_task(job):
            nonlocal partial_busy
            try:
                if finals.empty():  # bekleyen kesin metin varsa ara metin gereksiz
                    msg = await run(session.run_partial, job)
                    await announce()
                    await send(msg)
            except Exception as e:
                log.warning("ara metin hatası: %s", e)
            finally:
                partial_busy = False

        async def finish(reason: str) -> tuple[str | None, bool]:
            """Kalan cümleleri işler, kaydı kapatır, istenirse toplu işe gönderir ve oturum
            sonu bildirimini yazar. -> (toplu iş kimliği, bildirim yazıldı mı)"""
            for job in session.segmenter.flush():
                finals.put_nowait(job)
            finals.put_nowait(None)
            try:
                await asyncio.wait_for(worker, timeout=60)
            except Exception:
                worker.cancel()
            path = session.close()
            job_id = None
            if analyze and path and submit_recording and session.lines:
                try:  # toplu iş aynı callback'i alır: konuşmacı ayrımı bitince 'job.done'
                    job_id = await asyncio.to_thread(submit_recording, path, lock.code, callback, session_id)
                except Exception as e:
                    log.warning("kayıt toplu işe gönderilemedi: %s", e)
            if not (callback and notify_ended):
                return job_id, False
            payload = ended_payload(session_id, lock, session.items, started_at=started_at,
                                    ended_at=_now(), duration=session.segmenter.stream_time,
                                    reason=reason, job_id=job_id,
                                    recording=recorder.name if recorder else None)
            try:
                await asyncio.to_thread(notify_ended, callback, payload, job_id)
                return job_id, True
            except Exception as e:  # bildirim yazılamadı: oturum yine de düzgün kapansın
                log.exception("oturum sonu bildirimi yazılamadı: %s", e)
                return job_id, False

        worker = asyncio.create_task(final_worker())
        first = engines[lang] if lang != "auto" else None
        await send({"type": "ready", "language": lang,
                    "device": first.device if first else None,
                    "model": first.model_name if first else None,
                    "recording": recorder.name if recorder else None})

        finished = False
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    if t_start is None:
                        t_start = time.monotonic()
                    for job in session.feed_pcm16(msg["bytes"]):
                        if isinstance(job, FinalJob):
                            finals.put_nowait(job)
                        elif not partial_busy:  # yetişemiyorsak ara metni atla, gecikme birikmesin
                            partial_busy = True
                            asyncio.create_task(partial_task(job))
                elif msg.get("text"):
                    try:
                        data = json.loads(msg["text"])
                    except json.JSONDecodeError:
                        continue
                    if data.get("type") == "stop":
                        finished = True
                        job_id, notified = await finish("stop")
                        await send({"type": "done", "session_id": session_id, "job_id": job_id,
                                    "webhook": notified})
                        await ws.close()
                        break
        except WebSocketDisconnect:
            pass
        finally:
            if not finished:
                # bağlantı koptu: kalan cümleler yine de kayda geçer, analiz edilir, bildirilir.
                # Ayrı görev + shield: bu işleyici iptal edilse de (sunucu kapanışı, istemci
                # kütüphanesinin iptali) kayıt kapanır ve bildirim yazılır.
                task = asyncio.get_running_loop().create_task(finish("disconnect"))
                _background.add(task)
                task.add_done_callback(_background.discard)
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    pass

    app.add_api_websocket_route("/ws/live", ws_endpoint)
    app.add_api_websocket_route("/ws", ws_endpoint)
