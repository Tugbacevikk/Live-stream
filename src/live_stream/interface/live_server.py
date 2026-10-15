"""Canlı altyazı uçları (yerel sunucuya eklenir; aynı süreç, toplu işle aynı modeller).

    GET /canli        -> altyazı ekranı (sunum modu: F, yazı boyu: + / -)
    GET /live/health  -> diller ve modellerin durumu
    WS  /ws/live      -> canlı ses akışı  (/ws: ilk sürümün istemcileri için aynı uç)

WebSocket protokolü
  İstemci -> Sunucu : ikili mesaj = Int16 little-endian PCM, 16 kHz, mono
                      metin mesajı {"type": "stop"} = son parçayı işle ve kapat
                      sorgu parametreleri (isteğe bağlı): language (tr | vi | auto),
                      silence_ms, partial_sec, max_sec, save (1/0),
                      analyze (1/0: oturum sonu toplu işe gönder)
  Sunucu -> İstemci : {"type": "ready", "language": "tr"|"auto", "device": "cuda", "model": "...",
                       "recording": "..."}
                      {"type": "language", "code": "vi", "confidence": 0.97, "model": "...",
                       "device": "cpu"}   ('auto' seçildiyse ilk cümleden sonra bir kez)
                      {"type": "partial", "start": 3.1, "text": "..."}
                      {"type": "final", "id": 0, "start": 3.1, "end": 5.4, "text": "..."}
                      {"type": "discard", "start": 3.1}
                      {"type": "stats", "lag_sec": 0.4}
                      {"type": "error", "message": "..."}
                      {"type": "done", "job_id": "..." | null}  job_id: konuşmacı ayrımlı
                      transkript için açılan toplu iş (GET /jobs/{job_id})
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from live_stream.application.live import FinalJob, LanguageLock, LiveSession, Segmenter, SegmenterConfig
from live_stream.infrastructure.recorder import SessionRecorder

log = logging.getLogger(__name__)
_PAGE = Path(__file__).parent / "static" / "canli.html"
LANG_NAMES = {"tr": "Türkçe", "vi": "Vietnamca"}


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


def add_live_routes(app: FastAPI, settings, engines: dict[str, LiveEngine], detector_factory: Callable,
                    submit_recording: Callable[[str, str | None], str | None] | None = None,
                    detect_language: Callable | None = None) -> None:
    """engines: dil kodu -> motor. detect_language(ses) -> LanguageDecision ('auto' için).
    submit_recording(ses yolu, dil) -> toplu iş kimliği: oturum sonu konuşmacı ayrımı için."""
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

        async def finish() -> str | None:
            """Kalan cümleleri işler, kaydı kapatır, istenirse toplu işe gönderir."""
            for job in session.segmenter.flush():
                finals.put_nowait(job)
            finals.put_nowait(None)
            try:
                await asyncio.wait_for(worker, timeout=60)
            except Exception:
                worker.cancel()
            path = session.close()
            if not (analyze and path and submit_recording and session.lines):
                return None
            try:
                return await asyncio.to_thread(submit_recording, path, lock.code)
            except Exception as e:
                log.warning("kayıt toplu işe gönderilemedi: %s", e)
                return None

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
                        job_id = await finish()
                        await send({"type": "done", "job_id": job_id})
                        await ws.close()
                        break
        except WebSocketDisconnect:
            pass
        finally:
            if not finished:
                # bağlantı koptu: kalan cümleler yine de kayda geçer ve analiz edilir
                await finish()

    app.add_api_websocket_route("/ws/live", ws_endpoint)
    app.add_api_websocket_route("/ws", ws_endpoint)
