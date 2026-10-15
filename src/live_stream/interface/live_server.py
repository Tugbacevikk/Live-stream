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
                      callback_url (oturum bitince bildirim; yoksa LIVE_WEBHOOK_URL; API_KEY ister),
                      key (API_KEY tanımlıysa zorunlu; X-API-Key başlığı da olur),
                      resume + token + after (kopan oturuma devam: ready'deki session_id ve
                      resume_token; after = istemcinin aldığı son kesin satırın id'si)
  Sunucu -> İstemci : {"type": "ready", "session_id": "...", "resume_token": "...",
                       "resumed": false, "language": "tr"|"auto", "device": "cuda", "model": "...",
                       "recording": "..."}
                      (resumed: true ise kaçırılan kesin satırlar ardından yeniden gönderilir)
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

Kopma: bağlantı beklenmedik koparsa oturum hemen kapanmaz, LIVE_RESUME_SEC (30 sn) bekler.
İstemci resume+token ile dönerse aynı oturum (aynı kayıt, satır numaraları, dil) sürer;
dönmezse oturum "disconnect" nedeniyle kapanır (kayıt, toplu iş, bildirim).

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
from live_stream.interface.security import callback_problem, key_ok

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


class _Run:
    """Bir canlı oturum: bağlantıdan bağımsız yaşar (kopup geri gelen bağlantı aynı oturuma bağlanır)."""

    def __init__(self, *, session: LiveSession, lock: LanguageLock, recorder, callback: str | None,
                 analyze: bool, lang: str):
        self.id, self.token = uuid.uuid4().hex, uuid.uuid4().hex
        self.session, self.lock, self.recorder = session, lock, recorder
        self.callback, self.analyze, self.lang = callback, analyze, lang
        self.started_at = _now()
        self.finals: asyncio.Queue[FinalJob | None] = asyncio.Queue()
        self.ws: WebSocket | None = None
        self.send_lock = asyncio.Lock()
        self.partial_busy = False
        self.announced = False
        self.closed = False          # finish başladı: artık bağlanılamaz
        self.expiry: asyncio.Task | None = None
        self.worker: asyncio.Task | None = None
        # gecikme ölçümü: bu bağlantının ilk ses paketi (duvar saati) ve o andaki akış zamanı
        self.t_conn: float | None = None
        self.stream_at_conn = 0.0

    async def send(self, msg: dict | None) -> None:
        ws = self.ws
        if msg is None or ws is None:
            return  # bağlantı yokken: kesin satırlar items'ta kalır, dönünce yeniden gönderilir
        async with self.send_lock:
            try:
                await ws.send_json(msg)
            except Exception:
                pass


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
    runs: dict[str, _Run] = {}   # açık (ya da kopup beklenen) oturumlar
    app.state.live_runs = runs

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
            "auth": bool(settings.api_key),
            "ready": all(e.asr is not None for e in engines.values()),
        }

    async def reject(ws: WebSocket, message: str) -> None:
        await ws.send_json({"type": "error", "message": message})
        await ws.close()

    # ------------------------------------------------------------------ oturum işleri
    async def announce(r: _Run) -> None:
        """'auto' seçildiyse dil bulununca bir kez bildirir."""
        if r.lock.detected and not r.announced:
            r.announced = True
            e = engines[r.lock.code]
            await r.send({"type": "language", "code": r.lock.code, "confidence": r.lock.confidence,
                          "model": e.model_name, "device": e.device})

    async def final_worker(r: _Run) -> None:
        while (job := await r.finals.get()) is not None:
            try:
                msg = await run(r.session.run_final, job)
                await announce(r)
                await r.send(msg)
                if r.t_conn is not None:
                    # gecikme: konuşmanın bittiği an ile metnin hazır olduğu an arası
                    lag = (time.monotonic() - r.t_conn) - (job.end - r.stream_at_conn)
                    await r.send({"type": "stats", "lag_sec": round(max(0.0, lag), 2)})
            except Exception as e:
                log.exception("kesin metin üretilemedi")
                await r.send({"type": "error", "message": str(e)})

    async def partial_task(r: _Run, job) -> None:
        try:
            if r.finals.empty():  # bekleyen kesin metin varsa ara metin gereksiz
                msg = await run(r.session.run_partial, job)
                await announce(r)
                await r.send(msg)
        except Exception as e:
            log.warning("ara metin hatası: %s", e)
        finally:
            r.partial_busy = False

    async def finish(r: _Run, reason: str) -> tuple[str | None, bool]:
        """Kalan cümleleri işler, kaydı kapatır, istenirse toplu işe gönderir ve oturum
        sonu bildirimini yazar. -> (toplu iş kimliği, bildirim yazıldı mı)"""
        r.closed = True
        if r.expiry is not None and r.expiry is not asyncio.current_task():
            r.expiry.cancel()
        runs.pop(r.id, None)
        for job in r.session.segmenter.flush():
            r.finals.put_nowait(job)
        r.finals.put_nowait(None)
        try:
            await asyncio.wait_for(r.worker, timeout=60)
        except Exception:
            r.worker.cancel()
        path = r.session.close()
        job_id = None
        if r.analyze and path and submit_recording and r.session.lines:
            try:  # toplu iş aynı callback'i alır: konuşmacı ayrımı bitince 'job.done'
                job_id = await asyncio.to_thread(submit_recording, path, r.lock.code, r.callback, r.id)
            except Exception as e:
                log.warning("kayıt toplu işe gönderilemedi: %s", e)
        if not (r.callback and notify_ended):
            return job_id, False
        payload = ended_payload(r.id, r.lock, r.session.items, started_at=r.started_at,
                                ended_at=_now(), duration=r.session.segmenter.stream_time,
                                reason=reason, job_id=job_id,
                                recording=r.recorder.name if r.recorder else None)
        try:
            await asyncio.to_thread(notify_ended, r.callback, payload, job_id)
            return job_id, True
        except Exception as e:  # bildirim yazılamadı: oturum yine de düzgün kapansın
            log.exception("oturum sonu bildirimi yazılamadı: %s", e)
            return job_id, False

    def background(coro) -> asyncio.Task:
        """İptalden korunan görev: bağlantı işleyicisi iptal edilse de (sunucu kapanışı,
        istemci kütüphanesinin iptali) kayıt kapanır ve bildirim yazılır."""
        task = asyncio.get_running_loop().create_task(coro)
        _background.add(task)
        task.add_done_callback(_background.discard)
        return task

    async def expire(r: _Run, after: float) -> None:
        await asyncio.sleep(after)
        if r.ws is None and not r.closed:
            log.info("canlı oturum %s: %.0f sn içinde geri bağlanılmadı, kapatılıyor", r.id, after)
            await finish(r, "disconnect")

    def park(r: _Run) -> asyncio.Task | None:
        """Bağlantı koptu: oturum LIVE_RESUME_SEC bekler; dönülmezse kapanır."""
        r.ws = None
        if r.closed:
            return None
        if settings.live_resume_sec <= 0:
            return background(finish(r, "disconnect"))
        r.expiry = background(expire(r, settings.live_resume_sec))
        return None

    # ------------------------------------------------------------------ bağlantı
    async def new_run(ws: WebSocket, q) -> _Run | None:
        try:
            seg_cfg = SegmenterConfig(
                end_silence_ms=int(q.get("silence_ms", settings.end_silence_ms)),
                max_utterance_sec=float(q.get("max_sec", settings.max_utterance_sec)),
                partial_interval_sec=float(q.get("partial_sec", settings.partial_interval_sec)),
            )
            save = q.get("save", "1" if settings.save_recordings else "0") == "1"
            analyze = save and q.get("analyze", "1" if settings.live_auto_analyze else "0") == "1"
        except ValueError:
            await reject(ws, "Geçersiz ayar değeri.")
            return None
        asked = (q.get("callback_url") or "").strip() or None
        if asked and asked != settings.live_webhook_url and not settings.api_key:
            # anahtarsız biri sunucuya istediği adrese istek attıramasın
            await reject(ws, "Oturum başına callback_url için sunucuda API_KEY tanımlı olmalı "
                             "(ya da adresi LIVE_WEBHOOK_URL ile verin).")
            return None
        callback = asked or settings.live_webhook_url or None
        problem = callback_problem(callback, settings)
        if problem:
            await reject(ws, problem)
            return None
        lang = q.get("language", default).strip().lower()
        if lang == "auto" and (len(languages) == 1 or detect_language is None):
            lang = languages[0]  # tek dil açık: tespit gereksiz
        if lang != "auto" and lang not in engines:
            await reject(ws, f"'{lang}' dili açık değil; seçenekler: {', '.join(languages)}")
            return None
        try:  # dil belliyse onun modeli, 'auto' ise hepsi hazır olsun
            for code in ([lang] if lang != "auto" else languages):
                await engines[code].wait_ready()
        except Exception as e:
            await reject(ws, f"Model yüklenemedi: {e}")
            return None

        lock = LanguageLock(None if lang == "auto" else lang, detect_language, fallback=languages[0])

        def transcribe(audio, fast, prompt):
            code = lock.decide(audio, final=not fast)
            return "" if code is None else engines[code].transcribe(audio, fast, prompt)

        detector = await asyncio.to_thread(detector_factory)
        recorder = SessionRecorder(settings.recordings_dir) if save else None
        r = _Run(session=LiveSession(Segmenter(detector, seg_cfg), transcribe, recorder), lock=lock,
                 recorder=recorder, callback=callback, analyze=analyze, lang=lang)
        r.worker = asyncio.create_task(final_worker(r))
        runs[r.id] = r
        return r

    def ready_msg(r: _Run, resumed: bool) -> dict:
        code = r.lock.code
        e = engines[code] if code else None
        return {"type": "ready", "session_id": r.id, "resume_token": r.token, "resumed": resumed,
                "language": r.lang if not resumed or not code else code,
                "device": e.device if e else None, "model": e.model_name if e else None,
                "recording": r.recorder.name if r.recorder else None}

    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        q = ws.query_params
        if not key_ok(settings, q.get("key") or ws.headers.get("x-api-key")):
            await reject(ws, "Geçersiz ya da eksik API anahtarı (sayfadaki Ayarlar > API anahtarı).")
            return

        r, resumed = None, False
        old = runs.get(q.get("resume") or "")
        if old is not None and not old.closed and old.ws is None and q.get("token") == old.token:
            if old.expiry is not None:
                old.expiry.cancel()
                old.expiry = None
            r, resumed = old, True
            log.info("canlı oturum %s: bağlantı geri geldi", r.id)
        if r is None:
            r = await new_run(ws, q)
            if r is None:
                return
        r.ws = ws
        r.t_conn = None
        await r.send(ready_msg(r, resumed))
        if resumed:  # kopukken kesinleşen satırlar
            try:
                after = int(q.get("after", "-1"))
            except ValueError:
                after = -1
            for item in r.session.items:
                if item["id"] > after:
                    await r.send({"type": "final", **item})

        finished = False
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    if r.t_conn is None:
                        r.t_conn, r.stream_at_conn = time.monotonic(), r.session.segmenter.stream_time
                    for job in r.session.feed_pcm16(msg["bytes"]):
                        if isinstance(job, FinalJob):
                            r.finals.put_nowait(job)
                        elif not r.partial_busy:  # yetişemiyorsak ara metni atla, gecikme birikmesin
                            r.partial_busy = True
                            asyncio.create_task(partial_task(r, job))
                elif msg.get("text"):
                    try:
                        data = json.loads(msg["text"])
                    except json.JSONDecodeError:
                        continue
                    if data.get("type") == "stop":
                        finished = True
                        job_id, notified = await finish(r, "stop")
                        await r.send({"type": "done", "session_id": r.id, "job_id": job_id,
                                      "webhook": notified})
                        await ws.close()
                        break
        except WebSocketDisconnect:
            pass
        finally:
            if not finished and r.ws is ws:
                task = park(r)
                if task is not None:
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        pass

    app.add_api_websocket_route("/ws/live", ws_endpoint)
    app.add_api_websocket_route("/ws", ws_endpoint)
