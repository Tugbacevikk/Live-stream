"""
Canlı altyazı sunucusu.

    GET /          -> altyazı ekranı
    GET /health    -> model durumu
    WS  /ws        -> canlı ses akışı

WebSocket protokolü
  İstemci -> Sunucu : ikili mesaj = Int16 little-endian PCM, 16 kHz, mono
                      metin mesajı {"type": "stop"} = son parçayı işle ve kapat
  Sunucu -> İstemci : {"type": "ready", "device": "cuda", "model": "small", "recording": "..."}
                      {"type": "partial", "start": 3.1, "text": "..."}
                      {"type": "final", "id": 0, "start": 3.1, "end": 5.4, "text": "..."}
                      {"type": "discard", "start": 3.1}
                      {"type": "stats", "lag_sec": 0.4}
                      {"type": "error", "message": "..."} / {"type": "done"}
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from live_stream.config import Settings, get_settings
from live_stream.recorder import SessionRecorder
from live_stream.segmenter import FinalJob, Segmenter, SegmenterConfig
from live_stream.session import LiveSession
from live_stream.transcriber import WhisperTranscriber
from live_stream.vad import make_detector

logger = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"


class Engine:
    """Sunucu ömrü boyunca tek model; çıkarımlar sırayla (GPU/CPU'yu boğmamak için)."""

    def __init__(self, settings: Settings, transcriber=None):
        self.settings = settings
        self.transcriber = transcriber or WhisperTranscriber(settings)
        self.lock = asyncio.Lock()
        self.load_error: str | None = None
        self._load_task: asyncio.Task | None = None

    def start_loading(self) -> None:
        if self._load_task is None:
            self._load_task = asyncio.create_task(self._load())

    async def _load(self) -> None:
        try:
            await asyncio.to_thread(self.transcriber.load)
        except Exception as e:
            self.load_error = str(e)
            logger.exception("Model yüklenemedi")

    async def wait_ready(self) -> None:
        self.start_loading()
        await self._load_task
        if self.load_error:
            raise RuntimeError(self.load_error)

    async def run(self, fn, job):
        async with self.lock:
            return await asyncio.to_thread(fn, job)


def create_app(settings: Settings | None = None, transcriber=None, detector_factory=None) -> FastAPI:
    settings = settings or get_settings()
    engine = Engine(settings, transcriber)
    detector_factory = detector_factory or (lambda: make_detector(settings.vad_engine, settings.vad_threshold))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine.start_loading()  # model açılışta arka planda yüklenir
        yield

    app = FastAPI(title="Canlı Altyazı", lifespan=lifespan)
    app.state.engine = engine

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health():
        t = engine.transcriber
        return {
            "ready": bool(getattr(t, "ready", True)),
            "error": engine.load_error,
            "device": getattr(t, "device", None),
            "model": settings.model_size,
        }

    @app.websocket("/ws")
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
        except ValueError:
            await ws.send_json({"type": "error", "message": "Geçersiz ayar değeri."})
            await ws.close()
            return

        try:
            await engine.wait_ready()
        except Exception as e:
            await ws.send_json({"type": "error", "message": f"Model yüklenemedi: {e}"})
            await ws.close()
            return

        detector = await asyncio.to_thread(detector_factory)
        recorder = SessionRecorder(settings.recordings_dir) if save else None
        session = LiveSession(Segmenter(detector, seg_cfg), engine.transcriber.transcribe, recorder)

        finals: asyncio.Queue[FinalJob | None] = asyncio.Queue()
        send_lock = asyncio.Lock()
        partial_busy = False
        t_start: float | None = None  # ilk ses paketinin geldiği an

        async def send(msg: dict | None):
            if msg is None:
                return
            async with send_lock:
                try:
                    await ws.send_json(msg)
                except Exception:
                    pass

        async def final_worker():
            while (job := await finals.get()) is not None:
                try:
                    await send(await engine.run(session.run_final, job))
                    # Gecikme: konuşmanın bittiği an ile metnin hazır olduğu an arası
                    lag = (time.monotonic() - (t_start or time.monotonic())) - job.end
                    await send({"type": "stats", "lag_sec": round(max(0.0, lag), 2)})
                except Exception as e:
                    logger.exception("Kesin metin üretilemedi")
                    await send({"type": "error", "message": str(e)})

        async def partial_task(job):
            nonlocal partial_busy
            try:
                if finals.empty():  # bekleyen kesin metin varsa ara metin gereksiz
                    await send(await engine.run(session.run_partial, job))
            except Exception as e:
                logger.warning("Ara metin hatası: %s", e)
            finally:
                partial_busy = False

        worker = asyncio.create_task(final_worker())
        await send({
            "type": "ready",
            "device": getattr(engine.transcriber, "device", None),
            "model": settings.model_size,
            "recording": recorder.name if recorder else None,
        })

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
                        for job in session.segmenter.flush():
                            finals.put_nowait(job)
                        finals.put_nowait(None)
                        await worker
                        await send({"type": "done"})
                        await ws.close()
                        break
        except WebSocketDisconnect:
            pass
        finally:
            if not worker.done():
                # Bağlantı koptu: kalan cümleleri yine de kayda geçir
                for job in session.segmenter.flush():
                    finals.put_nowait(job)
                finals.put_nowait(None)
                try:
                    await asyncio.wait_for(worker, timeout=30)
                except Exception:
                    worker.cancel()
            session.close()

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    s = get_settings()
    print(f"\n  Canlı altyazı: http://{'localhost' if s.host in ('127.0.0.1', '0.0.0.0') else s.host}:{s.port}\n")
    uvicorn.run(create_app(s), host=s.host, port=s.port, log_level="info")


if __name__ == "__main__":
    main()
