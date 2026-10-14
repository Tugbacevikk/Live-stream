"""
Faster-Whisper sarmalayıcı.

* GPU varsa CUDA'da çalışır; CUDA kütüphaneleri eksikse (Windows'ta sık olur)
  otomatik olarak CPU'ya düşer, uygulama çökmez.
* Ara metin (partial) hızlı ayarla (beam=1), kesin metin (final) daha doğru ayarla üretilir.
* Whisper'ın sessizlikte uydurduğu tipik Türkçe kalıplar ("Altyazı M.K.",
  "İzlediğiniz için teşekkürler" vb.) filtrelenir.
"""

from __future__ import annotations

import logging
import re
import threading
import time

import numpy as np

from live_stream.config import Settings

logger = logging.getLogger(__name__)

# Whisper'ın boş/gürültülü seste sık uydurduğu ifadeler (küçük harfle, noktalamasız)
HALLUCINATIONS = {
    "altyazı mk", "altyazı m k", "altyazı", "izlediğiniz için teşekkürler",
    "izlediğiniz için teşekkür ederim", "abone olmayı unutmayın", "abone olun",
    "videoyu beğenmeyi unutmayın", "bir sonraki videoda görüşmek üzere", "müzik",
    "thanks for watching", "subtitles by the amaraorg community",
}
# Not: "teşekkür ederim" bilerek listede yok; konuşmacı gerçekten söyleyebilir.


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def is_hallucination(text: str) -> bool:
    norm = _normalize(text)
    if not norm:
        return True
    if norm in HALLUCINATIONS:
        return True
    words = norm.split()
    # "ve ve ve ve ..." gibi döngüler
    return len(words) >= 6 and len(set(words)) <= 2


def _add_nvidia_dll_dirs() -> None:
    """
    Windows: `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` ile gelen DLL'leri
    görünür yapar; aksi halde "cublas64_12.dll / cudnn_ops64_9.dll bulunamadı" hatası alınır.
    """
    import os
    import sys

    if sys.platform != "win32":
        return
    import site
    from pathlib import Path

    roots = [Path(p) for p in site.getsitepackages() + [site.getusersitepackages()]]
    for root in roots:
        for bin_dir in root.glob("nvidia/*/bin"):
            try:
                os.add_dll_directory(str(bin_dir))
            except (OSError, AttributeError):
                pass
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


class WhisperTranscriber:
    def __init__(self, settings: Settings):
        self.s = settings
        self.model = None
        self.device = None
        self.compute_type = None
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- yükleme
    def _create(self, device: str, compute_type: str):
        if device == "cuda":
            _add_nvidia_dll_dirs()
        from faster_whisper import WhisperModel

        model = WhisperModel(
            self.s.model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=self.s.cpu_threads,
            download_root=self.s.model_dir,
        )
        # Isınma: CUDA DLL eksikliği gibi hatalar genelde ilk çıkarımda patlar
        list(model.transcribe(np.zeros(16000, dtype=np.float32), language=self.s.language, beam_size=1)[0])
        return model

    def load(self) -> None:
        if self.model is not None:
            return
        with self._lock:
            if self.model is not None:
                return
            wanted = self.s.device
            candidates: list[tuple[str, str]] = []
            if wanted in ("auto", "cuda"):
                ct = self.s.compute_type if self.s.compute_type != "auto" else "int8_float16"
                candidates.append(("cuda", ct))
            if wanted in ("auto", "cpu"):
                ct = self.s.compute_type if self.s.compute_type != "auto" else "int8"
                candidates.append(("cpu", ct))

            last_err: Exception | None = None
            for device, ct in candidates:
                try:
                    t0 = time.monotonic()
                    self.model = self._create(device, ct)
                    self.device, self.compute_type = device, ct
                    logger.info("Whisper '%s' yüklendi: %s/%s (%.1f sn)", self.s.model_size, device, ct, time.monotonic() - t0)
                    return
                except Exception as e:
                    last_err = e
                    logger.warning("Whisper %s/%s üzerinde açılamadı: %s", device, ct, e)
            raise RuntimeError(f"Whisper modeli yüklenemedi: {last_err}")

    @property
    def ready(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------- çıkarım
    def transcribe(self, audio: np.ndarray, partial: bool, prompt: str | None = None) -> str:
        self.load()
        segments, _ = self.model.transcribe(
            audio,
            language=self.s.language,
            beam_size=1 if partial else self.s.final_beam_size,
            temperature=0.0 if partial else (0.0, 0.2, 0.4),
            initial_prompt=prompt,
            hotwords=self.s.hotwords,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,               # konuşma algılamayı biz yapıyoruz
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
        )
        parts = [seg.text.strip() for seg in segments if not is_hallucination(seg.text)]
        return " ".join(p for p in parts if p).strip()
