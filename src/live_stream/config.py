"""Uygulama ayarları. Tüm değerler ortam değişkeni veya .env dosyasıyla değiştirilebilir."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return _env(name, "true" if default else "false").lower() in ("1", "true", "yes", "evet")


@dataclass(frozen=True)
class Settings:
    # --- Whisper modeli ---
    model_size: str = field(default_factory=lambda: _env("WHISPER_MODEL", "small"))
    device: str = field(default_factory=lambda: _env("WHISPER_DEVICE", "auto"))          # auto | cuda | cpu
    compute_type: str = field(default_factory=lambda: _env("WHISPER_COMPUTE_TYPE", "auto"))
    cpu_threads: int = field(default_factory=lambda: _int("WHISPER_CPU_THREADS", 4))
    language: str | None = field(default_factory=lambda: _env("WHISPER_LANGUAGE", "tr") or None)
    hotwords: str | None = field(default_factory=lambda: _env("WHISPER_HOTWORDS", "") or None)
    final_beam_size: int = field(default_factory=lambda: _int("WHISPER_BEAM_SIZE", 5))
    model_dir: str | None = field(default_factory=lambda: _env("WHISPER_MODEL_DIR", "") or None)

    # --- Konuşma algılama (VAD) ve cümle sonu ---
    vad_engine: str = field(default_factory=lambda: _env("VAD_ENGINE", "silero"))         # silero | energy
    vad_threshold: float = field(default_factory=lambda: _float("VAD_THRESHOLD", 0.5))
    end_silence_ms: int = field(default_factory=lambda: _int("END_SILENCE_MS", 700))
    max_utterance_sec: float = field(default_factory=lambda: _float("MAX_UTTERANCE_SEC", 12.0))
    partial_interval_sec: float = field(default_factory=lambda: _float("PARTIAL_INTERVAL_SEC", 1.2))

    # --- Kayıt ---
    save_recordings: bool = field(default_factory=lambda: _bool("SAVE_RECORDINGS", True))
    recordings_dir: str = field(default_factory=lambda: _env("RECORDINGS_DIR", "recordings"))

    # --- Sunucu ---
    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _int("PORT", 8000))


@lru_cache
def get_settings() -> Settings:
    return Settings()
