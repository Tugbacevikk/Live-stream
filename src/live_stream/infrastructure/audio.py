"""Ses dosyasını 16 kHz mono float32 diziye çevirir.

Önce soundfile (wav, flac, ogg, opus, mp3); açamazsa ffmpeg (m4a, aac, amr, webm,
3gp, wma: telefon ve mesajlaşma uygulamalarının biçimleri). ffmpeg yoksa bu biçimler
anlaşılır bir hatayla reddedilir. Windows kurulumu: winget install Gyan.FFmpeg
"""
from __future__ import annotations

import shutil
import subprocess

import numpy as np
import soundfile as sf

TARGET_SR = 16000
FFMPEG_ONLY = {".m4a", ".aac", ".amr", ".webm", ".3gp", ".wma", ".mp4"}


class AudioDecodeError(Exception):
    pass


def ffmpeg_bin() -> str | None:
    return shutil.which("ffmpeg")


def _ffmpeg_decode(path: str, timeout: float = 300) -> np.ndarray:
    exe = ffmpeg_bin()
    if exe is None:
        raise AudioDecodeError("bu biçim için ffmpeg gerekli (Windows: winget install Gyan.FFmpeg)")
    cmd = [exe, "-nostdin", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1",
           "-ar", str(TARGET_SR), "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True, timeout=timeout).stdout
    except subprocess.CalledProcessError as e:
        raise AudioDecodeError(f"ffmpeg sesi çözemedi: {e.stderr.decode(errors='ignore')[:200]}")
    except subprocess.TimeoutExpired:
        raise AudioDecodeError("ffmpeg zaman aşımı")
    audio = np.frombuffer(out, dtype=np.float32)
    if audio.size == 0:
        raise AudioDecodeError("ses boş")
    return audio.copy()


class SoundfileLoader:
    def load(self, path: str) -> np.ndarray:
        try:
            audio, sr = sf.read(path, dtype="float32", always_2d=True)
        except Exception:
            return _ffmpeg_decode(path)
        audio = audio.mean(axis=1)  # stereo -> mono
        if sr != TARGET_SR:
            import soxr  # yalnızca gerekirse yüklenir

            audio = soxr.resample(audio, sr, TARGET_SR).astype(np.float32)
        return np.ascontiguousarray(audio)


def probe_duration(path: str) -> float:
    """Süre (sn). Okunamazsa AudioDecodeError (API bozuk dosyayı kuyruğa sokmamak için)."""
    try:
        return float(sf.info(path).duration)
    except Exception:
        return len(_ffmpeg_decode(path)) / TARGET_SR
