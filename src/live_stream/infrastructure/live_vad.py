"""Canlı altyazı için çerçeve çerçeve konuşma algılayıcılar.

* SileroFrameDetector : toplu işteki Silero modeli (models/silero_vad.onnx, sherpa-onnx).
  Live-stream'in ilk sürümü PyTorch + silero-vad paketini kullanıyordu; aynı model ONNX
  üzerinden çalıştığı için PyTorch bağımlılığı (~2 GB) kalktı.
* EnergyDetector      : ses seviyesine bakar, model gerektirmez (yedek ve testler).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from live_stream.application.live import FRAME_SAMPLES, SAMPLE_RATE

log = logging.getLogger(__name__)


class EnergyDetector:
    """Uyarlanabilir gürültü tabanlı basit enerji algılayıcı."""

    def __init__(self, floor: float = 0.006, multiplier: float = 3.0):
        self.floor = floor
        self.multiplier = multiplier
        self.noise = floor

    def __call__(self, frame: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(frame ** 2)) + 1e-12)
        speech = rms > max(self.floor, self.noise * self.multiplier)
        if not speech:
            self.noise = 0.95 * self.noise + 0.05 * rms
        return speech

    def reset(self) -> None:
        self.noise = self.floor


class SileroFrameDetector:
    """Her bağlantının kendi örneği olur (model durum tutar).

    sherpa-onnx olasılığı doğrudan vermez; dedektörün "şu an konuşma var mı" durumu kullanılır.
    Asıl cümle sonu kararı Segmenter'da (END_SILENCE_MS); buradaki süreler kısa tutulur ki
    iki karar birbirine eklenip gecikme yaratmasın."""

    def __init__(self, model: str, threshold: float = 0.5):
        import sherpa_onnx

        mc = sherpa_onnx.VadModelConfig()
        mc.silero_vad.model = model
        mc.silero_vad.threshold = threshold
        mc.silero_vad.min_silence_duration = 0.1
        mc.silero_vad.min_speech_duration = 0.1
        mc.silero_vad.max_speech_duration = 3600.0  # bölmeyi Segmenter yapar
        mc.silero_vad.window_size = FRAME_SAMPLES
        mc.sample_rate = SAMPLE_RATE
        mc.num_threads = 1
        self._cfg = mc
        self._make = lambda: sherpa_onnx.VoiceActivityDetector(mc, buffer_size_in_seconds=30)
        self.vad = self._make()

    def __call__(self, frame: np.ndarray) -> bool:
        self.vad.accept_waveform(np.ascontiguousarray(frame, dtype=np.float32))
        speech = self.vad.is_speech_detected()
        while not self.vad.empty():  # biten parçalar tutulmaz (bellek büyümesin)
            self.vad.pop()
        return speech

    def reset(self) -> None:
        self.vad = self._make()


def make_detector(kind: str, model: str, threshold: float = 0.5):
    if kind == "silero":
        if Path(model).exists():
            try:
                return SileroFrameDetector(model, threshold)
            except Exception as e:  # pragma: no cover - kurulum sorunu
                log.warning("Silero VAD açılamadı (%s); enerji tabanlı algılayıcı kullanılıyor", e)
        else:
            log.warning("%s yok (scripts/download_models.py); enerji tabanlı algılayıcı kullanılıyor",
                        model)
    return EnergyDetector()
