"""
Konuşma algılayıcılar (Voice Activity Detection).

Her algılayıcı 512 örneklik (16 kHz'de 32 ms) bir çerçeve alır ve
"bu çerçevede konuşma var mı?" sorusuna True/False döner.

* SileroDetector : sinir ağı tabanlı, salon gürültüsüne ve alkışa dayanıklı (önerilen)
* EnergyDetector : ses seviyesine bakar, bağımlılık gerektirmez (yedek / testler)
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512  # Silero VAD 16 kHz için tam olarak 512 örnek bekler


class SpeechDetector(Protocol):
    def __call__(self, frame: np.ndarray) -> bool: ...

    def reset(self) -> None: ...


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


class SileroDetector:
    """Silero VAD; her bağlantı kendi örneğini kullanır (model durum tutar)."""

    def __init__(self, threshold: float = 0.5):
        import torch
        from silero_vad import load_silero_vad

        torch.set_num_threads(1)
        self._torch = torch
        self.model = load_silero_vad()
        self.threshold = threshold

    def __call__(self, frame: np.ndarray) -> bool:
        with self._torch.no_grad():
            prob = self.model(self._torch.from_numpy(frame), SAMPLE_RATE).item()
        return prob >= self.threshold

    def reset(self) -> None:
        self.model.reset_states()


def make_detector(engine: str, threshold: float = 0.5) -> SpeechDetector:
    if engine == "silero":
        try:
            return SileroDetector(threshold)
        except Exception as e:  # torch / silero-vad yoksa çalışmaya devam et
            logger.warning("Silero VAD yüklenemedi (%s); enerji tabanlı algılayıcıya geçiliyor.", e)
    return EnergyDetector()
