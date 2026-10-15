"""Sesin TAMAMINI kapsayan parçalama: kesimler en sessiz anlara konur.

Neden? VAD (Silero) müzik altındaki ya da gürültülü konuşmayı "konuşma değil"
diye atıyor; atılan kısım ASR'a hiç ulaşmıyor (Vietnamca projesinde ölçüldü: VAD ile
eksik hece %13,1, sesin tamamıyla %1,9). Omnilingual sessizlik/müzikte metin
uydurmuyor, bu yüzden ona sesin tamamı verilir. Whisper uydurduğu için ona VAD parçaları verilmeye devam eder.

Yöntem: her parça en fazla max_len sn; kesim noktası [min_len, max_len] aralığındaki
en düşük enerjili 100 ms'lik pencerenin ortası. Tamamen sessiz parçalar atlanır.
"""
from __future__ import annotations

import numpy as np

from live_stream.domain import VadSegment

SR = 16000


def energy_cuts(audio: np.ndarray, sr: int = SR, max_len: float = 15.0, min_len: float = 5.0,
                frame: float = 0.1, silence_rms: float = 1e-3) -> list[tuple[int, int]]:
    """Sessiz olmayan parçaların [başlangıç, bitiş) örnek indeksleri. Enerji float64 ile hesaplanır
    (toplama sırası farkı iki dilde sonucu değiştirmesin)."""
    n, f = len(audio), int(frame * sr)
    if n == 0:
        return []
    frames = n // f
    x = np.asarray(audio, dtype=np.float64)
    energy = np.sqrt(np.mean(x[: frames * f].reshape(frames, f) ** 2, axis=1)) if frames else np.zeros(0)
    out, start = [], 0
    while start < n:
        if n - start <= max_len * sr:
            end = n
        else:
            lo = (start + int(min_len * sr)) // f
            hi = min(frames, (start + int(max_len * sr)) // f)
            k = lo + int(np.argmin(energy[lo:hi])) if hi > lo else hi
            end = k * f + f // 2
        if float(np.sqrt(np.mean(x[start:end] ** 2))) >= silence_rms:
            out.append((start, end))
        start = end
    return out


class EnergyChunker:
    def __init__(self, max_len: float = 15.0, min_len: float = 5.0, frame: float = 0.1,
                 silence_rms: float = 1e-3):
        self.max_len, self.min_len, self.frame, self.silence_rms = max_len, min_len, frame, silence_rms

    def cuts(self, audio: np.ndarray) -> list[tuple[int, int]]:
        return energy_cuts(audio, SR, self.max_len, self.min_len, self.frame, self.silence_rms)

    def split(self, audio: np.ndarray) -> list[VadSegment]:
        return [VadSegment(start=a / SR, samples=audio[a:b], sample_rate=SR) for a, b in self.cuts(audio)]
