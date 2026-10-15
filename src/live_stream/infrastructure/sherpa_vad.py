"""Silero VAD (sherpa-onnx üzerinden): sessizlikleri atıp konuşma parçalarını bulur."""
from __future__ import annotations

import numpy as np
import sherpa_onnx

from live_stream.application.config import VadConfig
from live_stream.domain import VadSegment


class SileroVad:
    def __init__(self, cfg: VadConfig):
        self.cfg = cfg
        mc = sherpa_onnx.VadModelConfig()
        mc.silero_vad.model = cfg.model
        mc.silero_vad.threshold = cfg.threshold
        mc.silero_vad.min_silence_duration = cfg.min_silence_duration
        mc.silero_vad.min_speech_duration = cfg.min_speech_duration
        mc.silero_vad.max_speech_duration = cfg.max_speech_duration
        mc.sample_rate = cfg.sample_rate
        mc.num_threads = 1
        self._model_cfg = mc

    def split(self, audio: np.ndarray) -> list[VadSegment]:
        sr = self.cfg.sample_rate
        # her çağrıda yeni dedektör: içerdeki durum (state) işler arasında taşınmasın
        vad = sherpa_onnx.VoiceActivityDetector(
            self._model_cfg, buffer_size_in_seconds=len(audio) / sr + 5)
        w = self._model_cfg.silero_vad.window_size
        out: list[VadSegment] = []

        def drain():
            while not vad.empty():
                seg = vad.front
                # front.start bir örnek (sample) indeksidir -> saniyeye çevir
                out.append(VadSegment(start=seg.start / sr,
                                      samples=np.asarray(seg.samples, dtype=np.float32),
                                      sample_rate=sr))
                vad.pop()

        for i in range(0, len(audio), w):
            vad.accept_waveform(audio[i:i + w])
            drain()
        vad.flush()
        drain()
        return self._pad(out, audio)

    def _pad(self, segs: list[VadSegment], audio: np.ndarray) -> list[VadSegment]:
        """Parçaları iki yandan genişletir; iki parça arasındaki boşluk ortadan paylaşılır."""
        p = int(self.cfg.pad * self.cfg.sample_rate)
        if p <= 0 or not segs:
            return segs
        sr = self.cfg.sample_rate
        bounds = [(round(s.start * sr), round(s.start * sr) + len(s.samples)) for s in segs]
        out = []
        for i, (a, b) in enumerate(bounds):
            lo = 0 if i == 0 else (bounds[i - 1][1] + a) // 2
            hi = len(audio) if i == len(bounds) - 1 else (b + bounds[i + 1][0]) // 2
            a2, b2 = max(lo, a - p), min(hi, b + p)
            out.append(VadSegment(start=a2 / sr, samples=audio[a2:b2], sample_rate=sr))
        return out
