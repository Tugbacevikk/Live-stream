"""Canlı konuşmacı etiketi için ses imzası: toplu işle aynı CAM++ modeli (sherpa-onnx).

Model bir kez yüklenir, oturumlar paylaşır (her çağrı kendi akışını açar). Bir cümlenin
imzası CPU'da gerçek zamanın ~30 katı hızlı (5 sn'lik cümle ~0,15 sn)."""
from __future__ import annotations

import threading

import numpy as np

SR = 16000


class SpeakerEmbedder:
    def __init__(self, model: str, num_threads: int = 1):
        self.model, self.num_threads = model, num_threads
        self._ex = None
        self._lock = threading.Lock()

    def _extractor(self):
        if self._ex is None:
            with self._lock:
                if self._ex is None:
                    import sherpa_onnx

                    self._ex = sherpa_onnx.SpeakerEmbeddingExtractor(
                        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=self.model,
                                                                    num_threads=self.num_threads))
        return self._ex

    def embed(self, audio: np.ndarray) -> np.ndarray:
        ex = self._extractor()
        st = ex.create_stream()
        st.accept_waveform(SR, np.ascontiguousarray(audio, dtype=np.float32))
        st.input_finished()
        e = np.asarray(ex.compute(st), dtype=np.float64)
        return e / (np.linalg.norm(e) + 1e-9)
