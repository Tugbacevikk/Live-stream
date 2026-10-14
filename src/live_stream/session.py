"""Tek bir canlı bağlantının durumu: ses -> parçalar -> metin."""

from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np

from live_stream.recorder import SessionRecorder
from live_stream.segmenter import FinalJob, PartialJob, Segmenter

TranscribeFn = Callable[[np.ndarray, bool, "str | None"], str]


class LiveSession:
    def __init__(self, segmenter: Segmenter, transcribe: TranscribeFn, recorder: SessionRecorder | None = None):
        self.segmenter = segmenter
        self.transcribe = transcribe
        self.recorder = recorder
        self._context: deque[str] = deque(maxlen=3)  # Whisper'a bağlam: son cümleler
        self._next_id = 0

    @staticmethod
    def pcm16_to_float(chunk: bytes) -> np.ndarray:
        if len(chunk) % 2:
            chunk = chunk[:-1]
        return np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0

    def feed_pcm16(self, chunk: bytes) -> list[PartialJob | FinalJob]:
        samples = self.pcm16_to_float(chunk)
        if self.recorder:
            self.recorder.write_audio(samples)
        return self.segmenter.feed(samples)

    def _prompt(self) -> str | None:
        return " ".join(self._context)[-220:] or None

    def run_partial(self, job: PartialJob) -> dict | None:
        text = self.transcribe(job.audio, True, self._prompt())
        return {"type": "partial", "start": job.start, "text": text} if text else None

    def run_final(self, job: FinalJob) -> dict:
        text = self.transcribe(job.audio, False, self._prompt())
        if not text:
            return {"type": "discard", "start": job.start}
        self._context.append(text)
        item = {"type": "final", "id": self._next_id, "start": job.start, "end": job.end, "text": text}
        self._next_id += 1
        if self.recorder:
            self.recorder.write_final(item)
        return item

    def close(self) -> None:
        if self.recorder:
            self.recorder.close()
