"""
Cümle sonu tespiti (endpointing).

Sürekli gelen sesi konuşma parçalarına böler ve iki tür iş üretir:
  * PartialJob : konuşma sürerken ara metin için o ana kadarki ses
  * FinalJob   : duraklama (veya azami süre) ile biten tam cümle
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from live_stream.vad import FRAME_SAMPLES, SAMPLE_RATE, SpeechDetector

FRAME_SEC = FRAME_SAMPLES / SAMPLE_RATE  # 0.032 sn


@dataclass
class SegmenterConfig:
    end_silence_ms: int = 700          # bu kadar sessizlik -> cümle bitti
    max_utterance_sec: float = 12.0    # durmadan konuşulursa zorla böl
    partial_interval_sec: float = 1.2  # ara metin sıklığı (0 = kapalı)
    pre_roll_ms: int = 300             # konuşma başlamadan önceki sesi de al (ilk hece kesilmesin)
    start_frames: int = 3              # konuşma başladı demek için ardışık konuşma çerçevesi
    min_voiced_sec: float = 0.3        # toplam sesli süre bundan azsa (öksürük, tık) at
    tail_keep_ms: int = 200            # cümle sonundaki sessizlikten bırakılacak pay


@dataclass
class PartialJob:
    audio: np.ndarray
    start: float


@dataclass
class FinalJob:
    audio: np.ndarray
    start: float
    end: float


class Segmenter:
    def __init__(self, detector: SpeechDetector, config: SegmenterConfig | None = None):
        self.detector = detector
        self.cfg = config or SegmenterConfig()
        self._leftover = np.zeros(0, dtype=np.float32)
        self._pre_roll: deque[np.ndarray] = deque(maxlen=max(1, round(self.cfg.pre_roll_ms / 1000 / FRAME_SEC)))
        self._frames: list[np.ndarray] = []
        self._in_speech = False
        self._speech_run = 0
        self._silence_run = 0
        self._voiced = 0
        self._frames_seen = 0
        self._start_frame = 0
        self._last_partial_len = 0

    @property
    def stream_time(self) -> float:
        return self._frames_seen * FRAME_SEC

    def _audio(self) -> np.ndarray:
        return np.concatenate(self._frames) if self._frames else np.zeros(0, dtype=np.float32)

    def _close(self, trim: bool) -> FinalJob | None:
        audio = self._audio()
        if trim and self._silence_run:
            cut = self._silence_run * FRAME_SAMPLES - int(self.cfg.tail_keep_ms / 1000 * SAMPLE_RATE)
            if cut > 0:
                audio = audio[: len(audio) - cut]
        voiced_sec = self._voiced * FRAME_SEC
        start = self._start_frame * FRAME_SEC

        self._frames = []
        self._in_speech = False
        self._speech_run = self._silence_run = self._voiced = 0

        if voiced_sec < self.cfg.min_voiced_sec:
            return None
        return FinalJob(audio=audio, start=round(start, 2), end=round(start + len(audio) / SAMPLE_RATE, 2))

    def feed(self, samples: np.ndarray) -> list[PartialJob | FinalJob]:
        """16 kHz mono float32 örnekleri besler; ortaya çıkan işleri döner."""
        jobs: list[PartialJob | FinalJob] = []
        data = np.concatenate([self._leftover, samples.astype(np.float32, copy=False)])
        n = len(data) // FRAME_SAMPLES
        self._leftover = data[n * FRAME_SAMPLES:]

        for i in range(n):
            frame = data[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES]
            frame_idx = self._frames_seen
            self._frames_seen += 1
            speech = self.detector(frame)

            if not self._in_speech:
                self._pre_roll.append(frame)
                self._speech_run = self._speech_run + 1 if speech else 0
                if self._speech_run >= self.cfg.start_frames:
                    self._in_speech = True
                    self._frames = list(self._pre_roll)
                    self._pre_roll.clear()
                    self._start_frame = frame_idx + 1 - len(self._frames)
                    self._voiced = self._speech_run
                    self._silence_run = 0
                    self._last_partial_len = 0
                continue

            self._frames.append(frame)
            self._silence_run = 0 if speech else self._silence_run + 1
            self._voiced += int(speech)
            length_sec = len(self._frames) * FRAME_SEC

            if self._silence_run * FRAME_SEC * 1000 >= self.cfg.end_silence_ms:
                if job := self._close(trim=True):
                    jobs.append(job)
            elif length_sec >= self.cfg.max_utterance_sec:
                if job := self._close(trim=False):
                    jobs.append(job)
            elif self.cfg.partial_interval_sec > 0 and (
                len(self._frames) - self._last_partial_len
            ) * FRAME_SEC >= self.cfg.partial_interval_sec:
                self._last_partial_len = len(self._frames)
                jobs.append(PartialJob(audio=self._audio(), start=round(self._start_frame * FRAME_SEC, 2)))
        return jobs

    def flush(self) -> list[FinalJob]:
        """Akış kapanırken yarım kalan konuşmayı kesinleştirir."""
        if self._in_speech and self._frames:
            job = self._close(trim=True)
            return [job] if job else []
        return []
