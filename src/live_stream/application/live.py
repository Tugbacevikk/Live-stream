"""Canlı altyazı: sürekli gelen ses -> cümleler -> metin.

Toplu iş (transcribe.py) bütün sesi görür ve kesin sonuç verir. Canlı oturum sesi geldikçe
işler; her cümle iki kez çözülür:
  * PartialJob : konuşma sürerken o ana kadarki ses (hızlı: beam 1) -> ekranda gri ara metin
  * FinalJob   : duraklama (ya da azami süre) ile biten cümle (doğru: beam 5 + önceki cümleler)

Bu modül yalnızca numpy ve portları bilir; konuşma algılayıcı (Silero), ASR (faster-whisper)
ve kayıtçı (soundfile) dışarıdan verilir. Live-stream'in ilk sürümündeki segmenter.py ve
session.py'nin aynısıdır; yalnızca bağımlılıklar portlara çevrildi.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .ports import LiveRecorder, SpeechDetector

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512                      # Silero VAD 16 kHz'de tam 512 örnek (32 ms) bekler
FRAME_SEC = FRAME_SAMPLES / SAMPLE_RATE  # 0.032 sn

# (ses, hızlı mı, istem) -> metin
TranscribeText = Callable[[np.ndarray, bool, "str | None"], str]


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
    """Cümle sonu tespiti (endpointing): sürekli sesi konuşma parçalarına böler."""

    def __init__(self, detector: SpeechDetector, config: SegmenterConfig | None = None):
        self.detector = detector
        self.cfg = config or SegmenterConfig()
        self._leftover = np.zeros(0, dtype=np.float32)
        self._pre_roll: deque[np.ndarray] = deque(
            maxlen=max(1, round(self.cfg.pre_roll_ms / 1000 / FRAME_SEC)))
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
        return FinalJob(audio=audio, start=round(start, 2),
                        end=round(start + len(audio) / SAMPLE_RATE, 2))

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


class LiveSession:
    """Tek bir canlı bağlantının durumu: ses -> parçalar -> metin (+ kayıt)."""

    def __init__(self, segmenter: Segmenter, transcribe: TranscribeText,
                 recorder: LiveRecorder | None = None, context_sentences: int = 3):
        self.segmenter = segmenter
        self.transcribe = transcribe
        self.recorder = recorder
        self._context: deque[str] = deque(maxlen=context_sentences)  # Whisper'a bağlam
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

    @property
    def lines(self) -> int:
        return self._next_id

    def close(self) -> str | None:
        """Kaydı kapatır; kaydedilen ses dosyasının yolunu döndürür (kayıt yoksa None)."""
        if self.recorder:
            self.recorder.close()
            return self.recorder.audio_path
        return None


class LanguageLock:
    """Canlı oturumun dili. Seçildiyse baştan bellidir; 'auto' ise ilk yeterince uzun cümleden
    bulunur ve oturum boyunca kilitlenir (konferansta dil değişmez; her cümlede tespit hem
    gecikme ekler hem de kısa cümlelerde yanılır).

    detect(ses) -> LanguageDecision; desteklenmeyen dil ya da hata olursa fallback kullanılır."""

    def __init__(self, language: str | None, detect: Callable[[np.ndarray], object] | None,
                 fallback: str, min_sec: float = 1.5):
        self.code: str | None = language
        self.confidence: float | None = 1.0 if language else None
        self.detected = False
        self._detect, self.fallback, self.min_sec = detect, fallback, min_sec

    @property
    def locked(self) -> bool:
        return self.code is not None

    def decide(self, audio: np.ndarray, final: bool) -> str | None:
        """Dil belliyse onu döner. Değilse: ses yeterince uzunsa (ya da kesin cümleyse) tespit
        edip kilitler; kısa ara metinde None (henüz karar yok, ara metin gösterilmez)."""
        if self.code is not None:
            return self.code
        if not final and len(audio) < self.min_sec * SAMPLE_RATE:
            return None
        try:
            d = self._detect(audio) if self._detect else None
            self.code, self.confidence = (d.code, d.confidence) if d else (self.fallback, None)
        except Exception:  # desteklenmeyen dil (UnsupportedLanguageError) ya da model hatası
            self.code, self.confidence = self.fallback, None
        self.detected = True
        return self.code
