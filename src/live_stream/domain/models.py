"""Çekirdek veri tipleri. Hiçbir dış kütüphaneye bağlı değildir.

Zaman birimi her yerde saniyedir (float), ses dosyasının başından itibaren.
Metin birimi Word'dür: boşlukla ayrılan her parça. Vietnamcada bu bir hecedir
(Vietnamca yazımda her hece ayrı yazılır), Türkçede bir kelimedir.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np

BELIRSIZ = "BELIRSIZ"  # konuşmacısı güvenle belirlenemeyen kelime/hece


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    REJECTED = "rejected"  # ör. desteklenmeyen dil: hata değil, bilinçli ret


class LanguageSource(str, Enum):
    REQUEST = "istek"        # dili isteği gönderen yazdı
    DETECTED = "tespit"      # sistem sesten buldu
    DEFAULT = "varsayilan"   # konuşma yok ya da tek dil var; tespit yapılmadı


class UnsupportedLanguageError(Exception):
    """İstenen ya da tespit edilen dil sistemde yok."""

    def __init__(self, message: str, detected: str | None = None):
        super().__init__(message)
        self.detected = detected


@dataclass(frozen=True)
class AudioJob:
    """Bir ses dosyasını işleme isteği. language=None ise dil sesten tespit edilir."""

    job_id: str
    audio_path: str
    language: str | None = None
    callback_url: str | None = None
    num_speakers: int | None = None  # biliniyorsa; None ise sesten bulunur
    # entegre olan sistemin kendi kimliği; aynısı tekrar gelirse yeni iş açılmaz
    external_id: str | None = None
    # işlem profili: "full" (her adım) ya da "feedback" (kısa, çok sayıda mesaj için hızlı);
    # None ise sistemin varsayılanı (PIPELINE_PROFILE)
    profile: str | None = None
    # hangi olaylarda callback'e bildirim gidecek; None = varsayılan (NOTIFY_DEFAULT)
    notify: tuple[str, ...] | None = None


PROFILES = ("full", "feedback")
# bildirim olayları: running = işlenmeye başladı; diğerleri son durumlar
NOTIFY_EVENTS = ("running", "done", "failed", "rejected")
NOTIFY_DEFAULT = ("done", "failed", "rejected")


@dataclass(frozen=True)
class VadSegment:
    """VAD'ın bulduğu konuşma parçası. samples 16 kHz mono float32."""

    start: float
    samples: np.ndarray = field(repr=False)
    sample_rate: int = 16000

    @property
    def end(self) -> float:
        return self.start + len(self.samples) / self.sample_rate


@dataclass(frozen=True)
class Word:
    """ASR çıktısındaki tek birim (Vietnamcada hece, Türkçede kelime), mutlak zamanlarla."""

    text: str
    start: float
    end: float
    conf: float | None = None  # 0-1, modelin bu kelimeden eminliği (modeli vermiyorsa None)

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2


@dataclass(frozen=True)
class SpeakerSegment:
    """Diarization çıktısı: [start, end) aralığında konuşan."""

    speaker: str
    start: float
    end: float


@dataclass(frozen=True)
class Utterance:
    """Aynı konuşmacının art arda gelen kelimeleri (son çıktının bir satırı)."""

    speaker: str
    start: float
    end: float
    text: str
    edited: bool = False  # bir kişi arayüzden düzeltti
    index: int = -1       # kayıttaki sıra numarası (silinen satırlar atlansa da değişmez)
    conf: float | None = None          # kelime güvenlerinin ortalaması
    low_conf: tuple[int, ...] = ()     # metinde düşük güvenli kelimelerin sırası (0'dan)


@dataclass(frozen=True)
class Correction:
    """Bir kişinin düzelttiği satır: ince ayar verisi (ses aralığı + doğru metin)."""

    job_id: str
    audio_path: str
    language: str | None
    idx: int
    start: float
    end: float
    speaker: str
    text: str
    original_speaker: str
    original_text: str
    edited_at: datetime


@dataclass(frozen=True)
class LanguageDecision:
    code: str
    confidence: float
    source: LanguageSource
    uncertain: bool = False  # tespit güveni eşiğin altında: sonuç insan kontrolüne düşmeli


@dataclass(frozen=True)
class JobRecord:
    """Bir işin kalıcı kaydı (veritabanındaki hali)."""

    job: AudioJob
    status: JobStatus
    decision: LanguageDecision | None = None  # dil kararı verildiyse
    error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class Transcript:
    job_id: str
    duration: float
    utterances: list[Utterance]
    model: str
    language: LanguageDecision
    profile: str = "full"
    # aşama süreleri (sn): yukleme, vad, dil, asr, konusmaci, birlestirme, toplam
    timings: dict[str, float] = field(default_factory=dict)
    # kalite: konuşma süresi, yazıya dökülmemiş konuşma, kapsama yüzdesi, kurtarılan kelime,
    # düşük güvenli satır sayısı
    quality: dict[str, float] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return " ".join(u.text for u in self.utterances)
