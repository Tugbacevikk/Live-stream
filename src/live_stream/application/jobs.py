"""İş akışı senaryoları (kuyruklu çalışma):

    SubmitJob       API'den gelir: kaydı açar, doğru kuyruğa koyar
    DetectLanguage  'lid' kuyruğu: dili bulur, işi o dilin ASR kuyruğuna aktarır
    RunTranscription 'asr.<dil>' kuyruğu: kayıttaki dil kararıyla metne çevirir

Neden dil tespiti ayrı kuyruk? Her ASR worker'ı yalnızca kendi dilinin modelini
yükler (faster-whisper medium ~1,5 GB, Zipformer 0,5 GB). Tespit hafif (Whisper-tiny) ve
ses başına ~0,25 sn; ayrı küçük bir worker yeter. Tek dil açıksa (LANGUAGES=tr) tespit
adımı atlanır, iş doğrudan o dilin kuyruğuna gider.
"""
from __future__ import annotations

import logging

from live_stream.domain import (
    NOTIFY_EVENTS, PROFILES, AudioJob, JobStatus, Transcript, UnsupportedLanguageError,
)

from .language import LanguageResolver
from .ports import AudioLoader, JobQueue, JobStore, Vad
from .transcribe import TranscribeAndDiarize

log = logging.getLogger(__name__)


class JobNotFound(Exception):
    pass


class SubmitJob:
    def __init__(self, store: JobStore, queue: JobQueue, supported: list[str]):
        self.store, self.queue, self.supported = store, queue, [s.lower() for s in supported]

    def __call__(self, job: AudioJob) -> str:
        """İşi kaydedip kuyruğa koyar; geçerli job_id'yi döndürür. Aynı external_id ile
        daha önce iş açıldıysa o işin kimliği döner ve kuyruğa tekrar girilmez."""
        lang = job.language.strip().lower() if job.language else None
        if lang in ("", "auto"):
            lang = None
        if lang is not None and lang not in self.supported:
            # API'de hemen hata: kuyruğa hiç girmez
            raise UnsupportedLanguageError(
                f"istenen dil '{lang}' desteklenmiyor; desteklenenler: {', '.join(self.supported)}")
        if job.num_speakers is not None and not 1 <= job.num_speakers <= 20:
            raise ValueError("num_speakers 1 ile 20 arasında olmalı")
        ext = (job.external_id or "").strip() or None
        if ext is not None and len(ext) > 255:
            raise ValueError("external_id en fazla 255 karakter olmalı")
        profile = (job.profile or "").strip().lower() or None
        if profile is not None and profile not in PROFILES:
            raise ValueError(f"profile '{profile}' yok; seçenekler: {', '.join(PROFILES)}")
        notify = None
        if job.notify is not None:
            notify = tuple(dict.fromkeys(e.strip().lower() for e in job.notify if e.strip()))
            bad = [e for e in notify if e not in NOTIFY_EVENTS]
            if bad:
                raise ValueError(f"bilinmeyen bildirim olayı {bad}; seçenekler: {', '.join(NOTIFY_EVENTS)}")
        job = AudioJob(job.job_id, job.audio_path, lang, job.callback_url, job.num_speakers, ext,
                       profile, notify)
        job_id = self.store.create(job)
        if job_id != job.job_id:
            return job_id  # tekrar gönderim: var olan iş
        if lang is not None:
            self.queue.enqueue_transcription(job.job_id, lang)
        elif len(self.supported) == 1:
            # tek dil: tespit adımı gereksiz (karar hat içinde "varsayılan" olarak yazılır)
            self.queue.enqueue_transcription(job.job_id, self.supported[0])
        else:
            self.queue.enqueue_detection(job.job_id)
        return job_id


class DetectLanguage:
    def __init__(self, store: JobStore, queue: JobQueue, loader: AudioLoader, vad: Vad,
                 resolver: LanguageResolver):
        self.store, self.queue, self.loader, self.vad, self.resolver = store, queue, loader, vad, resolver

    def __call__(self, job_id: str) -> str | None:
        rec = self.store.get(job_id)
        if rec is None:
            raise JobNotFound(job_id)
        try:
            segments = self.vad.split(self.loader.load(rec.job.audio_path))
            decision = self.resolver.resolve(None, segments)
        except UnsupportedLanguageError as e:
            self.store.set_status(job_id, JobStatus.REJECTED, error=str(e))
            return None
        except Exception as e:
            self.store.set_status(job_id, JobStatus.FAILED, error=f"{type(e).__name__}: {e}")
            raise
        self.store.set_language(job_id, decision)
        self.queue.enqueue_transcription(job_id, decision.code)
        log.info("job=%s dil=%s güven=%.2f%s", job_id, decision.code, decision.confidence,
                 " BELIRSIZ" if decision.uncertain else "")
        return decision.code


class RunTranscription:
    def __init__(self, store: JobStore, pipeline: TranscribeAndDiarize):
        self.store, self.pipeline = store, pipeline

    def __call__(self, job_id: str) -> Transcript | None:
        rec = self.store.get(job_id)
        if rec is None:
            raise JobNotFound(job_id)
        if rec.status in (JobStatus.DONE, JobStatus.REJECTED):
            return None  # aynı iş iki kez teslim edildiyse (kuyruk tekrarı) tekrar işleme
        try:
            return self.pipeline.run(rec.job, decision=rec.decision)
        except UnsupportedLanguageError:
            return None  # durum zaten REJECTED yazıldı
