"""Ana kullanım senaryosu: ses -> dil kararı -> konuşmacılı metin."""
from __future__ import annotations

import logging
import time

from live_stream.domain import (
    PROFILES, AudioJob, JobStatus, LanguageDecision, SpeakerSegment, Transcript,
    UnsupportedLanguageError, VadSegment,
)

from .config import ProfileConfig, SafetyNetConfig
from .coverage import add_rescued, coverage

from .fusion import FusionEngine
from .language import LanguageResolver
from .ports import AsrEngines, AudioLoader, Diarizer, Notifier, ResultRepository, Vad

log = logging.getLogger(__name__)


class TranscribeAndDiarize:
    def __init__(self, loader: AudioLoader, vad: Vad, language: LanguageResolver,
                 engines: AsrEngines, diarizer: Diarizer, fusion: FusionEngine,
                 repo: ResultRepository, notifier: Notifier | None = None,
                 chunker: Vad | None = None, profiles: ProfileConfig | None = None,
                 safety: SafetyNetConfig | None = None):
        """chunker: 'full' bölmeli motorlar için sesin tamamını parçalayan bölücü."""
        self.loader, self.vad, self.language, self.engines = loader, vad, language, engines
        self.chunker = chunker
        self.profiles = profiles or ProfileConfig()
        self.safety = safety or SafetyNetConfig()
        self.diarizer, self.fusion, self.repo, self.notifier = diarizer, fusion, repo, notifier

    def run(self, job: AudioJob, decision: LanguageDecision | None = None) -> Transcript:
        """decision verilirse (ör. dil tespiti ayrı kuyrukta yapıldıysa) tekrar tespit edilmez."""
        self.repo.set_status(job.job_id, JobStatus.RUNNING)
        try:
            transcript = self._process(job, decision)
        except UnsupportedLanguageError as e:
            self.repo.set_status(job.job_id, JobStatus.REJECTED, error=str(e))
            raise
        except Exception as e:
            self.repo.set_status(job.job_id, JobStatus.FAILED, error=f"{type(e).__name__}: {e}")
            raise
        self.repo.save(transcript)  # sonucu yazar VE işi DONE yapar (tek işlemde)
        if self.notifier is not None:
            self.notifier.notify(job, transcript)
        return transcript

    def _process(self, job: AudioJob, decision: LanguageDecision | None) -> Transcript:
        t0 = time.perf_counter()
        timings: dict[str, float] = {}
        mark = [t0]

        def lap(name: str) -> None:
            now = time.perf_counter()
            timings[name] = round(now - mark[0], 4)
            mark[0] = now

        profile = job.profile if job.profile in PROFILES else self.profiles.default
        audio = self.loader.load(job.audio_path)
        duration = len(audio) / 16000
        lap("yukleme")

        segments = self.vad.split(audio)
        lap("vad")
        lang = decision or self.language.resolve(job.language, segments)
        asr = self.engines[lang.code]
        lap("dil")
        # VAD müzik altındaki/gürültülü konuşmayı atıyor; uydurma yapmayan motorlara sesin tamamı
        full = self.chunker is not None and getattr(asr, "segmentation", "vad") == "full"
        # ...ama VAD hiç konuşma bulmadıysa (sessizlik, test sinyali, bip) ASR hiç çalışmaz:
        # 13 Ekim, 1 sn'lik test sesleri: Zipformer "ừ", Whisper "Altyazı M.K." uyduruyordu.
        asr_segments = (self.chunker.split(audio) if full else segments) if segments else []
        fast = profile == "feedback"
        if not asr_segments:
            words = []
        elif fast and getattr(asr, "supports_fast", False):
            words = asr.transcribe(asr_segments, fast=True)  # Whisper: beam 1
        else:
            words = asr.transcribe(asr_segments)
        lap("asr")
        quality: dict[str, float] = {}
        if segments and self.safety.enabled:
            gaps, quality = coverage(segments, words, self.safety.min_gap)
            if gaps and not fast and getattr(asr, "supports_rescue", False):
                m, sr = self.safety.margin, 16000
                clips = []
                for a, b in gaps:
                    a, b = max(0.0, a - m), min(duration, b + m)
                    clips.append(VadSegment(start=a, samples=audio[int(a * sr):int(b * sr)]))
                words, n = add_rescued(words, asr.rescue(clips))
                gaps, quality = coverage(segments, words, self.safety.min_gap)
                quality["kurtarilan_kelime"] = n
                lap("guvenlik_agi")
        if not segments:
            speakers = []
        elif job.num_speakers == 1 or (fast and duration < self.profiles.min_diarize_sec):
            # tek kişi biliniyor ya da kısa geri bildirim mesajı: konuşmacı ayırma atlanır
            speakers = [SpeakerSegment("speaker_00", 0.0, duration)]
        else:
            speakers = self.diarizer.diarize(audio, segments, job.num_speakers)
        lap("konusmaci")
        utterances = self.fusion.fuse(words, speakers)
        lap("birlestirme")
        if any(u.conf is not None for u in utterances):
            quality["dusuk_guvenli_kelime"] = sum(len(u.low_conf) for u in utterances)

        dt = time.perf_counter() - t0
        timings["toplam"] = round(dt, 4)
        log.info("job=%s dil=%s(%s %.2f%s) profil=%s sure=%.1fs rtf=%.3f parca=%d kelime=%d satir=%d",
                 job.job_id, lang.code, lang.source.value, lang.confidence,
                 " BELIRSIZ" if lang.uncertain else "", profile, duration, dt / max(duration, 1e-9),
                 len(asr_segments), len(words), len(utterances))
        return Transcript(job_id=job.job_id, duration=duration, utterances=utterances,
                          model=asr.name, language=lang, profile=profile, timings=timings,
                          quality=quality)
