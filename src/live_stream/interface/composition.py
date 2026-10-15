"""Bileşenleri birbirine bağlayan tek yer (CLI, API ve worker'lar buradan kurar)."""
from __future__ import annotations

from pathlib import Path

from live_stream.application.config import (
    DiarizerConfig, FusionConfig, LidConfig, PipelineConfig, ProfileConfig, VadConfig, WhisperParams,
)
from live_stream.application.language import LanguageResolver
from live_stream.application.ports import ResultRepository
from live_stream.application.transcribe import TranscribeAndDiarize
from live_stream.infrastructure.audio import SoundfileLoader
from live_stream.infrastructure.chunker import EnergyChunker
from live_stream.application.fusion import FusionEngine
from live_stream.infrastructure.catalog import language_profile, lid_stages
from live_stream.infrastructure.engines import LazyAsrEngines
from live_stream.infrastructure.normalizer import normalizer_for
from live_stream.infrastructure.sherpa_asr import build_asr
from live_stream.infrastructure.cluster_diarizer import EmbeddingClusterDiarizer
from live_stream.infrastructure.sherpa_diarizer import SherpaOnnxDiarizer
from live_stream.infrastructure.sherpa_vad import SileroVad
from live_stream.infrastructure.whisper_lid import LazyDetector, WhisperLanguageDetector


def default_config(models_dir: str | Path = "models", languages=("tr", "vi"),
                   models: dict[str, str] | None = None, threads: int = 2,
                   speakers: int = 0, hotwords: str | None = None,
                   diarizer: str = "cluster", workers: int = 1, profile: str = "full",
                   min_diarize_sec: float = 10.0, sandwich_max: float = 0.0,
                   device: str = "cpu", compute_type: str = "auto",
                   vi_hotwords: str | None = None) -> PipelineConfig:
    """workers: aynı süreçte aynı anda işlenecek ses sayısı (yerel mod); motorlar buna göre
    paralel çağrıya hazırlanır. device/compute_type: faster-whisper için (auto: GPU varsa GPU).
    hotwords: Türkçe (Whisper'a boşlukla ayrılmış terimler); vi_hotwords: Vietnamca terim dosyası
    (Zipformer, BÜYÜK HARF, satır başına bir terim)."""
    d, models = Path(models_dir), models or {}
    profiles = {}
    for code in languages:
        extra = {"num_threads": threads,
                 "whisper": WhisperParams(num_workers=max(1, workers), hotwords=hotwords,
                                          device=device, compute_type=compute_type)}
        if code == "vi" and vi_hotwords:
            extra["hotwords_file"] = vi_hotwords
        profiles[code] = language_profile(code, d, models.get(code), **extra)
    return PipelineConfig(
        languages=profiles,
        profile=ProfileConfig(default=profile, min_diarize_sec=min_diarize_sec),
        fusion=FusionConfig(sandwich_max=sandwich_max),
        # dil tespiti yalnızca birden çok dil tanımlıysa yüklenir
        lid=LidConfig(stages=lid_stages(d), num_threads=threads),
        vad=VadConfig(model=str(d / "silero_vad.onnx")),
        diarizer=DiarizerConfig(
            kind=diarizer,
            segmentation_model=str(d / "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
            embedding_model=str(d / ("3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
                                     if diarizer == "cluster" else "wespeaker_en_voxceleb_resnet34_LM.onnx")),
            num_speakers=speakers, num_threads=threads))


def build_diarizer(cfg: DiarizerConfig):
    if cfg.kind == "cluster":
        return EmbeddingClusterDiarizer(cfg)
    if cfg.kind == "pyannote":
        return SherpaOnnxDiarizer(cfg)
    raise ValueError(f"bilinmeyen diarizer: {cfg.kind}")


def build_resolver(cfg: PipelineConfig, with_detection: bool = True) -> LanguageResolver:
    """with_detection=False: ASR worker'ları dili kayıttan alır, Whisper LID yüklemez.
    Tek dil varken tespit yapılmaz (karar: varsayılan dil)."""
    detectors = []
    if with_detection and cfg.lid is not None and len(cfg.languages) > 1:
        for i, (enc, dec) in enumerate(cfg.lid.stages):
            make = (lambda e=enc, d=dec: WhisperLanguageDetector(e, d, cfg.lid.num_threads))
            detectors.append(make() if i == 0 else LazyDetector(make))  # small yalnızca gerekince
    return LanguageResolver(list(cfg.languages), detectors, cfg.lid)


def build_engines(cfg: PipelineConfig) -> LazyAsrEngines:
    def factory(profile):
        return lambda: build_asr(profile, normalizer_for(profile.code))

    return LazyAsrEngines({code: factory(p) for code, p in cfg.languages.items()})


def build_pipeline(cfg: PipelineConfig, repo: ResultRepository,
                   with_detection: bool = True) -> TranscribeAndDiarize:
    return TranscribeAndDiarize(
        loader=SoundfileLoader(), vad=SileroVad(cfg.vad),
        language=build_resolver(cfg, with_detection), engines=build_engines(cfg),
        diarizer=build_diarizer(cfg.diarizer), fusion=FusionEngine(cfg.fusion), repo=repo,
        chunker=EnergyChunker(cfg.chunk.max_len, cfg.chunk.min_len), profiles=cfg.profile,
        safety=cfg.safety)


# geriye dönük ad (CLI ve testler)
build = build_pipeline


def model_files(cfg: PipelineConfig) -> list[Path]:
    """Bu yapılandırmanın gerçekten kullandığı model dosyaları."""
    files = [Path(cfg.vad.model), Path(cfg.diarizer.embedding_model)]
    if cfg.diarizer.kind == "pyannote":
        files.append(Path(cfg.diarizer.segmentation_model))
    for prof in cfg.languages.values():
        files += [Path(prof.asr_model.path(role)) for role in prof.asr_model.files
                  if role != "bpe_vocab" or prof.asr.hotwords_file]
    if cfg.lid is not None and len(cfg.languages) > 1:
        files += [Path(p) for stage in cfg.lid.stages for p in stage]
    return files


def verify_models(models_dir: str | Path, files: list[Path]) -> list[str]:
    """MANIFEST.json'daki sha256 ile karşılaştırır. Sorun listesi döndürür (boş = tamam).
    İnternetsiz sunucuya taşınan modellerin bozulmadığını/değiştirilmediğini garanti eder."""
    import hashlib
    import json

    md = Path(models_dir)
    mpath = md / "MANIFEST.json"
    if not mpath.exists():
        return [f"{mpath} yok: scripts/download_models.py ile üretin"]
    expected = {}
    for pack in json.loads(mpath.read_text(encoding="utf-8")).values():
        expected.update(pack.get("sha256", {}))
    problems = []
    for f in files:
        if not f.exists():
            problems.append(f"dosya yok: {f}")
            continue
        try:
            rel = f.resolve().relative_to(md.resolve()).as_posix()
        except ValueError:
            continue  # models klasörü dışında (ör. ince ayarlı model): manifest kapsamaz
        if rel not in expected:
            if f.suffix in (".onnx", ".bin"):  # ağırlık dosyası doğrulanmadan çalışmaz
                problems.append(f"MANIFEST'te yok: {rel}")
            continue
        h = hashlib.sha256()
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != expected[rel]:
            problems.append(f"sha256 uyuşmuyor (dosya bozuk ya da değişmiş): {rel}")
    return problems


def store_from_settings(s, url: str | None = None, pool_size: int = 5):
    """İş deposu (yerel ya da tam mod); varsayılan bildirim olayları ayardan."""
    from live_stream.infrastructure.db import SqlAlchemyJobStore

    return SqlAlchemyJobStore(url or s.database_url, pool_size=pool_size,
                              notify_default=tuple(s.notify_default))


def config_from_settings(s, workers: int = 1) -> PipelineConfig:
    """Ayarlardan hat yapılandırması (yerel sunucu, kuyruk işçisi ve doğrulama aynı yerden)."""
    return default_config(s.models_dir, tuple(s.languages), s.models, s.threads, s.speakers,
                          s.hotwords, s.diarizer, workers=workers, profile=s.pipeline_profile,
                          vi_hotwords=s.vi_hotwords,
                          min_diarize_sec=s.pipeline_min_diarize_sec, sandwich_max=s.fusion_sandwich_max,
                          device=s.whisper_device, compute_type=s.whisper_compute_type)


def verify_models_or_exit(settings) -> None:
    import logging

    if not settings.verify_models:
        return
    cfg = config_from_settings(settings)
    problems = verify_models(settings.models_dir, model_files(cfg))
    if problems:
        raise SystemExit("Model doğrulaması başarısız (VERIFY_MODELS=0 ile atlanabilir):\n  "
                         + "\n  ".join(problems))
    logging.getLogger(__name__).info("modeller doğrulandı (sha256)")


def safe_file_remover(data_dir: str | Path):
    """Yalnızca yükleme klasörü içindeki dosyaları siler (ALLOW_LOCAL_PATHS ile verilen
    kullanıcı dosyalarına asla dokunmaz)."""
    root = Path(data_dir).resolve()

    def remove(path: str) -> bool:
        try:
            p = Path(path).resolve()
            if root not in p.parents:
                return False
            p.unlink(missing_ok=True)
            return True
        except OSError:
            return False

    return remove


def build_maintenance(store, settings, queue=None, queue_idle=None):
    """queue/queue_idle verilirse (tam mod) kuyruğa girememiş işler de yeniden kuyruğa konur."""
    from live_stream.application.maintenance import Maintenance

    return Maintenance(store, safe_file_remover(settings.data_dir), settings.audio_retention_hours,
                       settings.result_retention_days, settings.processing_stale_sec,
                       queue=queue, queue_idle=queue_idle, queued_stale_sec=settings.queued_stale_sec,
                       default_language=settings.languages[0] if len(settings.languages) == 1 else None)


def build_live(settings, cfg: PipelineConfig, engines: LazyAsrEngines, resolver: LanguageResolver):
    """Canlı altyazı: dil başına motor, konuşma algılayıcı üreticisi ve dil tespiti.

    Motorlar toplu işin model nesneleridir (bellekte bir kez). Yalnızca Türkçe için LIVE_MODEL
    farklı bir model seçebilir (4 GB GPU'da medium yetişmezse small); Vietnamca Zipformer
    zaten gerçek zamanın ~40 katı hızlı."""
    from live_stream.domain import VadSegment
    from live_stream.infrastructure.live_vad import make_detector
    from live_stream.infrastructure.normalizer import live_normalizer
    from live_stream.interface.live_server import LiveEngine

    live = {}
    for lang in settings.languages:
        batch_model = settings.model_for(lang)
        model = (settings.live_model or batch_model) if lang == "tr" else batch_model
        if model == batch_model:
            load = (lambda code=lang: engines[code])
        else:
            prof = cfg.languages[lang]
            live_prof = language_profile(lang, settings.models_dir, model,
                                         num_threads=prof.asr.num_threads, whisper=prof.asr.whisper)
            load = (lambda p=live_prof, code=lang: build_asr(p, normalizer_for(code)))
        live[lang] = LiveEngine(load, model, normalizer=live_normalizer(lang))

    def detector():
        return make_detector(settings.live_vad, cfg.vad.model, settings.live_vad_threshold)

    def detect(audio):
        """Otomatik dil: oturumun ilk cümlesinden (toplu işle aynı kademeli tespit)."""
        return resolver.resolve(None, [VadSegment(0.0, audio)])

    return live, detector, detect
