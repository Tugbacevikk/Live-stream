"""Konuşmacı ayrıştırma: pyannote segmentation-3.0 + WeSpeaker ResNet34 (sherpa-onnx)."""
from __future__ import annotations

import numpy as np
import sherpa_onnx

from live_stream.application.config import DiarizerConfig
from live_stream.domain import SpeakerSegment


class SherpaOnnxDiarizer:
    def __init__(self, cfg: DiarizerConfig):
        self.cfg = cfg
        clustering = (sherpa_onnx.FastClusteringConfig(num_clusters=cfg.num_speakers)
                      if cfg.num_speakers > 0
                      else sherpa_onnx.FastClusteringConfig(threshold=cfg.cluster_threshold))
        conf = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=cfg.segmentation_model),
                num_threads=cfg.num_threads),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=cfg.embedding_model, num_threads=cfg.num_threads),
            clustering=clustering,
            min_duration_on=cfg.min_duration_on,
            min_duration_off=cfg.min_duration_off)
        if not conf.validate():
            raise ValueError("diarization ayarı geçersiz (model yolu?)")
        self.sd = sherpa_onnx.OfflineSpeakerDiarization(conf)

    def diarize(self, audio: np.ndarray, segments=None, num_speakers=None) -> list[SpeakerSegment]:
        if num_speakers and num_speakers != self.cfg.num_speakers:
            # sherpa-onnx kümeleme ayarı kurulumda sabit; istek başına sayı için yeniden kur
            from dataclasses import replace
            return SherpaOnnxDiarizer(replace(self.cfg, num_speakers=num_speakers)).diarize(audio)
        # konuşmacı sayısından az ses varsa kümeleme çöker; çok kısa sesleri atla
        if len(audio) < self.sd.sample_rate:
            return []
        raw = self.sd.process(audio).sort_by_start_time()
        # etiketleri ilk konuşma sırasına göre yeniden adlandır: ilk konuşan speaker_00
        names: dict[int, str] = {}
        out = []
        for r in raw:
            if r.speaker not in names:
                names[r.speaker] = f"speaker_{len(names):02d}"
            out.append(SpeakerSegment(speaker=names[r.speaker], start=r.start, end=r.end))
        return out
