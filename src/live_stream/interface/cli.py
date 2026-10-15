"""Komut satırı: tek bir ses dosyasını konuşmacılı metne çevirir.

    python -m live_stream.interface.cli kayit.wav                 # dil sesten tespit edilir
    python -m live_stream.interface.cli kayit.wav --lang vi       # dil verilir, tespit atlanır
    python -m live_stream.interface.cli kayit.wav --speakers 2 --json sonuc.json
    python -m live_stream.interface.cli kayit.wav --tr-model faster-whisper-small --device cpu

API ve kuyruk işçileri de aynı build() ile kurulan TranscribeAndDiarize senaryosunu çağırır.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from live_stream.domain import AudioJob, UnsupportedLanguageError
from live_stream.infrastructure.catalog import MODEL_CHOICES
from live_stream.infrastructure.memory_store import MemoryJobStore
from live_stream.interface.composition import build, default_config  # noqa: F401  (testler kullanır)

MemoryRepo = MemoryJobStore  # geriye dönük ad


def main(argv=None):
    p = argparse.ArgumentParser(description="Ses -> konuşmacılı metin (Türkçe, Vietnamca)")
    p.add_argument("audio")
    p.add_argument("--lang", default="auto", help="tr, vi ya da auto (sesten tespit)")
    p.add_argument("--languages", default="tr,vi", help="sistemde açık diller")
    p.add_argument("--vi-model", default=None, help=f"{MODEL_CHOICES['vi']} ya da model klasörü")
    p.add_argument("--tr-model", default=None, help=f"{MODEL_CHOICES['tr']}")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="faster-whisper: auto = GPU varsa GPU")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--speakers", type=int, default=0, help="konuşmacı sayısı; 0: sesten bul")
    p.add_argument("--diarizer", choices=["cluster", "pyannote"], default="cluster")
    p.add_argument("--hotwords", default=None, help="Türkçe özel terimler, boşlukla (ör. 'YOLOv8 TÜBİTAK')")
    p.add_argument("--vi-hotwords", default=None, help="Vietnamca terim dosyası (BÜYÜK HARF)")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    models = {k: v for k, v in (("vi", a.vi_model), ("tr", a.tr_model)) if v}
    cfg = default_config(a.models_dir, tuple(a.languages.split(",")), models, a.threads,
                         a.speakers, a.hotwords, a.diarizer, device=a.device, vi_hotwords=a.vi_hotwords)
    repo = MemoryRepo()
    job = AudioJob(job_id=Path(a.audio).stem, audio_path=a.audio,
                   language=None if a.lang == "auto" else a.lang)
    try:
        t = build(cfg, repo).run(job)
    except UnsupportedLanguageError as e:
        print(f"REDDEDİLDİ: {e}", file=sys.stderr)
        return 2
    lang = t.language
    print(f"# dil: {lang.code} ({lang.source.value}, güven {lang.confidence:.2f}"
          f"{', BELİRSİZ' if lang.uncertain else ''}) model: {t.model}")
    for u in t.utterances:
        print(f"[{u.start:7.2f} - {u.end:7.2f}] {u.speaker}: {u.text}")
    if a.json:
        Path(a.json).write_text(json.dumps(asdict(t), ensure_ascii=False, indent=1, default=str),
                                encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
