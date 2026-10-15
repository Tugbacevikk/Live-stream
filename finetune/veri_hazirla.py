"""İnce ayar verisi: kısa ses + doğru metin -> lhotse kesitleri ve fbank özellikleri.

    # kendi verin (her bölüm için bir TSV: dosya, metin [, kayit])
    python finetune/veri_hazirla.py --train train.tsv --dev dev.tsv --test test.tsv --cikti ft_veri

    # VietMed'in kendi train/dev/test bölümleri (Hugging Face'ten; kesitler zaten kısa)
    python finetune/veri_hazirla.py --vietmed --cikti ft_veri

Çıktı: ft_veri/cuts_{train,dev,test}.jsonl.gz (+ feats/). egit.py bunu okur.
Kurallar:
  - metin modelin biçimine çevrilir (ortak.model_text: büyük harf, noktalamasız)
  - 1 sn'den kısa, 20 sn'den uzun kesit eğitime girmez (icefall sınırı); uzun kayıtları
    önce böl (VietMed kesitleri zaten kısa)
  - aynı ses ya da aynı kayıt iki bölüme düşerse durur (test sonucunu şişirir)
WSL2/Linux'ta, kurulum.sh ortamında çalışır (lhotse + torch gerekir).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ortak import audio_hash, leak_check, model_text  # noqa: E402

SR = 16000


def from_tsv(tsv: Path, out_audio: Path):
    """TSV satırları -> (kimlik, ses yolu, metin, kayıt)."""
    rows = list(csv.DictReader(tsv.open(encoding="utf-8"), delimiter="\t"))
    for i, r in enumerate(rows):
        p = Path(r["dosya"])
        p = p if p.is_absolute() else tsv.parent / p
        yield f"{tsv.stem}_{i:06d}", p, r["metin"], r.get("kayit", "")


def from_vietmed(split: str, out_audio: Path, seen: set):
    import numpy as np
    import soundfile as sf
    from datasets import load_dataset

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from vietmed_hazirla import kesitler, tekillestir

    out_audio.mkdir(parents=True, exist_ok=True)
    sayac: dict = {}
    for i, k in enumerate(tekillestir(kesitler(load_dataset("leduckhai/VietMed", split=split)), seen, sayac)):
        p = out_audio / f"vietmed_{split}_{i:06d}.flac"
        if not p.exists():
            sf.write(p, np.asarray(k.audio, dtype=np.float32), SR)
        yield f"vietmed_{split}_{i:06d}", p, k.text, k.recording
    print(f"  {split}: atılan kopya {sayac.get('kopya', 0)}")


def build_cuts(items, feats_dir: Path, num_jobs: int):
    from lhotse import CutSet, Fbank, FbankConfig, Recording, RecordingSet, SupervisionSegment, SupervisionSet

    recs, sups, skipped = [], [], 0
    for cid, path, text, _ in items:
        text = model_text(text)
        rec = Recording.from_file(str(path), recording_id=cid)
        if not text or not 1.0 <= rec.duration <= 20.0:
            skipped += 1
            continue
        recs.append(rec)
        sups.append(SupervisionSegment(id=cid, recording_id=cid, start=0.0, duration=rec.duration,
                                       channel=0, text=text, language="Vietnamese"))
    cuts = CutSet.from_manifests(recordings=RecordingSet.from_recordings(recs),
                                 supervisions=SupervisionSet.from_segments(sups)).resample(SR)
    # icefall/Zipformer ile aynı özellik: 80 boyutlu fbank
    cuts = cuts.compute_and_store_features(extractor=Fbank(FbankConfig(num_mel_bins=80)),
                                           storage_path=str(feats_dir), num_jobs=num_jobs)
    return cuts, skipped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train"), p.add_argument("--dev"), p.add_argument("--test")
    p.add_argument("--vietmed", action="store_true", help="VietMed train/dev/test bölümleri")
    p.add_argument("--cikti", required=True)
    p.add_argument("--is-sayisi", type=int, default=4)
    a = p.parse_args()
    out = Path(a.cikti)
    out.mkdir(parents=True, exist_ok=True)

    seen: set = set()
    sources = {}
    for split in ("train", "dev", "test"):
        if a.vietmed:
            sources[split] = list(from_vietmed(split, out / "audio", seen))
        elif getattr(a, split):
            sources[split] = list(from_tsv(Path(getattr(a, split)), out / "audio"))
    if "train" not in sources or "dev" not in sources:
        sys.exit("en az train ve dev gerekli")

    problems = leak_check({s: [(audio_hash(pth), rec) for _, pth, _, rec in items]
                           for s, items in sources.items()})
    if problems:
        sys.exit("bölümler arası sızıntı:\n  " + "\n  ".join(problems[:20]))

    for split, items in sources.items():
        cuts, skipped = build_cuts(items, out / "feats" / split, a.is_sayisi)
        cuts.to_file(out / f"cuts_{split}.jsonl.gz")
        hours = sum(c.duration for c in cuts) / 3600
        print(f"{split}: {len(cuts)} kesit, {hours:.1f} saat (atlanan {skipped})")
    print(f"Tamam: {out}")


if __name__ == "__main__":
    main()
