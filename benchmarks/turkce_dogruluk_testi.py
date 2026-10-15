"""Türkçe ASR testi: omnilingual-300m ile whisper-turbo, FLEURS tr_tr test bölümünde.

İki model de gerçek kullanımdaki gibi VAD parçaları üzerinde çalışır (Whisper'ın
kısa parçada yavaşlaması ancak böyle görünür). Karşılaştırma için hem noktalama
ve büyük harf silinerek (Türkçe kuralıyla küçük harf) hem de sayılar olduğu gibi
bırakılarak WER/CER hesaplanır; Whisper "3" yazıp Omnilingual "üç" yazınca
WER farkının bir kısmı bundan gelir, örneklere bakın.

Kurulum:
    pip install -e ".[bench]"
    python scripts/download_models.py --tr all
Çalıştırma:
    python benchmarks/turkce_dogruluk_testi.py --n 200
"""
from __future__ import annotations

import argparse
import json
import re
import time

import jiwer
import numpy as np
from datasets import load_dataset

from live_stream.application.config import VadConfig
from live_stream.infrastructure.catalog import MODEL_CHOICES, language_profile
from live_stream.infrastructure.normalizer import TrNormalizer, turkish_lower
from live_stream.infrastructure.sherpa_asr import build_asr
from live_stream.infrastructure.sherpa_vad import SileroVad


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", turkish_lower(s)).split())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--models-dir", default="models")
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args()

    ds = load_dataset("google/fleurs", "tr_tr", split="test", trust_remote_code=True)
    ds = ds.shuffle(seed=42).select(range(min(a.n, len(ds))))
    audios = [np.asarray(x["audio"]["array"], dtype=np.float32) for x in ds]
    refs = [norm(x["transcription"]) for x in ds]
    toplam_sn = sum(len(x) for x in audios) / 16000
    vad = SileroVad(VadConfig(model=f"{a.models_dir}/silero_vad.onnx"))
    segs = [vad.split(x) for x in audios]
    print(f"FLEURS tr_tr test: {len(refs)} kayıt, {toplam_sn / 60:.1f} dk, "
          f"{sum(map(len, segs))} VAD parçası")

    sonuc = {}
    for m in MODEL_CHOICES["tr"]:
        asr = build_asr(language_profile("tr", a.models_dir, m, num_threads=a.threads), TrNormalizer())
        t = time.perf_counter()
        hyps = [norm(" ".join(w.text for w in asr.transcribe(s))) for s in segs]
        dt = time.perf_counter() - t
        r = {"WER": round(jiwer.wer(refs, hyps) * 100, 2), "CER": round(jiwer.cer(refs, hyps) * 100, 2),
             "RTF": round(dt / toplam_sn, 4),
             "ornek": [{"ref": refs[i], "hyp": hyps[i]} for i in range(3)]}
        sonuc[m] = r
        print(f"  {m:18s} WER %{r['WER']:6.2f}  CER %{r['CER']:6.2f}  RTF {r['RTF']}")
    with open("sonuc_turkce.json", "w", encoding="utf-8") as f:
        json.dump({"threads": a.threads, "n": len(refs), "sonuc": sonuc}, f, ensure_ascii=False, indent=1)
    print("Kaydedildi: sonuc_turkce.json")


if __name__ == "__main__":
    main()
