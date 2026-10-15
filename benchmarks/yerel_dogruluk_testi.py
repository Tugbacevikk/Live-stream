"""Kendi kayıtlarınla ASR doğruluğu: TSV'deki dosyaları gerçek hattan (VAD + ASR) geçirir.

TSV sütunları: dosya, metin (doğru metin), isteğe bağlı rol. Yollar TSV'ye göre olabilir.

    python benchmarks/yerel_dogruluk_testi.py ornek.tsv                     # vi: iki Zipformer
    python benchmarks/yerel_dogruluk_testi.py tr.tsv --dil tr               # tr: Omnilingual + Whisper
    python benchmarks/yerel_dogruluk_testi.py ornek.tsv --modeller zipformer-68m

Hat üretimdekiyle aynıdır: Zipformer/Omnilingual'e sesin tamamı (en sessiz anlardan
bölünmüş), Whisper'a yalnızca VAD parçaları verilir.
Vietnamcada WER = hece hata oranı (her hece ayrı yazılır). Karşılaştırmadan önce iki
taraf da küçük harfe çevrilir, noktalama ve köşeli parantezler silinir (VietMed'de
[ ] içi araya giren konuşmacıdır; sözcükler sayılır, işaretler sayılmaz).
Sonuç ekrana ve sonuc_yerel_<dil>.json dosyasına yazılır.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import jiwer

from live_stream.application.config import VadConfig
from live_stream.infrastructure.audio import SoundfileLoader
from live_stream.infrastructure.chunker import EnergyChunker
from live_stream.infrastructure.catalog import MODEL_CHOICES, language_profile
from live_stream.infrastructure.normalizer import normalizer_for, turkish_lower
from live_stream.infrastructure.sherpa_asr import build_asr
from live_stream.infrastructure.sherpa_vad import SileroVad


def norm(text: str, lang: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = turkish_lower(text) if lang == "tr" else text.lower()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tsv")
    p.add_argument("--dil", default="vi", choices=["vi", "tr"])
    p.add_argument("--modeller", default=None, help="virgülle; varsayılan: o dilin tüm modelleri")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    tsv = Path(a.tsv)
    rows = list(csv.DictReader(tsv.open(encoding="utf-8"), delimiter="\t"))[: a.limit or None]
    loader = SoundfileLoader()
    vad = SileroVad(VadConfig(model=f"{a.models_dir}/silero_vad.onnx"))
    chunker = EnergyChunker()  # üretimdeki gibi: 'full' motorlara sesin tamamı
    items = []
    for r in rows:
        f = Path(r["dosya"].replace("\\", "/"))
        f = f if f.is_absolute() or f.exists() else tsv.parent / f
        audio = loader.load(str(f))
        items.append((r.get("rol", "-") or "-", norm(r["metin"], a.dil), len(audio) / 16000,
                      vad.split(audio), chunker.split(audio)))
    toplam = sum(x[2] for x in items)
    print(f"{len(items)} dosya, {toplam / 60:.1f} dk ses, "
          f"roller: {dict(sorted({k: sum(1 for i in items if i[0] == k) for k in {i[0] for i in items}}.items()))}")

    modeller = a.modeller.split(",") if a.modeller else MODEL_CHOICES[a.dil]
    sonuc = {}
    for m in modeller:
        asr = build_asr(language_profile(a.dil, a.models_dir, m, num_threads=a.threads),
                        normalizer_for(a.dil))
        t = time.perf_counter()
        hyps = [norm(" ".join(w.text for w in asr.transcribe(full if asr.segmentation == "full" else segs)),
                     a.dil) for *_, segs, full in items]
        dt = time.perf_counter() - t
        refs = [i[1] for i in items]
        by = defaultdict(lambda: ([], []))
        for (rol, ref, *_), h in zip(items, hyps):
            by[rol][0].append(ref)
            by[rol][1].append(h)
        r = {"WER": round(jiwer.wer(refs, hyps) * 100, 2), "CER": round(jiwer.cer(refs, hyps) * 100, 2),
             "RTF": round(dt / toplam, 4),
             "rol": {k: {"n": len(v[0]), "WER": round(jiwer.wer(*v) * 100, 2),
                         "CER": round(jiwer.cer(*v) * 100, 2)} for k, v in sorted(by.items())},
             "ornek": [{"ref": refs[i][:200], "hyp": hyps[i][:200]} for i in range(min(3, len(refs)))]}
        sonuc[m] = r
        roller = "  ".join(f"{k} %{v['WER']:.1f}" for k, v in r["rol"].items())
        print(f"  {m:18s} WER %{r['WER']:6.2f}  CER %{r['CER']:6.2f}  RTF {r['RTF']}   | {roller}")
    out = Path(f"sonuc_yerel_{a.dil}.json")
    out.write_text(json.dumps({"tsv": str(tsv), "dosya": len(items), "dakika": round(toplam / 60, 1),
                               "threads": a.threads, "sonuc": sonuc}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"Kaydedildi: {out}")


if __name__ == "__main__":
    main()
