"""Dil tespiti testi: gerçek seslerinde doğru dil ne sıklıkla bulunuyor?

Girdi: sekmeyle ayrılmış dosya (TSV), ilk satır başlık. "dosya" sütunu zorunlu,
"dil" sütunu yoksa --dil ile verilen dil herkese uygulanır.
    dosya<TAB>dil
    sesler/a.wav<TAB>vi
    sesler/b.wav<TAB>tr
    sesler/c.wav<TAB>en        <- desteklenmeyen dil: reddedilmesi beklenir

Örnek (VietMed referans.tsv'de dil sütunu yok, hepsi Vietnamca):
    python benchmarks/dil_tespiti_testi.py C:/vietnamca_test/vietmed_1_2dk/referans.tsv --dil vi

Çıktı: doğru / yanlış dil / yanlış ret / belirsiz sayıları, small'a devredilen oran,
dosya başına süre. Ayrıntı sonuc_dil_tespiti.json dosyasına yazılır.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

from live_stream.application.config import LidConfig, VadConfig
from live_stream.application.language import LanguageResolver
from live_stream.domain import UnsupportedLanguageError
from live_stream.infrastructure.audio import SoundfileLoader
from live_stream.infrastructure.catalog import lid_stages
from live_stream.infrastructure.sherpa_vad import SileroVad
from live_stream.infrastructure.whisper_lid import WhisperLanguageDetector


class Counting:
    def __init__(self, det):
        self.det, self.n = det, 0

    def probabilities(self, audio):
        self.n += 1
        return self.det.probabilities(audio)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tsv")
    p.add_argument("--dil", default=None, help="TSV'de dil sütunu yoksa herkes için bu dil")
    p.add_argument("--diller", default="vi,tr", help="sistemde desteklenen diller")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    supported = a.diller.split(",")
    tsv = Path(a.tsv)
    rows = list(csv.DictReader(tsv.open(encoding="utf-8"), delimiter="\t"))
    if a.limit:
        rows = rows[: a.limit]
    stages = [Counting(WhisperLanguageDetector(e, d)) for e, d in lid_stages(a.models_dir)]
    resolver = LanguageResolver(supported, stages, LidConfig())
    vad = SileroVad(VadConfig(model=str(Path(a.models_dir) / "silero_vad.onnx")))
    loader = SoundfileLoader()

    sayac, detay, sure = Counter(), [], 0.0
    for r in rows:
        f = Path(r["dosya"])
        f = f if f.is_absolute() or f.exists() else tsv.parent / f
        beklenen = (r.get("dil") or a.dil or "").lower()
        segs = vad.split(loader.load(str(f)))
        t = time.perf_counter()
        try:
            d = resolver.resolve(None, segs)
            bulunan, guven, belirsiz = d.code, d.confidence, d.uncertain
        except UnsupportedLanguageError as e:
            bulunan, guven, belirsiz = "RET:" + (e.detected or "?"), 0.0, False
        sure += time.perf_counter() - t
        if beklenen not in supported:
            sonuc = "dogru_ret" if bulunan.startswith("RET") else "yanlis_kabul"
        elif bulunan == beklenen:
            sonuc = "dogru_belirsiz" if belirsiz else "dogru"
        else:
            sonuc = "yanlis_ret" if bulunan.startswith("RET") else "yanlis_dil"
        sayac[sonuc] += 1
        detay.append({"dosya": str(f), "beklenen": beklenen, "bulunan": bulunan,
                      "guven": guven, "sonuc": sonuc})

    n = len(rows)
    print(f"{n} dosya, desteklenen: {supported}")
    for k in ("dogru", "dogru_belirsiz", "yanlis_dil", "yanlis_ret", "dogru_ret", "yanlis_kabul"):
        if sayac[k]:
            print(f"  {k:15s} {sayac[k]:5d}  %{sayac[k] / n * 100:.1f}")
    print(f"  small'a devredilen: {stages[1].n} (%{stages[1].n / max(n, 1) * 100:.1f})")
    print(f"  dosya başına tespit süresi: {sure / max(n, 1):.2f} sn")
    print("  hatalar:", [x for x in detay if x["sonuc"] in ("yanlis_dil", "yanlis_ret", "yanlis_kabul")][:10])
    Path("sonuc_dil_tespiti.json").write_text(
        json.dumps({"ozet": dict(sayac), "detay": detay}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
