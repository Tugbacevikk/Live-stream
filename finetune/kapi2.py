"""Kapı 2: ince ayarlı model, temel Zipformer-68M'in yerini alabilir mi?

    python finetune/kapi2.py --yeni models/vi-ince-ayar-2026-11 \\
        --tibbi veri/vietmed_bolumlu/test/tek/referans.tsv \\
        --genel veri/genel_vi/referans.tsv

İki model aynı hattan (gerçek bölme + ASR) aynı dosyalarla geçer
(benchmarks/yerel_dogruluk_testi.py). Kural: ortak.gate2 — tıbbi testte en az %5 göreli
WER kazancı, genel konuşmada en fazla 0,5 puan kayıp, hızda en fazla 1,2 kat yavaşlama.
Tıbbi test eğitimde KULLANILMAMIŞ bölüm olmalı (veri_hazirla.py sızıntıyı engeller).
Genel konuşma seti: tıbbi olmayan Vietnamca (ör. VIVOS test ya da FLEURS vi); yoksa karar verilmez.
Sonuç kapi2_sonuc.json; geçerse çıkış 0, geçmezse 1. Geçerse: .env'de VI_MODEL=<klasör>.
Windows'ta da çalışır (model ONNX, eğitim ortamı gerekmez).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from ortak import gate2  # noqa: E402


def measure(tsv: str, models: list[str], threads: int, limit: int) -> dict:
    cmd = [sys.executable, str(ROOT / "benchmarks/yerel_dogruluk_testi.py"), tsv, "--dil", "vi",
           "--modeller", ",".join(models), "--threads", str(threads),
           "--models-dir", str(ROOT / "models")]
    if limit:
        cmd += ["--limit", str(limit)]
    subprocess.run(cmd, check=True)
    return json.loads(Path("sonuc_yerel_vi.json").read_text(encoding="utf-8"))["sonuc"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--yeni", required=True, help="ince ayarlı model klasörü (disari_aktar.sh çıktısı)")
    p.add_argument("--temel", default="zipformer-68m")
    p.add_argument("--tibbi", required=True, help="eğitimde kullanılmamış tıbbi test TSV'si")
    p.add_argument("--genel", default=None, help="tıbbi olmayan Vietnamca test TSV'si")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--limit", type=int, default=0, help="hızlı deneme için dosya sınırı (karar için 0)")
    a = p.parse_args()

    models = [a.temel, a.yeni]
    res = {"tibbi": measure(a.tibbi, models, a.threads, a.limit)}
    if a.genel:
        res["genel"] = measure(a.genel, models, a.threads, a.limit)
    base = {k: v[a.temel] for k, v in res.items()}
    new = {k: v[a.yeni] for k, v in res.items()}
    ok, reasons = gate2(base, new)
    Path("kapi2_sonuc.json").write_text(json.dumps(
        {"temel": a.temel, "yeni": a.yeni, "gecti": ok, "gerekce": reasons, "limit": a.limit,
         "olcum": res}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nKapı 2:", "GEÇTİ" if ok else "GEÇMEDİ")
    for r in reasons:
        print("  -", r)
    if a.limit:
        print("  (uyarı: --limit ile ölçüldü; karar için tüm test setiyle tekrarlayın)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
