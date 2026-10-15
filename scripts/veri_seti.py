"""Veri seti onay kuyruğu: arayüzde "Veri setine ekle" ile gelen kayıtları test ya da eğitime alır.

    python scripts/veri_seti.py listele
    python scripts/veri_seti.py onayla <is_id> [--kullanim test|egitim] [--onaylayan "Ad Soyad"]
    python scripts/veri_seti.py reddet <is_id>

Klasör DATASET_DIR (varsayılan data/veri_seti); --kok ile değiştirilebilir.
  test/<dil>/diyalog.tsv  -> benchmarks/diyalog_testi.py ile DER/cpWER ölçümü
  egitim/<dil>/liste.tsv  -> finetune/veri_hazirla.py --train girdisi
Aynı ses iki bölüme giremez (ozetler.tsv ile denetlenir): test setine sızıntı olmaz.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from live_stream.infrastructure.dataset import USES, approve, pending, reject  # noqa: E402
from live_stream.interface.settings import Settings  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kok", type=Path, default=None, help="veri seti klasörü (varsayılan DATASET_DIR)")
    sub = ap.add_subparsers(dest="komut", required=True)
    sub.add_parser("listele")
    o = sub.add_parser("onayla")
    o.add_argument("is_id")
    o.add_argument("--kullanim", choices=USES, default=None, help="boşsa aktarırken seçilen")
    o.add_argument("--onaylayan", default="")
    r = sub.add_parser("reddet")
    r.add_argument("is_id")
    a = ap.parse_args(argv)
    root = a.kok or Path(Settings().dataset_dir)

    if a.komut == "listele":
        items = pending(root)
        if not items:
            print("onay bekleyen kayıt yok")
        for m in items:
            print(f"{m['job_id']}\t{m.get('dil') or '?'}\t{m.get('kullanim')}\t{m['sure_sn']} sn\t"
                  f"{m['kisi_sayisi']} kişi\tdüzeltilen {m.get('duzeltilen_satir', 0)}\t"
                  f"{m.get('kontrol_eden', '')}\t{m.get('not', '')}")
        return 0
    if not (root / "bekleyen" / a.is_id).is_dir():
        print(f"bekleyen kayıt yok: {a.is_id}", file=sys.stderr)
        return 1
    if a.komut == "reddet":
        reject(root, a.is_id)
        print(f"silindi: {a.is_id}")
        return 0
    try:
        target = approve(root, a.is_id, a.kullanim, a.onaylayan)
    except ValueError as e:
        print(f"onaylanamadı: {e}", file=sys.stderr)
        return 2
    print(f"eklendi: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
