"""Diyalog testi: tüm hat (dil, ASR, konuşmacı, birleştirme) RTTM'li diyaloglarda.

    python benchmarks/diyalog_testi.py veri/test/diyalog/diyalog.tsv
    python benchmarks/diyalog_testi.py diyalog.tsv --kisi-sayisi-ver     # num_speakers istekte
    python benchmarks/diyalog_testi.py diyalog.tsv --dil tr

Girdi: TSV (dosya, rttm, ref sütunları); veri seti onay kuyruğundan (scripts/veri_seti.py) ya da elle.
Ölçülenler (kullanıcının gördüğü son çıktı üzerinden, yani birleştirilmiş satırlardan):
  DER (collar 0,25 sn; kaçırılan / yanlış alarm / karışan), cpWER, hece WER,
  konuşmacı sayısı doğru mu, gecikme oranı (işlem süresi / ses süresi; hedef <= 1).
Sonuç: ekrana ve sonuc_diyalog.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import jiwer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from olcutler import cpwer, der, read_rttm  # noqa: E402

from live_stream.domain import AudioJob  # noqa: E402
from live_stream.infrastructure.memory_store import MemoryJobStore  # noqa: E402
from live_stream.interface.composition import build_pipeline, default_config  # noqa: E402


def norm(t: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", unicodedata.normalize("NFC", t).lower()).split())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tsv")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--dil", default=None, help="tr")
    p.add_argument("--kisi-sayisi-ver", action="store_true", help="doğru kişi sayısını istekte ver")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    tsv = Path(a.tsv)
    rows = list(csv.DictReader(tsv.open(encoding="utf-8"), delimiter="\t"))[: a.limit or None]
    pipe = build_pipeline(default_config(a.models_dir, threads=a.threads), MemoryJobStore())
    res = []
    for r in rows:
        ref_rttm = read_rttm(tsv.parent / r["rttm"])
        ref = json.loads((tsv.parent / r["ref"]).read_text(encoding="utf-8"))
        k = int(r["kisi_sayisi"])
        job = AudioJob(Path(r["dosya"]).stem, str(tsv.parent / r["dosya"]), a.dil, None,
                       k if a.kisi_sayisi_ver else None)
        t0 = time.perf_counter()
        t = pipe.run(job)
        dt = time.perf_counter() - t0
        hyp_segs = [(u.speaker, u.start, u.end) for u in t.utterances]
        d = der(ref_rttm, hyp_segs)
        per_hyp: dict[str, str] = {}
        for u in t.utterances:
            per_hyp[u.speaker] = (per_hyp.get(u.speaker, "") + " " + norm(u.text)).strip()
        per_ref = {s: norm(x) for s, x in ref["konusmaci_metin"].items()}
        ref_all = norm(" ".join(x["metin"] for x in ref["sira"]))
        res.append({"dosya": r["dosya"], "kisi": k,
                    "bulunan": len({u.speaker for u in t.utterances} - {"BELIRSIZ"}),
                    **{key: round(v, 4) for key, v in d.items()},
                    "cpwer": round(cpwer(per_ref, per_hyp), 4),
                    "wer": round(jiwer.wer(ref_all, norm(t.text) or "-"), 4),
                    "oran": round(dt / t.duration, 3), "dil": t.language.code})
        x = res[-1]
        print(f"  {x['dosya']}: kişi {k}->{x['bulunan']}  DER %{x['der']*100:.1f}  "
              f"cpWER %{x['cpwer']*100:.1f}  WER %{x['wer']*100:.1f}  oran {x['oran']}")

    def ort(key):
        return float(np.mean([x[key] for x in res])) if res else 0.0

    ozet = {"n": len(res), "kisi_sayisi_dogru": sum(x["kisi"] == x["bulunan"] for x in res),
            "der": ort("der"), "kacirilan": ort("kacirilan"), "yanlis_alarm": ort("yanlis_alarm"),
            "karisan": ort("karisan"), "cpwer": ort("cpwer"), "wer": ort("wer"),
            "oran_en_kotu": max((x["oran"] for x in res), default=0.0),
            "kisi_sayisi_verildi": a.kisi_sayisi_ver}
    print(f"\n{ozet['n']} diyalog: kişi sayısı doğru {ozet['kisi_sayisi_dogru']}/{ozet['n']}, "
          f"DER %{ozet['der']*100:.1f} (kaçırılan %{ozet['kacirilan']*100:.1f}, yanlış alarm "
          f"%{ozet['yanlis_alarm']*100:.1f}, karışan %{ozet['karisan']*100:.1f}), "
          f"cpWER %{ozet['cpwer']*100:.1f}, WER %{ozet['wer']*100:.1f}, "
          f"en kötü oran {ozet['oran_en_kotu']} (hedef <= 1)")
    Path("sonuc_diyalog.json").write_text(json.dumps({"ozet": ozet, "dosyalar": res},
                                                     ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
