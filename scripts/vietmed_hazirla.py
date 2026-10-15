"""VietMed'i bölüm bazında hazırlar: train / dev / test asla karışmaz.

    python scripts/vietmed_hazirla.py --cikti C:/vietnamca_test/vietmed_bolumlu
    python scripts/vietmed_hazirla.py --cikti veri --bolumler dev,test --mod diyalog

Her bölüm için iki çıktı:
  <bolum>/tek/       aynı konuşmacının ardışık kesitleri birleştirilmiş 1-2 dk dosyalar
                     + referans.tsv (dosya, metin, rol, konusmaci, kayit, sure_sn, bolum)
                     -> ASR doğruluğu (benchmarks/yerel_dogruluk_testi.py)
  <bolum>/diyalog/   aynı görüşmenin ardışık kesitleri, orijinal konuşma sırasıyla,
                     en az iki konuşmacı; her biri için RTTM (kim ne zaman konuştu) ve
                     konuşmacı başına doğru metin + diyalog.tsv
                     -> DER ve cpWER (benchmarks/diyalog_testi.py)
Veri hijyeni: ses içeriği aynı olan kesitler (aynı kaydın kopyaları) tek kez kullanılır;
kaç kopya atıldığı yazılır. Seçim ve kesit sınırları her çalıştırmada aynıdır.

Not: diyaloglarda kesitler arası boşluk yapaydır (gap) ve üst üste konuşma yoktur;
gerçek, kesintisiz bir görüşme kaydı bunun yerini tutmaz.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 16000


@dataclass
class Kesit:
    audio: np.ndarray  # 16 kHz mono float32
    text: str
    role: str
    speaker: str
    recording: str


def kesitler(rows: Iterable[dict]) -> Iterator[Kesit]:
    """HF VietMed satırları -> 16 kHz kesitler (sıra korunur)."""
    import soxr

    for r in rows:
        a = np.asarray(r["audio"]["array"], dtype=np.float32)
        sr = r["audio"]["sampling_rate"]
        if a.ndim > 1:
            a = a.mean(axis=1)
        if sr != SR:
            a = soxr.resample(a, sr, SR).astype(np.float32)
        yield Kesit(a, r["text"], r.get("role", "-"), r["speaker_name"], r["audio_name"])


def tekillestir(items: Iterable[Kesit], seen: set[str], sayac: dict) -> Iterator[Kesit]:
    for k in items:
        h = hashlib.sha1(np.round(k.audio, 4).tobytes()).hexdigest()
        if h in seen:
            sayac["kopya"] = sayac.get("kopya", 0) + 1
            continue
        seen.add(h)
        yield k


def tek_konusmacili(items: list[Kesit], min_s: float = 60, max_s: float = 120,
                    gap: float = 0.3) -> Iterator[list[Kesit]]:
    """Aynı görüşmede, aynı konuşmacının ardışık kesitleri; min_s..max_s sn."""
    cur: list[Kesit] = []

    def sure(ks):
        return sum(len(k.audio) for k in ks) / SR + gap * max(0, len(ks) - 1)

    for k in items:
        if cur and (k.speaker != cur[-1].speaker or k.recording != cur[-1].recording
                    or sure(cur + [k]) > max_s):
            if sure(cur) >= min_s:
                yield cur
            cur = []
        cur.append(k)
    if cur and sure(cur) >= min_s:
        yield cur


def diyaloglar(items: list[Kesit], min_s: float = 60, max_s: float = 180,
               gap: float = 0.4) -> Iterator[list[Kesit]]:
    """Aynı görüşmenin ardışık kesitleri (gerçek konuşma sırası), en az 2 konuşmacı.
    Pencereler örtüşmez; max_s'den uzun tek kesit atlanır."""
    cur: list[Kesit] = []

    def sure(ks):
        return sum(len(k.audio) for k in ks) / SR + gap * max(0, len(ks) - 1)

    def uygun(ks):
        return sure(ks) >= min_s and len({k.speaker for k in ks}) >= 2

    for k in items:
        if len(k.audio) / SR > max_s:
            if uygun(cur):
                yield cur
            cur = []
            continue
        if cur and (k.recording != cur[-1].recording or sure(cur + [k]) > max_s):
            if uygun(cur):
                yield cur
            cur = []
        cur.append(k)
    if uygun(cur):
        yield cur


def birlestir(ks: list[Kesit], gap: float) -> tuple[np.ndarray, list[tuple[str, float, float, str]]]:
    """Ses ve (konuşmacı, başlangıç, bitiş, metin) listesi."""
    parts, segs, t = [], [], 0.0
    z = np.zeros(int(gap * SR), np.float32)
    for i, k in enumerate(ks):
        if i:
            parts.append(z)
            t += gap
        parts.append(k.audio)
        segs.append((k.speaker, t, t + len(k.audio) / SR, k.text))
        t += len(k.audio) / SR
    return np.concatenate(parts), segs


def yaz_tek(out: Path, bolum: str, gruplar: Iterable[list[Kesit]]) -> int:
    import soundfile as sf

    d = out / bolum / "tek"
    (d / "audio").mkdir(parents=True, exist_ok=True)
    n = 0
    with (d / "referans.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["dosya", "metin", "rol", "konusmaci", "kayit", "sure_sn", "bolum"])
        for i, ks in enumerate(gruplar):
            a, _ = birlestir(ks, 0.3)
            name = f"audio/{bolum}_tek_{i:04d}.flac"
            sf.write(d / name, a, SR)
            w.writerow([name, " ".join(k.text for k in ks), ks[0].role, ks[0].speaker,
                        ks[0].recording, round(len(a) / SR, 1), bolum])
            n += 1
    return n


def yaz_diyalog(out: Path, bolum: str, gruplar: Iterable[list[Kesit]], gap: float = 0.4) -> int:
    import soundfile as sf

    from olcutler import write_rttm  # benchmarks/olcutler.py

    d = out / bolum / "diyalog"
    for sub in ("audio", "rttm", "ref"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    n = 0
    with (d / "diyalog.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["dosya", "rttm", "ref", "kisi_sayisi", "sure_sn", "kayit", "bolum"])
        for i, ks in enumerate(gruplar):
            a, segs = birlestir(ks, gap)
            fid = f"{bolum}_diyalog_{i:04d}"
            sf.write(d / "audio" / f"{fid}.flac", a, SR)
            write_rttm(d / "rttm" / f"{fid}.rttm", fid, [(s, b, e) for s, b, e, _ in segs])
            per = {}
            for s, _, _, t in segs:
                per[s] = (per.get(s, "") + " " + t).strip()
            roles = {k.speaker: k.role for k in ks}
            (d / "ref" / f"{fid}.json").write_text(json.dumps(
                {"konusmaci_metin": per, "rol": roles,
                 "sira": [{"konusmaci": s, "baslangic": round(b, 3), "bitis": round(e, 3), "metin": t}
                          for s, b, e, t in segs]}, ensure_ascii=False, indent=1), encoding="utf-8")
            w.writerow([f"audio/{fid}.flac", f"rttm/{fid}.rttm", f"ref/{fid}.json", len(per),
                        round(len(a) / SR, 1), ks[0].recording, bolum])
            n += 1
    return n


def main():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from datasets import load_dataset

    p = argparse.ArgumentParser()
    p.add_argument("--cikti", required=True)
    p.add_argument("--bolumler", default="train,dev,test")
    p.add_argument("--mod", default="ikisi", choices=["tek", "diyalog", "ikisi"])
    p.add_argument("--en-kisa", type=float, default=60)
    p.add_argument("--en-uzun", type=float, default=120)
    p.add_argument("--diyalog-en-uzun", type=float, default=180)
    a = p.parse_args()
    out = Path(a.cikti)
    seen: set[str] = set()   # bölümler arası da: aynı kayıt iki bölüme düşmesin
    for bolum in a.bolumler.split(","):
        sayac: dict = {}
        ds = load_dataset("leduckhai/VietMed", split=bolum)
        items = list(tekillestir(kesitler(ds), seen, sayac))
        print(f"{bolum}: {len(items)} kesit, atılan kopya {sayac.get('kopya', 0)}")
        if a.mod in ("tek", "ikisi"):
            n = yaz_tek(out, bolum, tek_konusmacili(items, a.en_kisa, a.en_uzun))
            print(f"  tek konuşmacılı dosya: {n}")
        if a.mod in ("diyalog", "ikisi"):
            n = yaz_diyalog(out, bolum, diyaloglar(items, a.en_kisa, a.diyalog_en_uzun))
            print(f"  diyalog: {n}")
    print(f"Tamam: {out}")


if __name__ == "__main__":
    main()
