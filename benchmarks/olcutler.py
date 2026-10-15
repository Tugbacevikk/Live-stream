"""Konuşmacı ayrıştırma ölçütleri: DER ve cpWER (yalnızca numpy/scipy/jiwer).

DER (Diarization Error Rate) = (kaçırılan konuşma + yanlış alarm + karışan konuşmacı)
                               / toplam referans konuşma süresi
  - 10 ms'lik karelerle hesaplanır; sistem etiketleri referansa en iyi eşleşmeyle
    (Macar algoritması) bağlanır, yani etiketin adı değil tutarlılığı ölçülür.
  - collar: referans sınırlarının ± collar sn çevresi sayılmaz (standart 0,25 sn).
  - Üst üste konuşma: referansta bir anda birden çok kişi varsa her biri ayrı sayılır.

cpWER (concatenated minimum-permutation WER): her konuşmacının söyledikleri arka
arkaya eklenir; sistem konuşmacıları referans konuşmacılarına en iyi eşleşmeyle
bağlanır ve toplam kelime hatası hesaplanır. Hem metin hem kişi hatasını tek sayıda toplar.

RTTM satırı: SPEAKER <dosya> 1 <başlangıç> <süre> <NA> <NA> <konuşmacı> <NA> <NA>
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

FRAME = 0.01


def read_rttm(path: str | Path) -> list[tuple[str, float, float]]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        p = line.split()
        if p and p[0] == "SPEAKER":
            start, dur = float(p[3]), float(p[4])
            out.append((p[7], start, start + dur))
    return out


def write_rttm(path: str | Path, file_id: str, segs: list[tuple[str, float, float]]) -> None:
    lines = [f"SPEAKER {file_id} 1 {s:.3f} {e - s:.3f} <NA> <NA> {spk} <NA> <NA>"
             for spk, s, e in segs]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _matrix(segs, n_frames, speakers):
    m = np.zeros((len(speakers), n_frames), dtype=bool)
    idx = {s: i for i, s in enumerate(speakers)}
    for spk, s, e in segs:
        a, b = int(round(s / FRAME)), int(round(e / FRAME))
        m[idx[spk], max(0, a):min(n_frames, b)] = True
    return m


def der(ref: list[tuple[str, float, float]], hyp: list[tuple[str, float, float]],
        collar: float = 0.25) -> dict[str, float]:
    """Sonuç oranlar (0-1): der, kacirilan, yanlis_alarm, karisan."""
    end = max([e for _, _, e in ref + hyp] + [0.0])
    n = int(np.ceil(end / FRAME)) + 1
    rs = sorted({s for s, _, _ in ref})
    hs = sorted({s for s, _, _ in hyp})
    R = _matrix(ref, n, rs) if rs else np.zeros((0, n), bool)
    H = _matrix(hyp, n, hs) if hs else np.zeros((0, n), bool)

    scored = np.ones(n, dtype=bool)
    c = int(round(collar / FRAME))
    if c > 0:
        for _, s, e in ref:
            for t in (s, e):
                k = int(round(t / FRAME))
                scored[max(0, k - c):min(n, k + c)] = False

    # en iyi eşleme: ortak konuşma süresini en çoklayan referans<->sistem eşlemesi
    mapping = {}
    if rs and hs:
        overlap = (R[:, None, :] & H[None, :, :] & scored).sum(axis=2)
        ri, hi = linear_sum_assignment(-overlap)
        mapping = {h: r for r, h in zip(ri, hi)}

    n_ref = R[:, scored].sum(axis=0)          # karede kaç referans konuşmacı
    n_hyp = H[:, scored].sum(axis=0)
    correct = np.zeros(scored.sum(), dtype=int)
    for h, r in mapping.items():
        correct += (R[r, scored] & H[h, scored]).astype(int)
    total = n_ref.sum()
    missed = np.maximum(n_ref - n_hyp, 0).sum()
    fa = np.maximum(n_hyp - n_ref, 0).sum()
    conf = (np.minimum(n_ref, n_hyp) - correct).sum()
    if total == 0:
        return {"der": 0.0, "kacirilan": 0.0, "yanlis_alarm": 0.0, "karisan": 0.0}
    return {"der": float((missed + fa + conf) / total), "kacirilan": float(missed / total),
            "yanlis_alarm": float(fa / total), "karisan": float(conf / total)}


def cpwer(ref: dict[str, str], hyp: dict[str, str]) -> float:
    """ref/hyp: konuşmacı -> o kişinin tüm sözleri (sırasıyla birleştirilmiş, normalize)."""
    import jiwer

    rs, hs = list(ref), list(hyp)
    k = max(len(rs), len(hs))
    rs += [f"_bos_r{i}" for i in range(k - len(rs))]
    hs += [f"_bos_h{i}" for i in range(k - len(hs))]
    ref = defaultdict(str, ref)
    hyp = defaultdict(str, hyp)

    def errors(r, h):
        rw, hw = ref[r].split(), hyp[h].split()
        if not rw:
            return len(hw)
        o = jiwer.process_words(" ".join(rw), " ".join(hw) if hw else "")
        return o.substitutions + o.deletions + o.insertions

    cost = np.array([[errors(r, h) for h in hs] for r in rs])
    ri, hi = linear_sum_assignment(cost)
    n = sum(len(v.split()) for v in ref.values())
    return float(cost[ri, hi].sum() / max(n, 1))
