"""Konuşma kapsaması: VAD'ın konuşma dediği yerlerin ne kadarı yazıya döküldü?

Fine-tuning projesindeki speech_gap_detector'dan uyarlandı. Whisper bazen konuşmayı atlıyor
(13 Ekim: önceki hat bir podcast parçasında 24 sn'yi hiç yazmadı); bu ölçü hem sonuçta
"kapsama %" olarak görünür hem de güvenlik ağının (yeniden çözme) hangi aralıklara
bakacağını söyler.
"""
from __future__ import annotations

from live_stream.domain import VadSegment, Word


def merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def coverage(speech: list[VadSegment], words: list[Word],
             min_gap: float = 2.0) -> tuple[list[tuple[float, float]], dict[str, float]]:
    """(min_gap'ten uzun yazısız konuşma aralıkları, özet). Aralık: konuşmanın içinde kelime
    bulunmayan kısım (baş, araları, son)."""
    regions = merge([(s.start, s.end) for s in speech if s.end > s.start])
    ws = sorted(words, key=lambda w: w.start)
    total = sum(b - a for a, b in regions)
    untranscribed, gaps = 0.0, []
    i = 0
    for a, b in regions:
        while i < len(ws) and ws[i].end <= a:
            i += 1
        cur, j = a, i
        while j < len(ws) and ws[j].start < b:
            if ws[j].start > cur:
                untranscribed += ws[j].start - cur
                if ws[j].start - cur >= min_gap:
                    gaps.append((cur, ws[j].start))
            cur = max(cur, ws[j].end)
            j += 1
        if b > cur:
            untranscribed += b - cur
            if b - cur >= min_gap:
                gaps.append((cur, b))
    pct = 100.0 if total <= 0 else max(0.0, (total - untranscribed) / total * 100)
    return gaps, {"konusma_sn": round(total, 2), "yazisiz_sn": round(untranscribed, 2),
                  "kapsama_yuzde": round(pct, 1), "yazisiz_aralik": len(gaps)}


def add_rescued(words: list[Word], rescued: list[Word]) -> tuple[list[Word], int]:
    """Kurtarılan kelimelerden, var olan bir kelimeyle çakışanları atar (projedeki güvenlik
    ağında kenar payı yüzünden aynı kelime iki kez yazılabiliyordu)."""
    spans = sorted((w.start, w.end) for w in words)
    added = []
    for r in rescued:
        mid = r.mid
        if any(a - 0.05 <= mid <= b + 0.05 for a, b in spans):
            continue
        added.append(r)
    return sorted(words + added, key=lambda w: w.start), len(added)
