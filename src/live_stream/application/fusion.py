"""Birleştirme motoru: ASR kelimelerini diarization aralıklarına atar.

Kural (yol haritasındaki gibi):
  1. Kelimenin orta noktası tek bir konuşmacının aralığına düşüyorsa o konuşmacı.
  2. Birden fazla konuşmacıya düşüyorsa (üst üste konuşma) kelimenin süresinin
     en çok örtüştüğü konuşmacı; payı min_dominance altındaysa BELIRSIZ.
  3. Hiçbirine düşmüyorsa en yakın aralık, uzaklık max_gap'ten küçükse; yoksa BELIRSIZ.
Sonra aynı konuşmacının art arda kelimeleri tek satırda (Utterance) toplanır; satır
max_utterance'tan uzunsa en uzun duraklamadan bölünür (konuşmacı değişmez, yalnızca satır).
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import replace
from collections import defaultdict

from live_stream.domain import BELIRSIZ, SpeakerSegment, Utterance, Word

from .config import FusionConfig


class FusionEngine:
    def __init__(self, config: FusionConfig | None = None):
        self.cfg = config or FusionConfig()

    def assign(self, words: list[Word], speakers: list[SpeakerSegment]) -> list[str]:
        """Her kelime için konuşmacı etiketi döndürür (aynı sırada)."""
        distinct = {s.speaker for s in speakers}
        if len(distinct) <= 1:
            # tek (ya da hiç) konuşmacı: karışacak kimse yok; VAD'ın kaçırdığı yerlerdeki
            # kelimeler de o kişiye aittir (BELIRSIZ demek bilgi kaybı olur)
            return [distinct.pop() if distinct else "speaker_00"] * len(words)
        segs = sorted(speakers, key=lambda s: s.start)
        starts = [s.start for s in segs]
        # aralıklar üst üste binebilir; orta nokta aramasında geriye bakmak için
        # o ana kadarki en uzun bitiş zamanı tutulur
        max_end_so_far, m = [], float("-inf")
        for s in segs:
            m = max(m, s.end)
            max_end_so_far.append(m)

        return [self._one(w, segs, starts, max_end_so_far) for w in words]

    def _one(self, w, segs, starts, max_end_so_far) -> str:
        if not segs:
            return BELIRSIZ
        mid = w.mid
        hi = bisect_right(starts, mid)  # segs[:hi] mid'den önce başlıyor
        lo = hi - 1
        while lo >= 0 and max_end_so_far[lo] > mid:
            lo -= 1
        hits = [s for s in segs[lo + 1:hi] if s.start <= mid < s.end]
        speakers_hit = {s.speaker for s in hits}

        if len(speakers_hit) == 1:
            return speakers_hit.pop()
        if len(speakers_hit) > 1:
            return self._by_overlap(w, hits)
        return self._nearest(mid, segs, hi)

    def _by_overlap(self, w: Word, hits: list[SpeakerSegment]) -> str:
        share = defaultdict(float)
        for s in hits:
            share[s.speaker] += max(0.0, min(w.end, s.end) - max(w.start, s.start))
        total = sum(share.values())
        if total <= 0:
            return BELIRSIZ
        best = max(share, key=share.get)
        return best if share[best] / total >= self.cfg.min_dominance else BELIRSIZ

    def _nearest(self, mid: float, segs: list[SpeakerSegment], hi: int) -> str:
        best, dist = None, float("inf")
        # mid'den sonra başlayan ilk aralık ve mid'den önce biten aralıklar
        if hi < len(segs):
            best, dist = segs[hi].speaker, segs[hi].start - mid
        for s in segs[:hi]:
            d = mid - s.end
            if 0 <= d < dist:
                best, dist = s.speaker, d
        return best if best is not None and dist <= self.cfg.max_gap else BELIRSIZ

    def merge(self, words: list[Word], labels: list[str]) -> list[Utterance]:
        """Etiketlenmiş kelimeleri konuşma satırlarına toplar."""
        out: list[tuple[Utterance, list[Word]]] = []
        cur_spk, cur = None, []
        for w, spk in zip(words, labels):
            if cur and (spk != cur_spk or w.start - cur[-1].end > self.cfg.new_utterance_gap):
                out += [(_utt(cur_spk, part), part) for part in self._split(cur)]
                cur = []
            cur_spk = spk
            cur.append(w)
        if cur:
            out += [(_utt(cur_spk, part), part) for part in self._split(cur)]
        return [_with_conf(u, ws, self.cfg.low_conf) for u, ws in out]

    def _split(self, ws: list[Word]) -> list[list[Word]]:
        """Uzun satırı, her parça en fazla max_utterance sn olacak şekilde böler. Kesim,
        satırın ilk %40'ı geçtikten sonraki en uzun duraklamaya konur (çok kısa parça olmasın)."""
        limit = self.cfg.max_utterance
        out = []
        while limit > 0 and len(ws) > 1 and ws[-1].end - ws[0].start > limit:
            t0 = ws[0].start
            cand = [i for i in range(1, len(ws)) if 0.4 * limit <= ws[i].start - t0 <= limit]
            if not cand:  # tek kelime çok uzun ya da sıkışık: sınırdaki ilk kelimeden kes
                cand = [next((i for i in range(1, len(ws)) if ws[i].start - t0 > limit), len(ws) - 1)]
            # en uzun duraklama; eşitse en geç olan (parçalar gereksiz kısalmasın)
            cut = max(cand, key=lambda i: (round(ws[i].start - ws[i - 1].end, 2), i))
            out.append(ws[:cut])
            ws = ws[cut:]
        out.append(ws)
        return out

    def fuse(self, words: list[Word], speakers: list[SpeakerSegment]) -> list[Utterance]:
        words = sorted(words, key=lambda s: s.start)
        labels = self.assign(words, speakers)
        if self.cfg.sandwich_max > 0:
            labels = self.smooth(words, labels)
        return self.merge(words, labels)

    def smooth(self, words: list[Word], labels: list[str]) -> list[str]:
        """A, kısa B, A -> A, A, A: iki yanında aynı kişi olan kısa konuşma parçası (sandwich_max
        sn'den kısa, aralarda new_utterance_gap'ten uzun sessizlik yok) o kişiye verilir.
        Kısa sıralar genelde konuşmacı ayırmanın pencere sınırı hatasıdır."""
        runs: list[list[int]] = []  # [başlangıç indeksi, bitiş indeksi (dahil)]
        for i, lab in enumerate(labels):
            if runs and labels[runs[-1][0]] == lab:
                runs[-1][1] = i
            else:
                runs.append([i, i])
        out = list(labels)
        gap = self.cfg.new_utterance_gap
        for k in range(1, len(runs) - 1):
            (p0, p1), (b0, b1), (n0, _) = runs[k - 1], runs[k], runs[k + 1]
            if out[p0] != out[n0]:
                continue
            short = words[b1].end - words[b0].start <= self.cfg.sandwich_max
            close = (words[b0].start - words[p1].end <= gap) and (words[n0].start - words[b1].end <= gap)
            if short and close:
                for i in range(b0, b1 + 1):
                    out[i] = out[p0]
        return out


def _utt(speaker: str, ws: list[Word]) -> Utterance:
    return Utterance(speaker=speaker, start=ws[0].start, end=ws[-1].end,
                     text=" ".join(s.text for s in ws))


def _with_conf(u: Utterance, ws: list[Word], low: float) -> Utterance:
    confs = [w.conf for w in ws if w.conf is not None]
    if not confs:
        return u
    return replace(u, conf=round(sum(confs) / len(confs), 3),
                   low_conf=tuple(i for i, w in enumerate(ws) if w.conf is not None and w.conf < low))
