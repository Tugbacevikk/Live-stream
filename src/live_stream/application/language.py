"""Dil kararı: istekte dil varsa onu doğrula, yoksa sesten tespit et.

Tespit, modelin bildiği tüm dillerin olasılığını alır ve iki soru sorar:
  1. Desteklenen dillerin toplam olasılığı yeterli mi? Değilse ses başka bir
     dilde demektir -> UnsupportedLanguageError (iş reddedilir; yanlış modelle
     anlamsız metin üretmekten iyidir).
  2. Desteklenen diller arasında en olası hangisi ve ne kadar emin?
     Güven min_confidence altındaysa yine o dille devam edilir ama sonuç
     "uncertain" işaretlenir.
Kademeli tespit: önce ucuz model (whisper-tiny, ~0,25 sn); ret ya da düşük
güven çıkarsa bir sonraki, daha güçlü model (whisper-small, ~1 sn) denenir.
Son kararı en son denenen model verir.

Ölçüm (12 Ekim):
  - tiny, kısıtlı seçim: 61/61 doğru; 99 dil arasında serbest seçim gerçek
    seslerin 7'sinde 4'ünü yanlış dile atadı -> kısıtlama şart.
  - tiny 1,6 sn'lik gerçek Vietnamca seste vi+tr toplamını %3 verdi (yanlış ret);
    small aynı seste %90. Diğer 29 dilde vi+tr toplamı tiny'de en fazla %5,
    small'da %0,4 -> tiny yanlış kabul yapmadı, sadece kısa seste yanlış ret
    yaptı; kademe bunu small'a devrederek çözer.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from live_stream.domain import LanguageDecision, LanguageSource, UnsupportedLanguageError, VadSegment

from .config import LidConfig
from .ports import LanguageDetector


class LanguageResolver:
    def __init__(self, supported: list[str],
                 detectors: LanguageDetector | Sequence[LanguageDetector] | None,
                 cfg: LidConfig | None = None):
        if not supported:
            raise ValueError("en az bir dil tanımlı olmalı")
        self.supported = [s.lower() for s in supported]
        if detectors is None:
            detectors = []
        elif not isinstance(detectors, Sequence):
            detectors = [detectors]
        self.detectors = list(detectors)
        self.cfg = cfg or LidConfig()

    def resolve(self, requested: str | None, segments: list[VadSegment]) -> LanguageDecision:
        if requested:
            code = requested.strip().lower()
            if code not in self.supported:
                raise UnsupportedLanguageError(
                    f"istenen dil '{code}' desteklenmiyor; desteklenenler: {', '.join(self.supported)}")
            return LanguageDecision(code, 1.0, LanguageSource.REQUEST)

        if len(self.supported) == 1:
            return LanguageDecision(self.supported[0], 1.0, LanguageSource.DEFAULT)
        speech = self._speech(segments)
        if speech.size == 0:
            # konuşma yok: metin de çıkmayacak, dil önemsiz
            return LanguageDecision(self.supported[0], 0.0, LanguageSource.DEFAULT)
        if not self.detectors:
            raise UnsupportedLanguageError("dil belirtilmedi ve dil tespiti kapalı")

        for det in self.detectors:
            result = self._evaluate(det.probabilities(speech))
            if isinstance(result, LanguageDecision) and not result.uncertain:
                return result
        if isinstance(result, UnsupportedLanguageError):
            raise result
        return result

    def _evaluate(self, probs: dict[str, float]):
        mass = sum(probs.get(c, 0.0) for c in self.supported)
        if mass < self.cfg.min_supported:
            top = max(probs, key=probs.get)
            return UnsupportedLanguageError(
                f"ses desteklenmeyen bir dilde görünüyor (en olası: {top} %{probs[top]*100:.0f}; "
                f"desteklenen dillerin toplamı %{mass*100:.0f})", detected=top)
        restricted = {c: probs.get(c, 0.0) / mass for c in self.supported}
        code = max(restricted, key=restricted.get)
        conf = restricted[code]
        return LanguageDecision(code, round(conf, 4), LanguageSource.DETECTED,
                                uncertain=conf < self.cfg.min_confidence)

    def _speech(self, segments: list[VadSegment]) -> np.ndarray:
        """Tespit için yalnızca konuşma kısmı, en fazla max_seconds (sessizlik tespiti bozar)."""
        out, n, limit = [], 0, int(self.cfg.max_seconds * 16000)
        for s in segments:
            if n >= limit:
                break
            out.append(s.samples[: limit - n])
            n += len(out[-1])
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
