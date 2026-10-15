"""Dil -> ASR motoru eşlemesi; her motor ilk kullanıldığında yüklenir.

Whisper turbo ~1,6 GB bellek tutar. Hiç Türkçe ses gelmeyen bir worker'ın onu
yüklemesi gereksiz, bu yüzden motorlar tembel (lazy) yüklenir. İstenirse
warm_up() ile başlangıçta hepsi yüklenir (ilk işte gecikme olmasın diye).
"""
from __future__ import annotations

import threading
from collections.abc import Callable, Mapping

from live_stream.application.ports import AsrEngine


class LazyAsrEngines(Mapping):
    def __init__(self, factories: dict[str, Callable[[], AsrEngine]]):
        self._factories = dict(factories)
        self._loaded: dict[str, AsrEngine] = {}
        self._lock = threading.Lock()

    def __getitem__(self, code: str) -> AsrEngine:
        if code not in self._loaded:
            with self._lock:
                if code not in self._loaded:
                    self._loaded[code] = self._factories[code]()  # KeyError: dil yok
        return self._loaded[code]

    def __iter__(self):
        return iter(self._factories)

    def __len__(self):
        return len(self._factories)

    def loaded(self) -> list[str]:
        return list(self._loaded)

    def warm_up(self) -> None:
        for code in self._factories:
            self[code]
