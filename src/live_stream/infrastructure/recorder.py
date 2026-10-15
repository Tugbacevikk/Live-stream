"""
Oturum kaydı: gelen sesi ve kesinleşen metni anında diske yazar.

Program/bilgisayar çökerse bile o ana kadarki ses (.wav) ve metin (.txt/.jsonl) kaybolmaz.
Oturum bitince .wav toplu işe gönderilir (LIVE_AUTO_ANALYZE): konuşmacı ayrımlı transkript.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf


def _fmt(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class SessionRecorder:
    def __init__(self, directory: str | Path, name: str | None = None):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name = name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.wav_path = self.dir / f"{self.name}.wav"
        self.txt_path = self.dir / f"{self.name}.txt"
        self.jsonl_path = self.dir / f"{self.name}.jsonl"
        self._wav = sf.SoundFile(self.wav_path, mode="w", samplerate=16000, channels=1, subtype="PCM_16")
        self._txt = open(self.txt_path, "a", encoding="utf-8")
        self._jsonl = open(self.jsonl_path, "a", encoding="utf-8")
        self._writes = 0

    @property
    def audio_path(self) -> str:
        return str(self.wav_path.resolve())

    def write_audio(self, samples: np.ndarray) -> None:
        self._wav.write(samples)
        self._writes += 1
        if self._writes % 50 == 0:  # ~5 sn'de bir diske boşalt
            self._wav.flush()

    def write_final(self, item: dict) -> None:
        spk = item.get("speaker")
        who = f"Konuşmacı {int(spk.rsplit('_', 1)[-1]) + 1}: " if spk else ""
        self._txt.write(f"[{_fmt(item['start'])}] {who}{item['text']}\n")
        self._txt.flush()
        self._jsonl.write(json.dumps(item, ensure_ascii=False) + "\n")
        self._jsonl.flush()

    def close(self) -> None:
        for f in (self._wav, self._txt, self._jsonl):
            try:
                f.close()
            except Exception:
                pass


def delete_old_recordings(directory: str | Path, older_than_days: float, now: float | None = None) -> int:
    """recordings/ altında son değişikliği older_than_days günden eski oturum dosyalarını siler
    (.wav, .txt, .jsonl). Açık oturumun dosyaları sürekli yazıldığı için silinmez. 0 = silme.
    Silinen dosya sayısını döndürür."""
    import time

    if older_than_days <= 0:
        return 0
    d = Path(directory)
    if not d.is_dir():
        return 0
    limit = (now if now is not None else time.time()) - older_than_days * 86400
    n = 0
    for f in d.iterdir():
        if f.suffix in (".wav", ".txt", ".jsonl") and f.is_file():
            try:
                if f.stat().st_mtime < limit:
                    f.unlink()
                    n += 1
            except OSError:
                pass
    return n
