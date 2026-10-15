"""Arayüzde düzeltilen satırları ince ayar verisine çevirir: her satır ayrı ses + doğru metin.

    python scripts/duzeltmeleri_disa_aktar.py --cikti duzeltmeler --dil vi
    python scripts/duzeltmeleri_disa_aktar.py --db postgresql://... --cikti duzeltmeler

Çıktı: duzeltmeler/audio/*.wav ve duzeltmeler/liste.tsv (dosya, metin, kayit, ilk_metin).
Bu TSV doğrudan finetune/veri_hazirla.py --train/--dev'e verilir. Aynı kayıttaki satırlar
aynı "kayit" değerini taşır; veri_hazirla.py bir kaydı iki bölüme koymaz.

Yalnızca veri setine aktarma onayı verilmiş işler (arayüzde "Veri setine ekle") dışa aktarılır.
Atlananlar: sesi silinmiş işler (saklama süresi), 1 sn'den kısa ya da 20 sn'den uzun satırlar
(icefall sınırı). Veritabanı varsayılan olarak yerel modunki (DATA_DIR/jobs.db).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from live_stream.infrastructure.audio import AudioDecodeError, SoundfileLoader  # noqa: E402
from live_stream.infrastructure.db import SqlAlchemyJobStore  # noqa: E402
from live_stream.interface.settings import Settings  # noqa: E402

SR = 16000


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=None, help="veritabanı adresi (varsayılan: yerel mod)")
    p.add_argument("--cikti", required=True)
    p.add_argument("--dil", default=None, help="yalnızca bu dil (vi / tr)")
    p.add_argument("--en-kisa", type=float, default=1.0)
    p.add_argument("--en-uzun", type=float, default=20.0)
    a = p.parse_args()

    store = SqlAlchemyJobStore(a.db or Settings().local_db)
    out = Path(a.cikti)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    loader, cache = SoundfileLoader(), {}
    n, skipped = 0, {"ses yok": 0, "süre": 0, "dil": 0}
    with (out / "liste.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["dosya", "metin", "kayit", "ilk_metin", "konusmaci"])
        # yalnızca veri setine aktarma onayı verilmiş işler (kontrol eden + açık onay)
        for c in store.corrections(consented_only=True):
            if a.dil and c.language != a.dil:
                skipped["dil"] += 1
                continue
            if not a.en_kisa <= c.end - c.start <= a.en_uzun:
                skipped["süre"] += 1
                continue
            if c.audio_path not in cache:
                try:
                    cache[c.audio_path] = loader.load(c.audio_path) if c.audio_path else None
                except (AudioDecodeError, OSError, RuntimeError):
                    cache[c.audio_path] = None
            audio = cache[c.audio_path]
            if audio is None:
                skipped["ses yok"] += 1
                continue
            name = f"audio/{c.job_id}_{c.idx:04d}.wav"
            sf.write(out / name, audio[int(c.start * SR):int(c.end * SR)], SR)
            w.writerow([name, c.text, c.job_id, c.original_text, c.speaker])
            n += 1
    print(f"{n} satır yazıldı -> {out / 'liste.tsv'}; atlanan: {skipped}")
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
