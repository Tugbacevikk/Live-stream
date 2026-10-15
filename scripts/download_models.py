"""Modelleri models/ klasörüne indirir, açar ve MANIFEST.json (sha256) üretir.

    python scripts/download_models.py                          # varsayılan: tr=faster-whisper-medium
    python scripts/download_models.py --tr faster-whisper-small  # canlı altyazı için hızlı model
    python scripts/download_models.py --tr all                 # tüm Türkçe adaylar (karşılaştırma için)

Her zaman inenler: Silero VAD (toplu iş + canlı altyazı), CAM++ ve WeSpeaker (konuşmacı imzası),
pyannote segmentation (DIARIZER=pyannote seçeneği).
İnternetsiz (air-gapped) sunucu için: bunu internetli makinede çalıştırıp models/
klasörünü olduğu gibi taşıyın. MANIFEST.json içindeki sha256 değerleri ile kontrol edin.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from live_stream.infrastructure.catalog import (  # noqa: E402
    ASR_PACKS, AUX_PACKS, BASE, DEFAULT_MODELS, MODEL_CHOICES, Pack,
)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(pack: Pack, dest: Path) -> Path:
    out = dest / pack.target
    if out.exists():
        print(f"var, atlanıyor: {pack.target}")
        return out
    if pack.hf_repo:
        from huggingface_hub import snapshot_download  # faster-whisper ile gelir

        print(f"indiriliyor: {pack.hf_repo}")
        snapshot_download(pack.hf_repo, revision=pack.hf_revision, local_dir=out,
                          allow_patterns=list(pack.files.values()))
        return out
    name = pack.url_path.rsplit("/", 1)[1]
    tmp = dest / name
    print(f"indiriliyor: {name}")
    urllib.request.urlretrieve(f"{BASE}/{pack.url_path}", tmp)
    if name.endswith(".tar.bz2"):
        with tarfile.open(tmp, "r:bz2") as t:
            t.extractall(dest, filter="data")
        tmp.unlink()
    if pack.keep and out.is_dir():
        for f in out.iterdir():
            if f.is_file() and f.name not in pack.keep:
                f.unlink()  # ör. fp32 kopyalar: yalnızca int8 kullanılıyor
    return out


def choose(arg: str, lang: str) -> list[str]:
    if arg == "none":
        return []
    if arg == "all":
        return MODEL_CHOICES[lang]
    return [arg]


def _env_model(lang: str) -> str:
    """.env'deki TR_MODEL (baslat.bat ile aynı seçim), yoksa varsayılan."""
    import os

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    v = os.environ.get(f"{lang.upper()}_MODEL")
    return v if v in MODEL_CHOICES[lang] else DEFAULT_MODELS[lang]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tr", choices=[*MODEL_CHOICES["tr"], "all", "none"], default=_env_model("tr"))
    p.add_argument("--dest", default="models")
    a = p.parse_args()
    dest = Path(a.dest)
    dest.mkdir(exist_ok=True)

    packs = dict(AUX_PACKS)
    keys = choose(a.tr, "tr")
    live = os.environ.get("LIVE_MODEL")  # canlı altyazı için ayrı model seçildiyse o da iner
    if live in MODEL_CHOICES["tr"] and live not in keys:
        keys.append(live)
    for key in keys:
        packs[key] = ASR_PACKS[key]

    mpath = dest / "MANIFEST.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    for key, pack in packs.items():
        if key in manifest and (dest / pack.target).exists():
            continue  # inmiş ve kayıtlı: yeniden özet hesaplamaya gerek yok (her başlatmada hızlı)
        path = fetch(pack, dest)
        if path.is_file():
            files = [path]
        elif pack.hf_repo:
            files = [path / f for f in pack.files.values()]
        else:
            files = sorted(path.rglob("*.onnx"))
        url = f"hf://{pack.hf_repo}" if pack.hf_repo else f"{BASE}/{pack.url_path}"
        manifest[key] = {"url": url, "license": pack.license,
                         "sha256": {f.relative_to(dest).as_posix(): sha256(f) for f in files}}
    mpath.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Tamam. {mpath} yazıldı.")


if __name__ == "__main__":
    main()
