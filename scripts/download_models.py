"""Modelleri models/ klasörüne indirir, açar, bpe.vocab ve MANIFEST.json üretir.

    python scripts/download_models.py                          # varsayılanlar: vi=zipformer-68m, tr=faster-whisper-medium
    python scripts/download_models.py --tr faster-whisper-small  # canlı altyazı için hızlı Türkçe model
    python scripts/download_models.py --vi zipformer-30m --tr omnilingual-300m
    python scripts/download_models.py --vi all --tr all        # tüm adaylar (karşılaştırma için)
    python scripts/download_models.py --tr none                # yalnızca Vietnamca
    python scripts/download_models.py --vi none                # yalnızca Türkçe (LANGUAGES=tr)

Her zaman inenler: Silero VAD, pyannote segmentation, WeSpeaker, Whisper-tiny ve -small (dil tespiti).
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
    ASR_PACKS, AUX_PACKS, BASE, DEFAULT_MODELS, MODEL_CHOICES, Pack, is_folder_model,
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


def make_bpe_vocab(model_dir: Path) -> None:
    """Hotword için gerekli bpe.vocab (yoksa sherpa-onnx hotword'leri sessizce yok sayar)."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(model_dir / "bpe.model"))
    with (model_dir / "bpe.vocab").open("w", encoding="utf-8") as f:
        for i in range(sp.get_piece_size()):
            f.write(f"{sp.id_to_piece(i)}\t{sp.get_score(i)}\n")


def choose(arg: str, lang: str) -> list[str]:
    if arg == "none":
        return []
    if arg == "all":
        return MODEL_CHOICES[lang]
    return [arg]


def _env_model(lang: str) -> str:
    """.env'deki VI_MODEL / TR_MODEL (baslat.bat ile aynı seçim), yoksa varsayılan.
    LANGUAGES'ta olmayan dilin modeli indirilmez (LANGUAGES=tr -> Vietnamca inmez)."""
    import os

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    langs = [x.strip().lower() for x in os.environ.get("LANGUAGES", "tr,vi").split(",") if x.strip()]
    if lang not in langs:
        return "none"
    v = os.environ.get(f"{lang.upper()}_MODEL")
    # klasör yolu verilmişse (ince ayarlı model) indirilecek bir şey yok; yedek olarak varsayılan iner
    return DEFAULT_MODELS[lang] if not v or is_folder_model(v) else v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vi", choices=[*MODEL_CHOICES["vi"], "all", "none"], default=_env_model("vi"))
    p.add_argument("--tr", choices=[*MODEL_CHOICES["tr"], "all", "none"], default=_env_model("tr"))
    p.add_argument("--dest", default="models")
    a = p.parse_args()
    dest = Path(a.dest)
    dest.mkdir(exist_ok=True)

    packs = dict(AUX_PACKS)
    keys = choose(a.vi, "vi") + choose(a.tr, "tr")
    live = os.environ.get("LIVE_MODEL")  # canlı altyazı için ayrı Türkçe model seçildiyse o da iner
    if live in MODEL_CHOICES["tr"] and live not in keys and a.tr != "none":
        keys.append(live)
    for key in keys:
        packs[key] = ASR_PACKS[key]

    mpath = dest / "MANIFEST.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    for key, pack in packs.items():
        if key in manifest and (dest / pack.target).exists():
            continue  # inmiş ve kayıtlı: yeniden özet hesaplamaya gerek yok (her başlatmada hızlı)
        path = fetch(pack, dest)
        if pack.needs_bpe_vocab and not (path / "bpe.vocab").exists():
            try:
                make_bpe_vocab(path)
            except ImportError:  # yalnızca Vietnamca terim listesi (VI_HOTWORDS) için gerekli
                print("UYARI: sentencepiece kurulu değil, bpe.vocab üretilmedi; VI_HOTWORDS "
                      "çalışmaz (pip install sentencepiece, sonra bu betiği yeniden çalıştırın)")
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
