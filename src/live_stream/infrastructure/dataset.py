"""Kontrol edilmiş sonuçları veri setine aktarma: önce onay kuyruğu, sonra test ya da eğitim.

Fine-tuning projesindeki "Gerçek Test Setine Kaydet" + onay kuyruğunun karşılığı. Farklar:
  - Dosyalar repo klasörüne (tests/fixtures) değil, DATASET_DIR'e (varsayılan data/veri_seti,
    .gitignore'da) yazılır: hasta sesi yanlışlıkla commit edilmez.
  - Test ve eğitim ayrı: aynı kayıt ikisine birden giremez (sızıntı denetimi, ses özetiyle).
  - Test biçimi benchmarks/diyalog_testi.py ile, eğitim biçimi finetune/veri_hazirla.py ile
    doğrudan uyumlu (ara dönüştürme yok).

Klasörler:
  bekleyen/<iş>/   ses.wav, konusmaci.rttm, ref.json, satirlar.tsv, meta.json
  test/<dil>/      diyalog.tsv + audio/ rttm/ ref/           (DER, cpWER ölçümü)
  egitim/<dil>/    liste.tsv + audio/ (satır başına 1-20 sn)  (ince ayar)
  ozetler.tsv      onaylanan her kaydın ses özeti ve bölümü    (sızıntı denetimi)
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
USES = ("test", "egitim")


def write_pending(root: Path, job_id: str, audio: np.ndarray, utterances: list[dict], meta: dict) -> Path:
    """utterances: API'deki sonuç satırları (speaker, start, end, text, ...)."""
    d = Path(root) / "bekleyen" / job_id
    d.mkdir(parents=True, exist_ok=True)
    sf.write(d / "ses.wav", audio, SR, subtype="PCM_16")
    rows = [u for u in utterances if u["speaker"] != "BELIRSIZ"]
    with (d / "konusmaci.rttm").open("w", encoding="utf-8") as f:
        for u in rows:
            f.write(f"SPEAKER {job_id} 1 {u['start']:.3f} {u['end'] - u['start']:.3f} <NA> <NA> "
                    f"{u['speaker']} <NA> <NA>\n")
    per: dict[str, str] = {}
    for u in rows:
        per[u["speaker"]] = (per.get(u["speaker"], "") + " " + u["text"]).strip()
    (d / "ref.json").write_text(json.dumps(
        {"konusmaci_metin": per, "sira": [{"konusmaci": u["speaker"], "baslangic": u["start"],
                                           "bitis": u["end"], "metin": u["text"]} for u in rows]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    with (d / "satirlar.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["baslangic", "bitis", "konusmaci", "metin", "duzeltildi"])
        for u in utterances:
            w.writerow([u["start"], u["end"], u["speaker"], u["text"], int(bool(u.get("edited")))])
    meta = {**meta, "job_id": job_id, "sha1": audio_hash(audio), "sure_sn": round(len(audio) / SR, 2),
            "kisi_sayisi": len(per), "aktarma": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return d


def audio_hash(audio: np.ndarray) -> str:
    return hashlib.sha1(np.round(np.asarray(audio, dtype=np.float32), 4).tobytes()).hexdigest()


def pending(root: Path) -> list[dict]:
    base = Path(root) / "bekleyen"
    return [json.loads((d / "meta.json").read_text(encoding="utf-8"))
            for d in sorted(base.iterdir()) if (d / "meta.json").exists()] if base.exists() else []


def _hashes(root: Path) -> dict[str, str]:
    f = Path(root) / "ozetler.tsv"
    if not f.exists():
        return {}
    return {r["sha1"]: r["bolum"] for r in csv.DictReader(f.open(encoding="utf-8"), delimiter="\t")}


def approve(root: Path, job_id: str, use: str | None = None, approver: str = "") -> Path:
    """Bekleyen kaydı test ya da eğitim bölümüne taşır. Aynı ses başka bölümdeyse hata."""
    root = Path(root)
    d = root / "bekleyen" / job_id
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    use = use or meta.get("kullanim", "test")
    if use not in USES:
        raise ValueError(f"kullanım '{use}' olamaz: {USES}")
    seen = _hashes(root)
    if meta["sha1"] in seen:
        raise ValueError(f"bu ses zaten '{seen[meta['sha1']]}' bölümünde (sızıntı olurdu)")
    lang = meta.get("dil") or "xx"
    target = root / use / lang
    if use == "test":
        _add_test(target, d, job_id, meta)
    else:
        _add_train(target, d, job_id)
    new = not (root / "ozetler.tsv").exists()
    with (root / "ozetler.tsv").open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        if new:
            w.writerow(["sha1", "bolum", "job_id", "onaylayan", "tarih"])
        w.writerow([meta["sha1"], use, job_id, approver,
                    datetime.now(timezone.utc).isoformat(timespec="seconds")])
    shutil.rmtree(d)
    return target


def reject(root: Path, job_id: str) -> None:
    shutil.rmtree(Path(root) / "bekleyen" / job_id)


def _append(tsv: Path, header: list[str], row: list) -> None:
    new = not tsv.exists()
    with tsv.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        if new:
            w.writerow(header)
        w.writerow(row)


def _add_test(target: Path, d: Path, job_id: str, meta: dict) -> None:
    for sub in ("audio", "rttm", "ref"):
        (target / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy(d / "ses.wav", target / "audio" / f"{job_id}.wav")
    shutil.copy(d / "konusmaci.rttm", target / "rttm" / f"{job_id}.rttm")
    shutil.copy(d / "ref.json", target / "ref" / f"{job_id}.json")
    _append(target / "diyalog.tsv", ["dosya", "rttm", "ref", "kisi_sayisi", "sure_sn", "kayit", "bolum"],
            [f"audio/{job_id}.wav", f"rttm/{job_id}.rttm", f"ref/{job_id}.json", meta["kisi_sayisi"],
             meta["sure_sn"], job_id, "test"])


def _add_train(target: Path, d: Path, job_id: str) -> None:
    (target / "audio").mkdir(parents=True, exist_ok=True)
    audio, _ = sf.read(d / "ses.wav", dtype="float32")
    ref = json.loads((d / "ref.json").read_text(encoding="utf-8"))
    for i, s in enumerate(ref["sira"]):
        if not 1.0 <= s["bitis"] - s["baslangic"] <= 20.0 or not s["metin"].strip():
            continue  # icefall sınırı; uzun satırlar zaten 25 sn'de bölünüyor
        name = f"audio/{job_id}_{i:04d}.wav"
        sf.write(target / name, audio[int(s["baslangic"] * SR):int(s["bitis"] * SR)], SR, subtype="PCM_16")
        _append(target / "liste.tsv", ["dosya", "metin", "kayit"], [name, s["metin"], job_id])
