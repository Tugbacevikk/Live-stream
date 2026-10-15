"""Kabul testi: "1 dakikalık ses en geç 1 dakikada biter" bu bilgisayarda sağlanıyor mu?

    python benchmarks/kabul_testi.py ses1.wav ses2.mp3            # dil sesten bulunur
    python benchmarks/kabul_testi.py --dil tr kayit.wav           # Türkçe, tespit atlanır
    python benchmarks/kabul_testi.py --dil tr --tr-model faster-whisper-small kayit.wav
    python benchmarks/kabul_testi.py --device cpu kayit.wav       # sunucu koşulu: yalnızca CPU
    python benchmarks/kabul_testi.py --es-zamanli 4 a.wav b.wav   # aynı anda 4 iş (yük altında)

Her dosya tüm hattan geçer (dil, ASR, konuşmacı, birleştirme). Ölçülen: işlem süresi /
ses süresi (oran). Hedef: en kötü oran <= 1. Modeller ilk iş öncesi yüklenir (sunucuda
da açılışta yüklenir), yani ölçüm kullanıcının gördüğü süredir.

Sonuç kabul_sonucu.json'a yazılır: donanım, işletim sistemi, model dosyalarının sha256
özeti (MANIFEST), git sürümü — aynı sonuç başka makinede tekrar üretilebilsin diye.
Çıkış kodu: hedef sağlandıysa 0, sağlanmadıysa 1 (otomatik kontrolde kullanılabilir).
Windows'ta da çalışır (Linux'a özel modül yok).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from live_stream.domain import AudioJob  # noqa: E402
from live_stream.infrastructure.memory_store import MemoryJobStore  # noqa: E402
from live_stream.interface.composition import build_pipeline, default_config  # noqa: E402
from live_stream.interface.settings import available_cpus  # noqa: E402


def git_version() -> str:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "describe", "--always", "--dirty"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "?"
    except Exception:
        return "?"


def hardware() -> dict:
    info = {"isletim_sistemi": platform.platform(), "islemci": platform.processor() or platform.machine(),
            "cekirdek_mantiksal": os.cpu_count(), "kullanilabilir_cekirdek": available_cpus(),
            "python": platform.python_version()}
    try:
        import psutil

        info["bellek_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
        info["cekirdek_fiziksel"] = psutil.cpu_count(logical=False)
    except ImportError:
        pass
    return info


def manifest_digest(models_dir: Path) -> str | None:
    m = models_dir / "MANIFEST.json"
    return hashlib.sha256(m.read_bytes()).hexdigest()[:16] if m.exists() else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dosyalar", nargs="+")
    p.add_argument("--dil", default=None, help="vi / tr; verilmezse sesten bulunur")
    p.add_argument("--vi-model", default=None)
    p.add_argument("--tr-model", default=None)
    p.add_argument("--device", default="cpu", choices=["cpu", "auto", "cuda"],
                   help="faster-whisper cihazı (sunucu hedefi CPU olduğu için varsayılan cpu)")
    p.add_argument("--models-dir", default=str(ROOT / "models"))
    p.add_argument("--threads", type=int, default=2, help="iş başına çekirdek (ONNX_THREADS)")
    p.add_argument("--es-zamanli", type=int, default=1, help="aynı anda işlenen ses sayısı")
    p.add_argument("--tekrar", type=int, default=1, help="her dosya kaç kez işlensin")
    p.add_argument("--hedef", type=float, default=1.0, help="izin verilen en kötü oran")
    p.add_argument("--cikti", default="kabul_sonucu.json")
    a = p.parse_args()

    models = {k: v for k, v in (("vi", a.vi_model), ("tr", a.tr_model)) if v}
    cfg = default_config(a.models_dir, models=models, threads=a.threads, device=a.device)
    pipe = build_pipeline(cfg, MemoryJobStore())
    t0 = time.perf_counter()
    langs = [a.dil] if a.dil else list(cfg.languages)
    for code in langs:
        pipe.engines[code]  # model yükleme ölçüme girmesin (sunucu açılışta yükler)
    load_s = time.perf_counter() - t0
    print(f"modeller yüklendi: {load_s:.1f} sn ({', '.join(f'{c}={cfg.languages[c].asr_model.name}' for c in langs)})")

    jobs = [(f, i) for i in range(a.tekrar) for f in a.dosyalar]

    def run(item):
        f, i = item
        start = time.perf_counter()
        t = pipe.run(AudioJob(f"{Path(f).stem}_{i}", str(Path(f).resolve()), a.dil))
        dt = time.perf_counter() - start
        return {"dosya": f, "sure_sn": round(t.duration, 2), "islem_sn": round(dt, 2),
                "oran": round(dt / max(t.duration, 1e-6), 3), "dil": t.language.code,
                "model": t.model, "konusmaci": len({u.speaker for u in t.utterances} - {"BELIRSIZ"})}

    wall = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.es_zamanli) as ex:
        results = []
        for r in ex.map(run, jobs):
            results.append(r)
            ok = "OK " if r["oran"] <= a.hedef else "AŞTI"
            print(f"  [{ok}] {r['dosya']}: {r['sure_sn']} sn ses -> {r['islem_sn']} sn "
                  f"(oran {r['oran']}, dil {r['dil']}, {r['konusmaci']} kişi)")
    wall = time.perf_counter() - wall

    worst = max(r["oran"] for r in results)
    total_audio = sum(r["sure_sn"] for r in results)
    ozet = {"en_kotu_oran": worst, "hedef": a.hedef, "gecti": worst <= a.hedef,
            "toplam_ses_sn": round(total_audio, 1), "toplam_duvar_sn": round(wall, 1),
            "verim_ses_sn_per_sn": round(total_audio / wall, 2), "model_yukleme_sn": round(load_s, 1),
            "es_zamanli": a.es_zamanli, "threads": a.threads}
    out = {"tarih": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": git_version(),
           "donanim": hardware(), "modeller": {c: cfg.languages[c].asr_model.name for c in langs},
           "manifest_sha256": manifest_digest(Path(a.models_dir)), "ozet": ozet, "dosyalar": results}
    Path(a.cikti).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nEn kötü oran {worst} (hedef <= {a.hedef}): {'GEÇTİ' if ozet['gecti'] else 'GEÇMEDİ'}. "
          f"Verim: saniyede {ozet['verim_ses_sn_per_sn']} sn ses. -> {a.cikti}")
    return 0 if ozet["gecti"] else 1


if __name__ == "__main__":
    sys.exit(main())
