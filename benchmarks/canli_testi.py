"""Canlı altyazı ölçümü: bir kayıt canlı hattan (Silero + cümle sonu + ara/kesin metin) geçer.

    python benchmarks/canli_testi.py prova.wav                          # Türkçe, .env'deki model
    python benchmarks/canli_testi.py prova.wav --metin prova.txt        # + doğruluk (WER)
    python benchmarks/canli_testi.py --dil vi kayit.wav
    python benchmarks/canli_testi.py --live-model faster-whisper-small --device cuda prova.wav
    python benchmarks/canli_testi.py --tsv test.tsv                     # dosya<TAB>metin sütunları

Ses tarayıcıdaki gibi 100 ms'lik paketlerle verilir; ara metin ve kesin metin modelden geçer ve
süreleri ölçülür. Ekrandaki gecikme yaklaşık:  cümle sonu duraklaması (END_SILENCE_MS) +
kesin metin süresi (+ o sırada sürmekte olan ara metin). Raporlanan:

  kesin_ms   p50/p95/en kötü: bir cümlenin kesin metninin üretim süresi
  gecikme_s  p95: END_SILENCE_MS + kesin_ms p95 + ara_ms p95 (en kötü durumda sırada bekleme)
  oran       toplam işlem süresi / ses süresi (1'in altında olmalı; değilse gecikme birikir)
  wer        doğru metin verildiyse kelime hata oranı (büyük/küçük harf ve noktalama sayılmaz)

Hedef (öneri): gecikme_s p95 <= 2,5 sn ve oran < 0,7. Aşılırsa: --live-model
faster-whisper-small, PARTIAL_INTERVAL_SEC'i artırmak (2) ya da 0 yapmak.
Sonuç canli_sonucu.json'a yazılır.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from live_stream.application.live import (  # noqa: E402
    FinalJob, PartialJob, Segmenter, SegmenterConfig,
)
from live_stream.infrastructure.audio import SoundfileLoader  # noqa: E402
from live_stream.infrastructure.live_vad import make_detector  # noqa: E402
from live_stream.infrastructure.normalizer import live_normalizer, normalizer_for, turkish_lower  # noqa: E402
from live_stream.infrastructure.sherpa_asr import build_asr  # noqa: E402
from live_stream.interface.composition import default_config  # noqa: E402
from live_stream.interface.settings import Settings  # noqa: E402

CHUNK = 1600  # 100 ms


def norm(text: str, lang: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = turkish_lower(text) if lang == "tr" else text.lower()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def pct(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else 0.0


def run_file(path: str, asr, lang: str, s: Settings, vad_model: str) -> dict:
    audio = SoundfileLoader().load(path)
    seg = Segmenter(make_detector(s.live_vad, vad_model, s.live_vad_threshold),
                    SegmenterConfig(end_silence_ms=s.end_silence_ms, max_utterance_sec=s.max_utterance_sec,
                                    partial_interval_sec=s.partial_interval_sec))
    disp = live_normalizer(lang)
    finals, partial_ms, final_ms, context = [], [], [], []
    busy = 0.0
    t_all = time.perf_counter()
    for i in range(0, len(audio), CHUNK):
        jobs = seg.feed(audio[i:i + CHUNK])
        if i + CHUNK >= len(audio):
            jobs += seg.flush()
        for job in jobs:
            prompt = " ".join(context[-3:])[-220:] or None
            t0 = time.perf_counter()
            if isinstance(job, PartialJob):
                asr.transcribe_text(job.audio, fast=True, prompt=prompt, normalizer=disp)
                partial_ms.append((time.perf_counter() - t0) * 1000)
            elif isinstance(job, FinalJob):
                text = asr.transcribe_text(job.audio, fast=False, prompt=prompt, normalizer=disp)
                final_ms.append((time.perf_counter() - t0) * 1000)
                if text:
                    finals.append({"start": job.start, "end": job.end, "text": text})
                    context.append(text)
            busy += time.perf_counter() - t0
    dur = len(audio) / 16000
    return {"dosya": path, "sure_sn": round(dur, 2), "cumle": len(finals),
            "islem_sn": round(busy, 2), "duvar_sn": round(time.perf_counter() - t_all, 2),
            "oran": round(busy / max(dur, 1e-6), 3), "partial_ms": partial_ms, "final_ms": final_ms,
            "metin": " ".join(f["text"] for f in finals), "satirlar": finals}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dosyalar", nargs="*")
    p.add_argument("--dil", default="tr", choices=["tr", "vi"])
    p.add_argument("--metin", default=None, help="tek dosyanın doğru metni (.txt)")
    p.add_argument("--tsv", default=None, help="dosya<TAB>metin sütunlu liste (WER için)")
    p.add_argument("--live-model", default=None, help="Türkçe canlı model (varsayılan: LIVE_MODEL ya da TR_MODEL)")
    p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    p.add_argument("--models-dir", default=None)
    p.add_argument("--cikti", default="canli_sonucu.json")
    a = p.parse_args()

    s = Settings()
    models_dir = a.models_dir or s.models_dir
    items = [(f, None) for f in a.dosyalar]
    if a.metin and len(items) == 1:
        items = [(items[0][0], Path(a.metin).read_text(encoding="utf-8"))]
    if a.tsv:
        base = Path(a.tsv).parent
        for row in csv.DictReader(open(a.tsv, encoding="utf-8"), delimiter="\t"):
            f = Path(row["dosya"])
            items.append((str(f if f.is_absolute() else base / f), row.get("metin")))
    if not items:
        p.error("en az bir ses dosyası ya da --tsv verin")

    model = (a.live_model or s.live_model or s.model_for("tr")) if a.dil == "tr" else s.model_for("vi")
    cfg = default_config(models_dir, (a.dil,), {a.dil: model}, s.threads,
                         device=a.device or s.whisper_device, compute_type=s.whisper_compute_type,
                         hotwords=s.hotwords, vi_hotwords=s.vi_hotwords)
    t0 = time.perf_counter()
    asr = build_asr(cfg.languages[a.dil], normalizer_for(a.dil))
    load_s = time.perf_counter() - t0
    device = getattr(asr, "device", "cpu")
    print(f"model: {model} ({device}), yükleme {load_s:.1f} sn; END_SILENCE_MS={s.end_silence_ms}, "
          f"PARTIAL_INTERVAL_SEC={s.partial_interval_sec}")

    results, errs, words = [], 0.0, 0
    for path, ref in items:
        r = run_file(path, asr, a.dil, s, cfg.vad.model)
        if ref:
            import jiwer

            h, rr = norm(r["metin"], a.dil), norm(ref, a.dil)
            r["wer"] = round(jiwer.wer(rr, h), 4) if rr else None
            n = len(rr.split())
            errs, words = errs + (r["wer"] or 0) * n, words + n
        fm, pm = r["final_ms"], r["partial_ms"]
        print(f"{Path(path).name}: {r['sure_sn']:.0f} sn, {r['cumle']} cümle, oran {r['oran']:.2f}, "
              f"kesin p95 {pct(fm, 95):.0f} ms, ara p95 {pct(pm, 95):.0f} ms"
              + (f", WER %{r['wer'] * 100:.1f}" if r.get("wer") is not None else ""))
        results.append(r)

    fm = [x for r in results for x in r["final_ms"]]
    pm = [x for r in results for x in r["partial_ms"]]
    total_dur = sum(r["sure_sn"] for r in results)
    ozet = {
        "model": model, "cihaz": device, "dil": a.dil, "yukleme_sn": round(load_s, 1),
        "end_silence_ms": s.end_silence_ms, "partial_interval_sec": s.partial_interval_sec,
        "kesin_ms": {"p50": round(pct(fm, 50)), "p95": round(pct(fm, 95)), "en_kotu": round(max(fm or [0]))},
        "ara_ms": {"p50": round(pct(pm, 50)), "p95": round(pct(pm, 95))},
        "gecikme_s_p95": round((s.end_silence_ms + pct(fm, 95) + pct(pm, 95)) / 1000, 2),
        "oran": round(sum(r["islem_sn"] for r in results) / max(total_dur, 1e-6), 3),
        "wer": round(errs / words, 4) if words else None,
    }
    print("\nÖZET")
    print(f"  kesin metin  p50 {ozet['kesin_ms']['p50']} ms, p95 {ozet['kesin_ms']['p95']} ms, "
          f"en kötü {ozet['kesin_ms']['en_kotu']} ms")
    print(f"  ara metin    p50 {ozet['ara_ms']['p50']} ms, p95 {ozet['ara_ms']['p95']} ms")
    print(f"  ekranda gecikme (tahmini p95): {ozet['gecikme_s_p95']} sn   (hedef <= 2,5)")
    print(f"  oran (işlem / ses): {ozet['oran']}   (hedef < 0,7)")
    if ozet["wer"] is not None:
        print(f"  WER: %{ozet['wer'] * 100:.1f}")
    for r in results:
        r.pop("final_ms"), r.pop("partial_ms")
    Path(a.cikti).write_text(json.dumps({"ozet": ozet, "dosyalar": results}, ensure_ascii=False, indent=1),
                             encoding="utf-8")
    print(f"\n{a.cikti} yazıldı.")
    ok = ozet["gecikme_s_p95"] <= 2.5 and ozet["oran"] < 0.7
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
