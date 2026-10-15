"""Konuşmacı ayrıştırma testi: konuşmacı sayısı doğru mu, cümleler doğru kişide mi?

İki kaynak:
  --kaynak tts     (varsayılan) Piper sesleriyle sentetik diyaloglar üretir:
                   her dil için 3 ses, 1-2-3 konuşmacılı tüm kombinasyonlar.
                   Sesler models/tts-test/ altına bir kez indirilir (~400 MB).
  --kaynak tsv X   Gerçek, tek konuşmacılı kayıtlardan diyalog kurar. TSV sütunları:
                   dosya, konusmaci (aynı kişinin kayıtları aynı etiket).
                   VietMed'in referans.tsv dosyasında 'konusmaci' sütunu zaten var.

    python benchmarks/konusmaci_testi.py
    python benchmarks/konusmaci_testi.py --bosluksuz          # sıra değişiminde duraksama yok
    python benchmarks/konusmaci_testi.py --kaynak tsv C:/vietnamca_test/vietmed_1_2dk/referans.tsv
    python benchmarks/konusmaci_testi.py --diarizer pyannote  # eski yolla karşılaştırma
"""
from __future__ import annotations

import argparse
import csv
import itertools
import tarfile
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

from live_stream.application.config import VadConfig
from live_stream.infrastructure.audio import SoundfileLoader
from live_stream.infrastructure.sherpa_vad import SileroVad
from live_stream.interface.composition import build_diarizer, default_config

TTS = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models"
VOICES = {"vi": ["vi_VN-vais1000-medium", "vi_VN-25hours_single-low", "vi_VN-vivos-x_low"],
          "tr": ["tr_TR-dfki-medium", "tr_TR-fahrettin-medium", "tr_TR-fettah-medium"]}
TEXT = {
    "vi": ["Chào bác sĩ, hôm nay tôi bị đau đầu và sốt nhẹ.", "Bạn đã uống thuốc gì chưa?",
           "Tôi bị ho khoảng ba ngày rồi.", "Bác sĩ khuyên tôi nên nghỉ ngơi và uống nhiều nước.",
           "Huyết áp của anh hơi cao, cần theo dõi thêm.", "Chị có tiền sử bệnh tiểu đường không?",
           "Ngày mai tôi sẽ đi xét nghiệm máu.", "Cảm ơn bác sĩ rất nhiều."],
    "tr": ["Merhaba doktor bey, iki gündür başım çok ağrıyor.", "Daha önce herhangi bir ilaç kullandınız mı?",
           "Ateşiniz var mı, ölçtünüz mü?", "Size bir kan tahlili yazacağım.",
           "Tansiyonunuz biraz yüksek görünüyor.", "Günde üç kez tok karnına alın.",
           "Öksürüğüm geceleri daha da artıyor.", "Teşekkür ederim, iyi günler."],
}


def tts_clips(models_dir: Path) -> dict[str, dict[str, list[np.ndarray]]]:
    import sherpa_onnx
    import soxr

    d = models_dir / "tts-test"
    d.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict[str, list]] = {}
    for lang, voices in VOICES.items():
        out[lang] = {}
        for v in voices:
            vd = d / f"vits-piper-{v}"
            if not vd.exists():
                print(f"indiriliyor: {v}")
                tmp = d / f"{v}.tar.bz2"
                urllib.request.urlretrieve(f"{TTS}/vits-piper-{v}.tar.bz2", tmp)
                with tarfile.open(tmp, "r:bz2") as t:
                    t.extractall(d, filter="data")
                tmp.unlink()
            tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(vd / f"{v}.onnx"), tokens=str(vd / "tokens.txt"),
                    data_dir=str(vd / "espeak-ng-data")), num_threads=2)))
            clips = []
            for text in TEXT[lang]:
                a = tts.generate(text, sid=0, speed=1.0)
                clips.append(soxr.resample(np.asarray(a.samples, np.float32), a.sample_rate, 16000))
            out[lang][v] = clips
    return out


def tsv_clips(tsv: Path) -> dict[str, dict[str, list[np.ndarray]]]:
    loader, by = SoundfileLoader(), defaultdict(list)
    for r in csv.DictReader(tsv.open(encoding="utf-8"), delimiter="\t"):
        f = Path(r["dosya"])
        f = f if f.is_absolute() or f.exists() else tsv.parent / f
        if f.exists() and len(by[r["konusmaci"]]) < 8:
            a = loader.load(str(f))
            by[r["konusmaci"]].append(a[: 8 * 16000])  # sıra başına en fazla 8 sn
    spk = [s for s, c in by.items() if len(c) >= 3][:6]
    return {"gercek": {s: by[s] for s in spk}}


def dialog(clips_by_voice, voices, rng, n, gap):
    parts, truth, t = [], [], 0.0
    for i in range(n):
        k = i % len(voices) if len(voices) <= 2 else int(rng.integers(len(voices)))
        pool = clips_by_voice[voices[k]]
        x = pool[int(rng.integers(len(pool)))]
        g = rng.uniform(*gap)
        parts += [x, np.zeros(int(g * 16000), np.float32)]
        truth.append((t, t + len(x) / 16000, k))
        t += len(x) / 16000 + int(g * 16000) / 16000
    return np.concatenate(parts), truth


def sentence_accuracy(truth, segs):
    pred = []
    for s0, s1, _ in truth:
        ov = defaultdict(float)
        for s in segs:
            ov[s.speaker] += max(0.0, min(s1, s.end) - max(s0, s.start))
        pred.append(max(ov, key=ov.get) if ov else "?")
    tl = [t[2] for t in truth]
    P = sorted(set(pred))
    K = max(len(P), max(tl) + 1)
    best = max(sum(1 for p, t in zip(pred, tl) if dict(zip(P, perm)).get(p) == t)
               for perm in itertools.permutations(range(K), len(P)))
    return best / len(tl)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kaynak", default="tts", choices=["tts", "tsv"])
    p.add_argument("tsv", nargs="?")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--diarizer", default="cluster", choices=["cluster", "pyannote"])
    p.add_argument("--bosluksuz", action="store_true", help="sıra değişiminde duraksama yok")
    p.add_argument("--tohum", type=int, default=0)
    a = p.parse_args()

    md = Path(a.models_dir)
    data = tts_clips(md) if a.kaynak == "tts" else tsv_clips(Path(a.tsv))
    cfg = default_config(md, diarizer=a.diarizer)
    diar = build_diarizer(cfg.diarizer)
    vad = SileroVad(VadConfig(model=str(md / "silero_vad.onnx")))
    rng = np.random.default_rng(a.tohum)
    gap = (0.0, 0.05) if a.bosluksuz else (0.15, 0.8)

    ok = total = 0
    accs, sec, dt = [], 0.0, 0.0
    for lang, clips in data.items():
        voices = list(clips)
        for k in (1, 2, 3):
            for combo in itertools.combinations(voices, k):
                audio, truth = dialog(clips, list(combo), rng, 12, gap)
                t0 = time.perf_counter()
                segs = diar.diarize(audio, vad.split(audio))
                dt += time.perf_counter() - t0
                sec += len(audio) / 16000
                n = len({s.speaker for s in segs})
                gercek = len({t[2] for t in truth})
                acc = sentence_accuracy(truth, segs) if segs else 0.0
                ok += n == gercek
                total += 1
                accs.append(acc)
                print(f"  {lang} {k} kişi: bulunan {n}, cümle doğruluğu %{acc * 100:.0f}")
    print(f"\n{a.diarizer}{' (duraksamasız)' if a.bosluksuz else ''}: konuşmacı sayısı doğru "
          f"{ok}/{total}, cümle doğruluğu ort. %{np.mean(accs) * 100:.1f}, RTF {dt / sec:.3f}")


if __name__ == "__main__":
    main()
