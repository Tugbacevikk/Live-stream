"""Zipformer kabul testi: Vietnamca ASR doğruluk + hız (CPU).

İki Zipformer'ı iki veri setinde ölçer:
  - VIVOS test  : iki modelin de eğitim verisinde OLMAYAN tarafsız set -> seçim buna göre yapılır
  - VietMed dev : tıbbi alan, doktor/hasta ayrımıyla -> alan performansı
Zipformer-30M, VietMed ve FLEURS ile eğitildiği için VietMed sonucu iyimser çıkar;
seçimde VIVOS sonucu esas alınır.

Kurulum (repo kökünde):
    pip install -e ".[bench]"
    python scripts/download_models.py

Çalıştırma:
    python benchmarks/dogruluk_testi.py

Sonuç ekrana yazılır ve sonuc_dogruluk.json dosyasına kaydedilir.
"""
import json, os, re, time, unicodedata
import numpy as np
import jiwer
import sherpa_onnx
from datasets import load_dataset, concatenate_datasets

VIETMED_SPLIT = "dev"   # model seçimi için; son rapor için "test"
HER_ROL = 100
THREADS = 2
MODEL_DIR = os.environ.get("MODEL_DIR", "models")  # download_models.py buraya indirir

MODELLER = {
    "Zipformer-68M (Apache-2.0)": ("sherpa-onnx-zipformer-vi-int8-2025-04-20",
        "encoder-epoch-12-avg-8.int8.onnx", "decoder-epoch-12-avg-8.onnx", "joiner-epoch-12-avg-8.int8.onnx"),
    "Zipformer-30M (CC BY-NC-ND)": ("sherpa-onnx-zipformer-vi-30M-int8-2026-02-09",
        "encoder.int8.onnx", "decoder.onnx", "joiner.int8.onnx"),
}


def normalize(m):
    m = unicodedata.normalize("NFC", m.lower())
    m = re.sub(r"[^\w\s]", " ", m)
    return " ".join(m.split())


def vietmed():
    veri = load_dataset("leduckhai/VietMed", split=VIETMED_SPLIT)
    secim = []
    for rol in ("Doctor", "Patient"):
        r = veri.filter(lambda x: x["role"] == rol).shuffle(seed=42)
        n = min(HER_ROL, len(r))
        print(f"VietMed {VIETMED_SPLIT}, {rol}: {n} kayıt (mevcut {len(r)})")
        secim.append(r.select(range(n)))
    veri = concatenate_datasets(secim)
    return ([np.asarray(o["audio"]["array"], dtype=np.float32) for o in veri],
            [o["audio"]["sampling_rate"] for o in veri],
            [o["text"] for o in veri],
            [o["role"] for o in veri])


def vivos():
    try:
        veri = load_dataset("AILAB-VNUHCM/vivos", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"VIVOS yüklenemedi, atlanıyor: {type(e).__name__}: {e}")
        return None
    print(f"VIVOS test: {len(veri)} kayıt")
    return ([np.asarray(o["audio"]["array"], dtype=np.float32) for o in veri],
            [o["audio"]["sampling_rate"] for o in veri],
            [o["sentence"] for o in veri],
            ["-"] * len(veri))


def olc(recognizer, sesler, srs, refs, roller):
    t = time.time()
    hyps = []
    for s, sr in zip(sesler, srs):
        st = recognizer.create_stream()
        st.accept_waveform(sr, s)
        recognizer.decode_stream(st)
        hyps.append(normalize(st.result.text))
    sure = time.time() - t
    refs = [normalize(r) for r in refs]

    def wc(idx):
        r = [refs[i] for i in idx]
        h = [hyps[i] for i in idx]
        return {"WER": round(jiwer.wer(r, h) * 100, 2), "CER": round(jiwer.cer(r, h) * 100, 2)}

    tum = list(range(len(refs)))
    out = {"toplam": wc(tum),
           "RTF": round(sure / (sum(len(s) / sr for s, sr in zip(sesler, srs))), 4),
           "ornek": [{"ref": refs[i], "hyp": hyps[i]} for i in tum[:3]]}
    for rol in sorted(set(roller) - {"-"}):
        out[rol] = wc([i for i in tum if roller[i] == rol])
    return out


def main():
    setler = {"VIVOS test (tarafsız)": vivos(), f"VietMed {VIETMED_SPLIT}": vietmed()}
    setler = {k: v for k, v in setler.items() if v is not None}

    sonuc = {}
    for ad, (klasor, enc, dec, joi) in MODELLER.items():
        rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=f"{MODEL_DIR}/{klasor}/{enc}", decoder=f"{MODEL_DIR}/{klasor}/{dec}",
            joiner=f"{MODEL_DIR}/{klasor}/{joi}",
            tokens=f"{MODEL_DIR}/{klasor}/tokens.txt", num_threads=THREADS, decoding_method="greedy_search")
        sonuc[ad] = {}
        print(f"\n{ad}")
        for set_ad, veri in setler.items():
            r = olc(rec, *veri)
            sonuc[ad][set_ad] = r
            satir = f"  {set_ad:22s} WER %{r['toplam']['WER']:6.2f}  CER %{r['toplam']['CER']:6.2f}  RTF {r['RTF']}"
            for rol in ("Doctor", "Patient"):
                if rol in r:
                    satir += f"  | {rol} CER %{r[rol]['CER']}"
            print(satir)

    with open("sonuc_dogruluk.json", "w", encoding="utf-8") as f:
        json.dump({"threads": THREADS, "sonuc": sonuc}, f, ensure_ascii=False, indent=1)
    print("\nKaydedildi: sonuc_dogruluk.json")
    print("Seçim kuralı: VIVOS CER'i düşük olan model; fark 1 puandan azsa daha hızlı olan (30M).")


if __name__ == "__main__":
    main()
