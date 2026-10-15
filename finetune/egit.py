"""Zipformer-68M ince ayarı: icefall'ın zipformer/finetune.py'si, kendi verimizle.

    python finetune/egit.py --icefall ~/ft/icefall --veri ~/ft/veri --model ~/ft/temel_model \\
        --exp-dir ~/ft/exp --num-epochs 10

icefall'ın betiği GigaSpeech/LibriSpeech kesitlerini okuyor; dosyasını değiştirmek yerine
veri modülünün üç yöntemi burada bizim kesitlerimize yönlendirilir (icefall güncellense de
yama yok). Mimari ayarları Zipformer-68M ile aynı (icefall varsayılanı = VietASR ayarı);
farklı olursa ağırlık yüklenirken hata verir, sessizce yanlış model eğitilmez.

--max-duration = bir adımda işlenen toplam ses (sn); GPU belleğini bu belirler.
Varsayılan 100: 4 GB'lık kart (RTX 3050 Ti Laptop) için. Bellek hatası (CUDA out of memory)
alırsanız 70'e indirin; 12 GB kartta 300 olabilir. --base-lr 0.0045 (icefall ince ayar
örneği). Her epoch sonunda dev kaybı yazılır; yükselmeye başlarsa erken bırakın (aşırı uyum).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--icefall", required=True, help="icefall deposunun yolu")
    p.add_argument("--veri", required=True, help="veri_hazirla.py çıktısı")
    p.add_argument("--model", required=True, help="kurulum.sh'in indirdiği temel model klasörü")
    a, rest = p.parse_known_args()

    recipe = Path(a.icefall) / "egs/librispeech/ASR/zipformer"
    sys.path[:0] = [str(recipe), str(Path(a.icefall))]
    from lhotse import CutSet, load_manifest_lazy

    import asr_datamodule

    veri = Path(a.veri)
    dm = asr_datamodule.LibriSpeechAsrDataModule
    dm.gigaspeech_subset_small_cuts = lambda self: load_manifest_lazy(veri / "cuts_train.jsonl.gz")
    dm.dev_clean_cuts = lambda self: load_manifest_lazy(veri / "cuts_dev.jsonl.gz")
    dm.dev_other_cuts = lambda self: CutSet.from_cuts([])
    dm.gigaspeech_dev_cuts = lambda self: load_manifest_lazy(veri / "cuts_dev.jsonl.gz")

    model = Path(a.model)
    ckpt = next((model / "exp" / n for n in ("epoch-12-avg-8.pt", "pretrained.pt", "epoch-12.pt")
                 if (model / "exp" / n).exists()), None)
    if ckpt is None:
        sys.exit(f"{model}/exp içinde başlangıç ağırlığı yok: kurulum.sh çalıştırın")
    defaults = ["--do-finetune", "1", "--use-mux", "0", "--finetune-ckpt", str(ckpt),
                "--bpe-model", str(model / "data/Vietnam_bpe_2000_new/bpe.model"),
                "--manifest-dir", str(veri), "--enable-musan", "0", "--causal", "0",
                "--base-lr", "0.0045", "--use-fp16", "1", "--world-size", "1", "--start-epoch", "1",
                "--max-duration", "100"]
    given = {x for x in rest if x.startswith("--")}
    args = [x for i in range(0, len(defaults), 2) if defaults[i] not in given
            for x in defaults[i:i + 2]] + rest
    print("finetune.py " + " ".join(args))
    sys.argv = [str(recipe / "finetune.py"), *args]
    import finetune

    finetune.main()


if __name__ == "__main__":
    main()
