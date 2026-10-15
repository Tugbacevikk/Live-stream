# Faz 2 — Vietnamca modelin (Zipformer-68M) ince ayarı

Amaç: tıbbi görüşmelerde Vietnamca hata oranını düşürmek. Temel model VietASR'ın 70 bin
saatle eğittiği Zipformer-68M'dir (Apache-2.0, ince ayar serbest). Eğitim **GPU ister** ve
icefall/k2 yalnızca Linux'ta çalıştığı için Windows'ta **WSL2** içinde yapılır. Çıkan model
sistemin kullandığı biçimdedir (int8 ONNX); sistem Windows'ta olduğu gibi çalışır.

## Adımlar

| # | Ne | Nerede | Komut |
|---|----|--------|-------|
| 0 | Ubuntu (bir kez) | PowerShell | `wsl --install -d Ubuntu-22.04` (Kali'ye dokunmaz; içinde `nvidia-smi` çalışmalı) |
| 1 | Ortam (bir kez) | WSL2 Ubuntu | `bash finetune/kurulum.sh ~/ft` |
| 2 | Veri | WSL2 | `python finetune/veri_hazirla.py --vietmed --cikti ~/ft/veri` (ya da arayüzde düzeltilen satırlar: `scripts/duzeltmeleri_disa_aktar.py` çıktısı `--train` olarak) |
| 3 | Eğitim | WSL2 + GPU | `python finetune/egit.py --icefall ~/ft/icefall --veri ~/ft/veri --model ~/ft/temel_model --exp-dir ~/ft/exp --num-epochs 10` |
| 4 | Dışa aktarma | WSL2 | `bash finetune/disari_aktar.sh ~/ft ~/ft/exp 10 3 /mnt/c/.../models/vi-ince-ayar-2026-11` |
| 5 | Kapı 2 (karar) | Windows | `python finetune/kapi2.py --yeni models/vi-ince-ayar-2026-11 --tibbi <test.tsv> --genel <genel.tsv>` |
| 6 | Kullanım | Windows | `.env` içinde `VI_MODEL=models/vi-ince-ayar-2026-11` |

Her oturumda (WSL2): `source ~/ft/venv/bin/activate && export PYTHONPATH=~/ft/icefall:$PYTHONPATH`

## Neye dikkat edildi

- **Aynı sözlük:** temel modelin BPE'si (`Vietnam_bpe_2000_new`) kullanılır, yeniden
  eğitilmez; metin modelin biçimine çevrilir (BÜYÜK HARF, noktalamasız; `ortak.model_text`).
- **Sızıntı yok:** aynı ses ya da aynı kayıt iki bölüme düşerse `veri_hazirla.py` durur.
  VietMed'in kendi train/dev/test ayrımı korunur; Kapı 2'de yalnızca test bölümü kullanılır.
- **Unutma ölçülür:** ince ayar genel konuşmayı bozabilir. Kapı 2 tıbbi olmayan bir set
  (`--genel`, ör. VIVOS test) olmadan karar vermez.
- **Mimari kontrolü:** icefall'ın varsayılan Zipformer ayarları VietASR'ınkiyle aynı; farklı
  olsaydı ağırlık yüklenirken hata verir (sessizce yanlış model eğitilmez).
- **Tekrar üretilebilir:** icefall belirli bir sürüme sabitlenir; model klasörüne
  `MODEL_KARTI.json` (temel, epoch, tarih, icefall sürümü) yazılır.

## Kapı 2 kuralı (`ortak.gate2`)

Yeni model ancak hepsi sağlanırsa kullanılır:
1. Tıbbi testte WER en az **%5 göreli** düşer.
2. Genel konuşmada WER en fazla **0,5 puan** kötüleşir.
3. Hız en fazla **1,2 kat** yavaşlar (aynı mimari; beklenmez).

## Başlangıç ayarları (RTX 3050 Ti Laptop, 4 GB)

- `--max-duration 100` (varsayılan): bir adımda en fazla 100 sn ses. Bellek hatası
  (CUDA out of memory) olursa 70. 12 GB kartta 300 olabilir.
- `--use-fp16 1` (varsayılan): yarım hassasiyet, belleği yaklaşık yarıya indirir.
- Önce küçük deneme: veriyi `--train` ile birkaç saatlik bir TSV'ye sınırlayıp 1 epoch
  çalıştırın; hata yoksa ve süre makulse tüm veriye geçin. Laptop kartı ısınınca yavaşlar;
  şarjda ve iyi havalandırmayla çalıştırın.
- `--base-lr 0.0045` (icefall'ın ince ayar örneği), `--num-epochs 10`. Dev kaybı yükselmeye
  başlarsa erken bırakın; `disari_aktar.sh` son birkaç epoch'un ortalamasını alır (`AVG`).

## Lisans

Temel model Apache-2.0. Eğitim verisinin lisansı ayrıca geçerlidir: VietMed'in lisans
koşullarını (ticari olmayan kullanım vb.) kullanmadan önce veri setinin sayfasından kontrol edin.
Zipformer-30M (CC BY-NC-ND) **ince ayar yapılamaz**; bu yüzden temel 68M'dir.

## Bu ortamda denenmedi

GPU ve Hugging Face erişimi olmadığı için eğitim burada çalıştırılmadı. Denenenler: metin
dönüşümü, sızıntı kontrolü ve Kapı 2 kuralı (birim testleri); Kapı 2 betiği gerçek modelle
(temel modeli "yeni" diye verip aynı sonucu aldığını görerek); icefall'ın `finetune.py` ve
`export-onnx.py` argüman adları sabitlenen sürümle karşılaştırıldı. İlk gerçek çalıştırmada
çıkan hataları not edip bu dosyaya ekleyin.
