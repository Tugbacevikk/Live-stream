# Ölçümler

> 15 Ekim 2026'dan önceki ölçümler Vietnamese-speech-to-text-models reposunda yapıldı.
> Konuşmacı ayırma ayarları (CAM++, eşik 0,70) Vietnamca seslerle seçildi; Türkçe doğru
> metinli ölçüm henüz yok (yol haritası, Faz 4).

Her sayı bir betikten gelir; betik adı ve veri yanında yazılıdır. Donanım: 2 çekirdekli
sanal makine (Xeon), ONNX_THREADS=2, yalnızca CPU.

## VietMed örneği — 50 dosya, 77,2 dk (12 Ekim 2026)

Kullanıcının `vietmed_1_2dk` klasöründen rastgele 50 dosya (seed 42), 8 kHz, 1–2 dk'lık
birleştirilmiş kayıtlar. Roller: Doctor 33, Host 6, Narrator 6, Podcaster 3, Lecturer 2
(hasta kaydı çıkmadı). **Uyarı:** 626 dosyalık küme VietMed train/dev/test bölümlerini
karıştırıyor; Zipformer-30M VietMed ile eğitildiği için onun sonucu iyimser olabilir.

### ASR — `benchmarks/yerel_dogruluk_testi.py`

| Parçalama | Model | WER (hece) | CER | Eksik hece | RTF |
|---|---|---|---|---|---|
| VAD (Silero, eşik 0,5) — eski | 68M | %20,8 | %18,2 | %13,1 | 0,025 |
| VAD eşik 0,2 | 68M | %15,6 | – | %7,3 | 0,031 |
| VAD yok, sabit 15 sn | 68M | %10,8 | – | %1,9 | 0,025 |
| **Tüm ses, en sessiz anda kesim (15 sn)** | **68M** | **%10,1** | **%7,3** | %1,8 | 0,026 |
| Tüm ses, en sessiz anda kesim (15 sn) | 30M | %9,7 | %7,6 | %1,6 | 0,024 |

Bulgu: VAD müzik altındaki / gürültülü konuşmayı atıyordu (en kötü dosyada 95 sn'nin
yalnızca 3,8 sn'si ASR'a gitti). Zipformer sessizlik ve müzikte metin uydurmuyor
(eklenen hece +%0,5), bu yüzden ona artık sesin tamamı veriliyor.
30M ile 68M arasındaki 0,4 puan fark, 30M'in VietMed'i eğitimde görmüş olmasıyla
açıklanabilir; seçim tarafsız VIVOS testine bırakıldı.

Bellek: parçaları hepsini birden çözmek 92 sn'lik seste 1,7 GB'a, 77 dk'lık testte 6 GB'a
(süreç öldürüldü) çıkıyordu. 10 sn'lik gruplarla 460 MB ve daha hızlı (RTF 0,029; 60 sn
gruplarda 2,2 GB, RTF 0,043).

### Dil tespiti — `benchmarks/dil_tespiti_testi.py`

50/50 doğru (vi), 1 dosya Whisper-small'a devredildi, dosya başına 0,29 sn.

### Konuşmacı ayrıştırma — tek kişilik dosyalarda fazladan kişi

Her dosya tek konuşmacının birleştirilmiş kayıtları; doğru cevap çoğunlukla 1.

| Küçük küme kuralı | Gerçek: 1 kişi bulunan | Sentetik 28 diyalog: doğru sayı |
|---|---|---|
| en az 1 sn | 34/50 | 27/28 |
| en az 1,5 sn ve konuşmanın %8'i | **48/50** | **28/28** |

## VietMed hasta örneği — 35 dosya, 54,1 dk (13 Ekim 2026)

15 hasta, 13 doktor, 7 sunucu kaydı; VietMed_011 görüşmesinde sunucu, doktor ve hasta aynı kayıtta.
Hastaların çoğu telefonla bağlanıyor (8 kHz, hat gürültüsü).

### ASR

| Model | Genel WER | Doktor | Sunucu | Hasta | Hasta (duraksama sesleri çıkarılınca) |
|---|---|---|---|---|---|
| 68M | %10,5 | %6,8 | %7,0 | %17,3 | %14,7 |
| 30M | %10,0 | %7,3 | %6,7 | %15,5 | – |

Hasta hatalarının bir kısmı modelin "ờ", "ơ", "à" gibi duraksama seslerini yazmasından:
referansta hastaların bunları yazılmamış (en sık eklenen hece "ờ", 100 kez). Bunlar
çıkarılınca da hasta %14,7: kendiliğinden konuşma ve yöresel ağız gerçekten daha zor.

### Dil tespiti

35/35 doğru, 1 dosya Whisper-small'a devredildi.

### Konuşmacı ayrıştırma — gerçek seslerden kurulan diyaloglar

`konusmaci_testi.py --kaynak tsv`: 4 gerçek konuşmacının (sunucu, doktor, iki hasta)
8 sn'lik kesitlerinden 1-3 kişilik diyaloglar, duraksamalı ve duraksamasız, 2 tohum = 56 diyalog.

| İmza modeli | Eşik | Sayı doğru | Cümle doğru kişide | Sayı verilirse cümle | RTF |
|---|---|---|---|---|---|
| ERes2Net base (önceki) | 0,65 | 16/56 | %80,8 | %86,2 | 0,049 |
| ERes2Net base | 0,75 | 40/56 | %85,6 | %86,2 | 0,049 |
| ERes2NetV2 | 0,70 | 40/56 | %87,1 | %87,2 | 0,103 |
| TitaNet large | 0,80 | 38/56 | %85,0 | %84,5 | 0,055 |
| **CAM++ common** | **0,70** | **47/56** | **%91,7** | %89,9 | **0,030** |

CAM++ 0,70 ile tek kişilik dosyalarda: hasta 10/15, diğer roller 68/70 tek kişi buldu.
Sentetik (temiz TTS) seste 22/28 (eşik 0,65'te 25/28): ayar gerçek sese göre yapıldı.
**Sınır:** veri küçük (4 konuşmacı); eşik 0,65/0,70/0,75 arasında sonuç çok değişiyor,
daha fazla gerçek diyalogla yeniden ayarlanmalı. Hasta seslerinde tek kişiyi iki kişi
sanma hâlâ var (5/15).

## Sentetik diyaloglar — `benchmarks/konusmaci_testi.py`

Piper sesleri (vi ve tr, her dilde 3 ses), 1–3 kişi, 12 konuşma sırası.

| Yöntem | Sayı doğru | Cümle doğru kişide | RTF |
|---|---|---|---|
| pyannote seg-3.0 + WeSpeaker (eski) | 8/14 | %81,0 | 0,121 |
| parça imzası + kümeleme (yeni) | 14/14 | %100 | 0,037 |
| aynısı, sıra değişiminde duraksama yok | 14/14 | %100 | 0,046 |

## Türkçe kayıtlar — 22 dosya, 34 dk (13 Ekim 2026)

Kullanıcının gönderdiği set: 16 podcast/diyalog parçası (1,7–2,2 dk, 16 kHz), 1 müşteri
hizmetleri diyaloğu (1 dk), 2 telefon kaydı (8 kHz, 17 sn ve 63 sn), 3 adet 1 sn'lik test sesi
(saf ton). **Doğru metin yok**: doğruluk yalnızca modellerin birbirine göre farkıyla ve gözle
değerlendirildi. 2 çekirdekli bulut makinesi; dil verilmedi (sesten bulundu).

| | Omnilingual-300M | Whisper-turbo | faster-whisper small (proje, önceki sistem) |
|---|---|---|---|
| Hız, ~2 dk'lık dosya (oran) | 0,26–0,29 | 0,58–0,83 | 0,5–0,7 (kendi bilgisayarında, önceki hatla) |
| Hız, 17 sn telefon kaydı | 0,96 | 1,14 | 0,70 |
| Dil tespiti | 19/19 tr | 4/4 tr | — |
| 1 sn test sesi | düzeltmeden önce "ừ" uydurdu; düzeltmeden sonra boş | boş | "Altyazı M.K." uydurdu |

- **Kısa ses:** 17 sn'lik telefon kaydında sürenin 5 sn'si dil tespiti. Whisper-tiny 8 kHz
  telefon sesinde Türkçeye %36 verdi (desteklenen dillerin toplamı %50'nin altında), ikinci
  kademe Whisper-small %98 ile buldu. Dil istekte verilirse bu süre sıfır.
- **Uydurma:** VAD hiç konuşma bulmazsa artık ASR çalışmıyor (önceden Zipformer/Omnilingual'e
  sesin tamamı veriliyordu).
- **Kaçırılan konuşma:** önceki hat (faster-whisper small) bir podcast parçasında 6–30. sn
  arasını tamamen atladı (181 kelime; Omnilingual 230, turbo 236).
- **Modeller arası fark** (turbo'ya göre WER; turbo doğru kabul edilmez, yalnızca yön gösterir):
  podcast Omnilingual %20 / small %26; müşteri hizmetleri Omnilingual %29 / small %18;
  telefon kayıtlarında üçü de birbirinden %59–78 farklı → 8 kHz telefon sesinde hiçbirine
  güvenilemez, doğru metinle ölçülmeli.
- **Konuşmacı:** 18 dosyada 2 kişi bulundu (önceki hat aynı dosyalarda 1–4 arası). Müşteri
  hizmetleri diyaloğunda iki ses çok benzer (merkez uzaklığı 0,345, birleştirme eşiği 0,35):
  tek kişi çıktı; `num_speakers=2` verilince cümleler doğru ayrıldı. Eşiği 0,34'e indirmek bu
  dosyayı düzeltiyor ama 17 sn'lik telefon kaydını 6 kişiye bölüyor; değiştirilmedi.

## Kelime güveni — VietMed 8 dosya, 2808 hece (14 Ekim 2026)

Zipformer parça olasılıklarından kelime güveni (exp(log olasılık ortalaması)), doğru metinle
hizalanarak:

| eşik | işaretlenen | işaretlilerde hata | diğerlerinde hata | yakalanan hata |
|---|---|---|---|---|
| < 0,3 | %0,7 | %42,9 | %11,9 | %3 |
| < 0,5 (seçilen) | %3,0 | %36,5 | %11,3 | %9 |
| < 0,7 | %17,0 | %21,8 | %10,1 | %31 |

Model aşırı emin (transducer'larda bilinen durum): güven hatayı tek başına bulmaz ama
işaretli kelime 3 kat daha sık yanlış. Arayüzde 0,5 altı altı çizili gösterilir; düzeltme
yapan kişi önce onlara bakar. Genel hata (%12,1) burada kapsama ölçüsünden bağımsızdır.

## Başka alfabeden uydurma — Türkçe 22 dosya (14 Ekim 2026)

Omnilingual (1600 dil) 3174 kelimenin 5'inde başka alfabe kullandı: podcast sonundaki müzikte
Devanagari "कजतच", Arapça tek harfler ve Kiril harf karışmış "nasыsыn" (nasılsın). Artık
(normalizer_for): hiç Latin harf içermeyen kelime atılır; karışık kelimede Latin'e benzeyen
Kiril harfler düzeltilir ("nasısın"), kelime kaybolmaz.

## Araya sıkışmış kısa konuşmacı düzeltmesi — 9 Vietnamca diyalog (14 Ekim 2026)

A, kısa B, A dizisinde B'yi A'ya vermek (fine-tuning projesindeki "sandwich" düzeltmesi):

| eşik | DER | cpWER |
|---|---|---|
| kapalı | %10,64 | %32,01 |
| 1,0 sn | %10,44 | %31,82 |
| 1,5 sn | %10,14 | %31,54 |
| 2,0 sn | %10,07 | %31,35 |

Hiçbir dosya kötüleşmedi, ama **kapalı bırakıldı**: diyaloglar kesitlerden kurulduğu için
gerçek kısa araya girmeler ("ừ", "evet") yok; düzeltme onları siler ve bu sette zararı
görünmez. Kesintisiz gerçek görüşme kaydıyla ölçülünce `FUSION_SANDWICH_MAX` ile açılabilir.

## Henüz yapılmadı

- VIVOS test (tarafsız) — `benchmarks/dogruluk_testi.py`; bu ortamdan Hugging Face'e erişim yok.
- Türkçe doğru metinli ölçüm (FLEURS tr ya da elle yazılmış metin) — `benchmarks/turkce_dogruluk_testi.py`, `yerel_dogruluk_testi.py --dil tr`.
- Gerçek, kesintisiz çekilmiş doktor-hasta görüşmesi (şimdiki diyaloglar kesitlerden kuruldu;
  üst üste konuşma yok).

## Kişi sayısı tahmini: silhouette (fine-tuning projesi) ve NME-SC — 14 Ekim 2026

Aynı CAM++ gömmeleri (önbellek), yalnızca kişi sayısını bulan adım değişiyor. Doğru tahmin sayısı:

| Set | Silhouette (proje) | NME-SC (burada) |
|---|---|---|
| Tek kişilik kayıt | 34/84 | 72/84 |
| Gerçek diyalog | 6/9 | 7/9 |
| Kesit | 11/56 | 34/56 |
| Sentetik diyalog | 14/28 | 26/28 |

Silhouette tek kişiyi hiç bulamaz (en az 2 küme ister); projedeki eşikle de 1'e düşmesi zor.
Ayrıca köşegeni sıfırlanmamış uzaklık matrisinde sklearn hata veriyor. Alınmadı.

"Küçük konuşmacıyı koru" (SpeakerMerger): hasta setinde 7 → 6. Alınmadı, ayar olarak kaldı.

## Türkçe hız ve 4000 mesaj — fine-tuning projesindeki ölçümler (Tuğba'nın bilgisayarı)

Kaynak: Tugbacevikk/fine-tuning README ve docs/benchmarks (gerçek Türkçe görüşmeler, 4 iş
parçacığı, GPU yok). Bu sistemde yeniden ölçülmedi; proje sahibinin onayıyla kabul edildi.

| Model | Medyan oran | En kötü oran | 1 dk ses |
|---|---|---|---|
| faster-whisper medium | 0,94 | 0,98 | ~56–58 sn |
| faster-whisper small | 0,23 | 0,28 | ~14–16 sn |

- Medium hedefin (≤ 1) içinde ama sınıra yakın. O ölçümde sıcaklık yedeği her bölümde
  çalışıyordu; burada yalnızca boşluk kurtarmada çalıştığı için oran aynı ya da daha düşük olmalı.
- 4000 mesaj: sahte modellerle 4 eşzamanlı işçi, çift işleme yok (test_worker_accuracy.py).
  Süre tablosu small modelin oranından hesaplanmış (10 sn'lik 4000 mesaj, 4 işçi ≈ 22 dk);
  medium ile aynı yük yaklaşık 4 kat sürer. `/metrics` kuyrukta bekleme süresini gösterir.
