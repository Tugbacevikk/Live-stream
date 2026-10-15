# 0004 — Dil seçimi ve dil başına model

**Tarih:** 13 Ekim 2026 · **Durum:** Kabul

## Bağlam
Sistem Türkçe ve Vietnamca sesleri işler. Dil istekte gelebilir ya da gelmeyebilir.
Fine-tuning projesinin Türkçe modeli faster-whisper (projede small, 13 Ekim'de medium kararı);
Vietnamca için Zipformer-68M
(Apache-2.0, ince ayar serbest). Hedef: 1 dk ses en geç 1 dk'da biter (CPU).

## Karar
- Dil istekte varsa (`tr`, `vi`) tespit atlanır; yoksa Whisper-tiny → small kademeli tespit,
  yalnızca desteklenen diller arasında. Desteklenen dillerin toplam olasılığı < 0,5 ise ret
  (`job.rejected`), güven < 0,8 ise sonuçta "emin değil" işareti.
- Her dilin ayrı kuyruğu ve ayrı ASR motoru (dil profili). Yeni dil = yeni profil.
- Vietnamca: Zipformer-68M; sesin tamamı en sessiz anlardan bölünür (VAD konuşmayı atıyordu).
- Türkçe: faster-whisper **medium** (kullanıcı kararı, 13 Ekim; kelime zamanlı), yalnızca VAD
  parçaları (Whisper sessizlikte metin uyduruyor). Hız bu makinede ölçülemedi (Hugging Face
  erişimi yok); `benchmarks/kabul_testi.py --dil tr` oranı 1'i aşarsa sırayla
  `TR_MODEL=faster-whisper-small`, `TR_MODEL=omnilingual-300m`. Medium iş başına ~1,5–2 GB
  bellek ister: 16 GB'lık bilgisayarda eşzamanlı Türkçe iş sayısı belleğe göre de sınırlanmalı.

## Sonuçlar
Türkçe model değişse de arayüz (Word listesi) aynı kalır; birleştirme ve konuşmacı ayırma
dile bağlı değildir.
