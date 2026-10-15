# 0003 — Konuşmacı ayırma: CAM++ imzası + spektral kümeleme

**Tarih:** 13 Ekim 2026 · **Durum:** Kabul

## Bağlam
Mesajlar her zaman iki kişilik değil; kişi sayısı bilinmiyor (istekte isteğe bağlı
`num_speakers`). pyannote segmentation-3.0, kısa sıralarla konuşan Vietnamca sesleri
ayıramadı. Sabit eşikli birleştirme (AHC), aynı kayıttaki iki kişiyi (uzaklık 0,38–0,58) ve
telefondaki tek kişinin kendi içindeki farkı (~0,3) aynı anda ayıramıyordu.

## Karar
VAD parçaları → 1,5 sn pencere / 0,75 sn kayma → CAM++ imzası → NME-SC spektral kümeleme
(kişi sayısını özdeğer boşluğundan bulur) → merkezleri 0,35'ten yakın kümeleri birleştir →
konuşmanın %8'inden (en az 1,5 sn) azını kaplayan "kişi" en yakın kişiye katılır.
Sayı istekte verilirse doğrudan o kadar küme, birleştirme ve en az pay kuralı uygulanmaz.

## Ölçüm (VietMed'den 9 gerçek görüşme diyaloğu, tüm hat)
DER %26,8 → %10,2 (sayı otomatik), sayı verilince %19 → %5,2. Ayrıntı: docs/olcumler.md.

## Sonuçlar
Üst üste konuşma ayrılmaz; değişim ~0,75 sn çözünürlükle yakalanır. %8'den az konuşan kişi
otomatik modda ayrı çıkmaz (sayı verilirse çıkar).
