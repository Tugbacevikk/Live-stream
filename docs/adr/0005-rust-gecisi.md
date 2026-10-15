# 0005 — Rust'a kademeli geçiş

**Tarih:** 13 Ekim 2026 · **Durum:** Öneri (ilk adım yapıldı)

## Bağlam
Sistem ileride Rust'a taşınacak. Hepsini bir kerede yeniden yazmak riskli: çalışan ve
ölçülmüş (WER, DER, gecikme) bir hat var; yeniden yazımda sessiz davranış farkları çıkar.

## Karar
Parça parça taşınır. Her parça için aynı kalıp:
1. Python sürümü **başvuru kaynağıdır**; önce saf hale getirilir (girdi/çıktı düz veri,
   genel ayar okuması yok, yan etki yok).
2. Rust sürümü `native/` crate'inde; saf çekirdek PyO3'süz derlenebilir (`rlib`), Python bağı
   yalnızca `python` özelliğiyle.
3. **Eşlik testi**: rastgele binlerce girdide iki sürüm birebir aynı sonucu vermeli
   (`tests/unit/test_native.py`). Eşitlik durumları dahil.
4. Rust modülü kurulu değilse Python sürümü çalışır (`VIET_ASR_NATIVE=0` ile de kapatılır).

Sınırlar Rust'a hazır tutulur: kuyruk mesajı yalnızca `job_id` (0001), şemanın sahibi
Alembic (0002), dış biçim `serialization.py`'deki JSON (API yanıtı = webhook gövdesi).

## Sıra
| Adım | Parça | Neden bu sırada | Durum |
|------|-------|-----------------|-------|
| 1 | Birleştirme (`fusion.assign`) | Saf, küçük, iyi tanımlı: kalıbı kurmak için düşük risk | Yapıldı: 9000 rastgele girdide eşit; 16 bin kelimede 112 ms → 4,5 ms |
| 2a | Enerji bölücü (`energy_cuts`) | Saf, sayısal; numpy dizisi kopyasız okunur | Yapıldı (14 Ekim): 300 rastgele seste eşit; 10 dk seste 70 ms → 30 ms |
| 2b | Parça → kelime (`tokens_to_words`) | Saf; metin işleme | Yapıldı (14 Ekim): gruplama, zaman ve güven Rust'ta, dile özgü yazım düzeltmesi Python'da; 5000 rastgele girdide eşit; 60 parçada 34 µs → 17 µs (dakika başına ~0,1 ms: hız için değil, taşıma için) |
| 3 | Konuşmacı kümeleme (NME-SC) | Özdeğer çözümü gerekir (nalgebra/faer); DER ile doğrulanır | — |
| 4 | İşçi süreci (Redis Streams tüketici + depo) | sherpa-onnx'in C API'si ve `ort` crate'i ile; model çağrıları aynı | — |
| 5 | HTTP API (axum) | En son: dış sözleşme `serialization.py` + OpenAPI | — |

Not: birleştirme darboğaz değildir (zamanın çoğu ASR'da). İlk parça hız için değil, eşlik
testi ve iki dilli derleme düzenini oturtmak için seçildi.

## Sonuçlar
- Windows'ta `pip install ./native` için Rust (rustup) gerekir; kurulmazsa sistem değişmeden çalışır.
- Her taşınan parça CI'da hem `cargo test` hem eşlik testiyle denetlenir.
