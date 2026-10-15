# 0001 — Kuyruk: Celery yerine Redis Streams

**Tarih:** 12 Ekim 2026 · **Durum:** Kabul

## Bağlam
Fine-tuning projesi Celery + Redis kullanıyordu. Sistem şimdilik Windows bir bilgisayarda
çalışacak; Celery 4'ten beri Windows'u desteklemiyor (prefork havuzu çalışmıyor, `solo`
havuzuyla tek iş). Ani yük ~4000 mesaj; işler 1 dk'lık sesler, her biri CPU'yu dakikalarca değil
saniyelerce tutuyor. Gerekenler: dile göre ayrı kuyruklar, çöken işçinin işini başkasının
devralması, sınırlı tekrar deneme, ölü mektup kutusu.

## Karar
Redis Streams tüketici grupları (`XREADGROUP`, `XACK`, `XAUTOCLAIM`). Kuyruklar `asr:lid`,
`asr:vi`, `asr:tr`; çalışan iş 30 sn'de bir `XCLAIM` ile "hâlâ bende" der (kalp atışı);
2 dk sessiz kalan işi başka işçi alır; 3 denemeden sonra `asr:dlq` ve `job.failed` bildirimi.
Tek bilgisayar için ayrıca yerel mod: Redis/PostgreSQL olmadan iş parçacığı kuyruğu + SQLite.

## Sonuçlar
- Windows'ta çok süreçli çalışır (`multiprocessing` spawn), ek kütüphane yok (redis-py).
- Celery'nin zamanlayıcı, sonuç deposu gibi özellikleri yok; gerekmiyor (sonuç PostgreSQL'de,
  bakım ayrı süreç).
- Kuyruk mesajı yalnızca `job_id` taşır; durum veritabanındadır (Rust işçisi aynı akışı okuyabilir).
