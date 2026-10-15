# 0002 — Veritabanı: SQLAlchemy 2.0 + Alembic (code-first)

**Tarih:** 13 Ekim 2026 · **Durum:** Kabul

## Bağlam
Önceki sürümde iki depo vardı: PostgreSQL için elle SQL (`postgres.py`) ve SQLite için ayrı
bir kopya (`sqlite_store.py`). Şema iki yerde tutuluyor, değişiklikler `ALTER ... IF NOT EXISTS`
satırlarıyla yamanıyordu; sürüm takibi yoktu. Fine-tuning projesi SQLAlchemy + Alembic
kullanıyordu (code-first), ama Alembic'in SQLite için toplu değişiklik modu (`render_as_batch`)
açık değildi ve `external_id` yarış durumu hata metni aranarak yakalanıyordu.

## Karar
- Şemanın tek kaynağı `infrastructure/db/models.py`; değişiklik Alembic göçüyle
  (`migrations/versions`, paketin içinde). Uygulama açılışta `upgrade head` çalıştırır.
- Tek depo sınıfı `SqlAlchemyJobStore`; SQLite (yerel mod) ve PostgreSQL (tam mod) aynı kod.
- Zamanlar her iki veritabanında saat dilimli UTC; metinler TEXT; JSON yalnızca outbox gövdesi.
- `external_id` benzersiz indeks: aynı istek tekrar gelirse var olan iş döner; yarış
  `IntegrityError` ile yakalanır.
- Eski (Alembic'siz) veritabanı açılışta tanınır ve veri kaybı olmadan aktarılır.
- `test_sema_modellerle_ayni` ve CI'daki `alembic check`: model değişip göç unutulursa kırılır.

## Sonuçlar
- SQLite tek süreçlidir; çok süreçli tam mod PostgreSQL ister (`stream_worker` SQLite'ı reddeder).
- Rust'a geçişte şemanın sahibi Alembic kalır; Rust tarafı (ör. sqlx) aynı tabloları kullanır.
