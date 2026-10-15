"""Veritabanı şeması: tek doğru kaynak (code-first).

Tablolar burada Python sınıfı olarak tanımlanır; değişiklik Alembic göçüyle
(migrations/versions) veritabanına uygulanır. Aynı şema hem SQLite (yerel mod)
hem PostgreSQL (tam mod) için geçerlidir.

Rust'a geçiş notu: tablo/sütun adları ve tipleri sözleşmedir. İleride Rust
tarafı (ör. sqlx) bu tabloları aynen okur/yazar; şemanın sahibi Alembic kalır,
yani göç dosyaları dilden bağımsız tek kaynak olur. Bu yüzden:
  - zamanlar her zaman UTC ve saat dilimli saklanır (UtcDateTime),
  - JSON yalnızca outbox.payload'da (gönderilecek gövde) kullanılır,
  - enum'lar veritabanında düz metin (JobStatus.value), CHECK yok: yeni durum
    eklemek göç gerektirmesin,
  - metinler uzunluk sınırsız TEXT (iki veritabanında da aynı davranış).
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON, BigInteger, Boolean, DateTime, Float, ForeignKey, Index, Integer, MetaData, Text,
    TypeDecorator, text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Kısıt adları sabit: Alembic göçleri iki veritabanında da aynı adları üretir.
NAMING = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class UtcDateTime(TypeDecorator):
    """Saat dilimli UTC zaman. SQLite saat dilimini saklamaz; okurken UTC eklenir,
    yazarken UTC'ye çevrilir. Böylece iki veritabanı aynı değeri döndürür."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("saat dilimsiz zaman kaydedilemez (UTC kullanın)")
        return value.astimezone(timezone.utc)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING)


class JobRow(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    status: Mapped[str] = mapped_column(Text)
    audio_path: Mapped[str] = mapped_column(Text)
    requested_lang: Mapped[str | None] = mapped_column(Text)
    callback_url: Mapped[str | None] = mapped_column(Text)
    num_speakers: Mapped[int | None] = mapped_column(Integer)
    # Entegre olan sistemin kendi kimliği: aynı istek iki kez gelirse yeni iş açılmaz.
    external_id: Mapped[str | None] = mapped_column(Text)
    profile: Mapped[str | None] = mapped_column(Text)       # full | feedback (None: varsayılan)
    notify: Mapped[str | None] = mapped_column(Text)        # "done,failed" (None: varsayılan)
    # aşama süreleri {"vad": 0.2, "asr": 4.1, ...}: metrikler bütün süreçlerden buradan toplanır
    timings: Mapped[dict | None] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    # kapsama yüzdesi, yazısız konuşma, kurtarılan kelime, düşük güvenli kelime sayısı
    quality: Mapped[dict | None] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    # veri setine aktarma onayı (kim, ne zaman, hangi amaçla: test | egitim); yoksa aktarılmaz
    consent_by: Mapped[str | None] = mapped_column(Text)
    consent_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    consent_use: Mapped[str | None] = mapped_column(Text)
    lang_code: Mapped[str | None] = mapped_column(Text)
    lang_confidence: Mapped[float | None] = mapped_column(Float)
    lang_source: Mapped[str | None] = mapped_column(Text)
    lang_uncertain: Mapped[bool | None] = mapped_column(Boolean)
    model: Mapped[str | None] = mapped_column(Text)
    duration: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)

    __table_args__ = (
        Index("ix_jobs_status", "status"),
        Index("ix_jobs_updated_at", "updated_at"),  # saklama/bakım sorguları
        Index("uq_jobs_external_id", "external_id", unique=True),
    )


class UtteranceRow(Base):
    __tablename__ = "utterances"

    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True)
    idx: Mapped[int] = mapped_column(Integer, primary_key=True)
    speaker: Mapped[str] = mapped_column(Text)
    start_s: Mapped[float] = mapped_column(Float)
    end_s: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    # Arayüzden düzeltildiyse: modelin ilk yazdığı metin ve düzeltme zamanı (ince ayar verisi)
    original_text: Mapped[str | None] = mapped_column(Text)
    original_speaker: Mapped[str | None] = mapped_column(Text)
    edited_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    conf: Mapped[float | None] = mapped_column(Float)        # kelime güvenlerinin ortalaması
    low_conf: Mapped[str | None] = mapped_column(Text)       # düşük güvenli kelime sıraları "2,5"
    # arayüzden silindi (uydurma satır): sonuçta görünmez, ince ayar için kayıtta kalır
    deleted_at: Mapped[datetime | None] = mapped_column(UtcDateTime)


class OutboxRow(Base):
    """Gönderilecek webhook bildirimleri (sonuçla aynı işlemde yazılır)."""

    __tablename__ = "outbox"

    # PostgreSQL'de BIGINT, SQLite'ta INTEGER (SQLite yalnız INTEGER PRIMARY KEY'i otomatik artırır)
    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True,
                                    autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    event: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"))
    next_attempt_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    sent_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)

    __table_args__ = (
        # yalnızca gönderilmemiş satırlar: gönderici sorgusu küçük bir indeksi tarar
        Index("ix_outbox_job_id", "job_id"),
        Index("ix_outbox_pending", "next_attempt_at",
              postgresql_where=text("sent_at IS NULL"), sqlite_where=text("sent_at IS NULL")),
    )
