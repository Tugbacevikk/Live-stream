"""Alembic ortamı. Göçler paketin içinde: kurulu sistemde de `migrate()` çalışır.

İki kullanım:
  - uygulama: SqlAlchemyJobStore.migrate() bağlantıyı config.attributes["connection"] ile verir
  - geliştirici: `alembic -c alembic.ini revision --autogenerate -m "..."` (DATABASE_URL'den)
"""
from __future__ import annotations

import os

from alembic import context
from sqlalchemy import create_engine, pool

from live_stream.infrastructure.db.models import Base
from live_stream.infrastructure.db.url import normalize_url

config = context.config
target_metadata = Base.metadata


def _configure(connection) -> None:
    context.configure(
        connection=connection, target_metadata=target_metadata, compare_type=True,
        # SQLite ALTER TABLE'ı kısıtlı: sütun değişiklikleri tabloyu yeniden kurarak yapılır
        render_as_batch=connection.dialect.name == "sqlite",
    )


def run_online() -> None:
    conn = config.attributes.get("connection")
    if conn is not None:
        _configure(conn)
        with context.begin_transaction():
            context.run_migrations()
        return
    url = normalize_url(config.get_main_option("sqlalchemy.url")
                        or os.environ.get("DATABASE_URL", "sqlite:///data/jobs.db"))
    engine = create_engine(url, poolclass=pool.NullPool)
    with engine.connect() as connection:
        _configure(connection)
        with context.begin_transaction():
            context.run_migrations()


def run_offline() -> None:
    """SQL dosyası üret (veritabanı yöneticisine verilecekse): alembic upgrade head --sql"""
    url = normalize_url(config.get_main_option("sqlalchemy.url")
                        or os.environ.get("DATABASE_URL", "sqlite:///data/jobs.db"))
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_offline()
else:
    run_online()
