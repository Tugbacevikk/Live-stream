"""Bağlantı adresini SQLAlchemy sürücüsüne çevirir (tek yer)."""
from __future__ import annotations

from pathlib import Path


def normalize_url(url: str) -> str:
    """postgresql:// -> postgresql+psycopg:// (psycopg 3); sqlite dosyasının klasörünü açar."""
    if url.startswith(("postgres://", "postgresql://")):
        return "postgresql+psycopg://" + url.split("://", 1)[1]
    if url.startswith("sqlite:///") and not url.startswith("sqlite:///:memory:"):
        Path(url[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
    return url
