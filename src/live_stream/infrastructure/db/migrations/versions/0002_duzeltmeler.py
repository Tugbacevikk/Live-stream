"""Arayüzden yapılan düzeltmeler: modelin ilk metni/konuşmacısı ve düzeltme zamanı.

Düzeltilen satırlar ince ayar verisi olur (scripts/duzeltmeleri_disa_aktar.py).

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-14
"""
import sqlalchemy as sa
from alembic import op

from live_stream.infrastructure.db.models import UtcDateTime

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("utterances") as b:
        b.add_column(sa.Column("original_text", sa.Text(), nullable=True))
        b.add_column(sa.Column("original_speaker", sa.Text(), nullable=True))
        b.add_column(sa.Column("edited_at", UtcDateTime(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("utterances") as b:
        b.drop_column("edited_at")
        b.drop_column("original_speaker")
        b.drop_column("original_text")
