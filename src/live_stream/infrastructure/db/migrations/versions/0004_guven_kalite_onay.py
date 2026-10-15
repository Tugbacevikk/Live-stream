"""Satır güveni, iş kalitesi (kapsama), veri setine aktarma onayı.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-14
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from live_stream.infrastructure.db.models import UtcDateTime

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("jobs") as b:
        b.add_column(sa.Column("quality", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
                               nullable=True))
        b.add_column(sa.Column("consent_by", sa.Text(), nullable=True))
        b.add_column(sa.Column("consent_at", UtcDateTime(), nullable=True))
        b.add_column(sa.Column("consent_use", sa.Text(), nullable=True))
    with op.batch_alter_table("utterances") as b:
        b.add_column(sa.Column("conf", sa.Float(), nullable=True))
        b.add_column(sa.Column("low_conf", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("utterances") as b:
        b.drop_column("low_conf")
        b.drop_column("conf")
    with op.batch_alter_table("jobs") as b:
        b.drop_column("consent_use")
        b.drop_column("consent_at")
        b.drop_column("consent_by")
        b.drop_column("quality")
