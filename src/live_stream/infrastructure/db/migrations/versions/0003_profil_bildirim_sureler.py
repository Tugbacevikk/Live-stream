"""İşlem profili, iş başına bildirim olayları, aşama süreleri, satır silme.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-14
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from live_stream.infrastructure.db.models import UtcDateTime

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("jobs") as b:
        b.add_column(sa.Column("profile", sa.Text(), nullable=True))
        b.add_column(sa.Column("notify", sa.Text(), nullable=True))
        b.add_column(sa.Column("timings", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
                               nullable=True))
    with op.batch_alter_table("utterances") as b:
        b.add_column(sa.Column("deleted_at", UtcDateTime(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("utterances") as b:
        b.drop_column("deleted_at")
    with op.batch_alter_table("jobs") as b:
        b.drop_column("timings")
        b.drop_column("notify")
        b.drop_column("profile")
