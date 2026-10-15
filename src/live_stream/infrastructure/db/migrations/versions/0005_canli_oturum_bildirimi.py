"""Outbox'ta işe bağlı olmayan olaylar (canlı oturum sonu bildirimi: live.ended).

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-15
"""
import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("outbox") as b:
        b.alter_column("job_id", existing_type=sa.Text(), nullable=True)


def downgrade() -> None:
    op.execute("DELETE FROM outbox WHERE job_id IS NULL")
    with op.batch_alter_table("outbox") as b:
        b.alter_column("job_id", existing_type=sa.Text(), nullable=False)
