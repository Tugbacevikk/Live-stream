"""İlk şema (code-first): jobs, utterances, outbox.

Önceki sürümün elle SQL ile kurduğu veritabanları store.migrate() içinde bu şemaya
aktarılır (_adopt_legacy); göç geçmişi temiz başlar.

Revision ID: 0001
Revises:
Create Date: 2026-10-13
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from live_stream.infrastructure.db.models import UtcDateTime

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("audio_path", sa.Text(), nullable=False),
        sa.Column("requested_lang", sa.Text(), nullable=True),
        sa.Column("callback_url", sa.Text(), nullable=True),
        sa.Column("num_speakers", sa.Integer(), nullable=True),
        sa.Column("external_id", sa.Text(), nullable=True),
        sa.Column("lang_code", sa.Text(), nullable=True),
        sa.Column("lang_confidence", sa.Float(), nullable=True),
        sa.Column("lang_source", sa.Text(), nullable=True),
        sa.Column("lang_uncertain", sa.Boolean(), nullable=True),
        sa.Column("model", sa.Text(), nullable=True),
        sa.Column("duration", sa.Float(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", UtcDateTime(), nullable=False),
        sa.Column("updated_at", UtcDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_jobs")),
    )
    op.create_index("ix_jobs_status", "jobs", ["status"])
    op.create_index("ix_jobs_updated_at", "jobs", ["updated_at"])
    op.create_index("uq_jobs_external_id", "jobs", ["external_id"], unique=True)

    op.create_table(
        "utterances",
        sa.Column("job_id", sa.Text(), nullable=False),
        sa.Column("idx", sa.Integer(), nullable=False),
        sa.Column("speaker", sa.Text(), nullable=False),
        sa.Column("start_s", sa.Float(), nullable=False),
        sa.Column("end_s", sa.Float(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], name=op.f("fk_utterances_job_id_jobs"),
                                ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("job_id", "idx", name=op.f("pk_utterances")),
    )

    op.create_table(
        "outbox",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), autoincrement=True,
                  nullable=False),
        sa.Column("job_id", sa.Text(), nullable=False),
        sa.Column("event", sa.Text(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("payload", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=False),
        sa.Column("attempts", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("next_attempt_at", UtcDateTime(), nullable=False),
        sa.Column("sent_at", UtcDateTime(), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_at", UtcDateTime(), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], name=op.f("fk_outbox_job_id_jobs"),
                                ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_outbox")),
    )
    op.create_index("ix_outbox_job_id", "outbox", ["job_id"])
    op.create_index("ix_outbox_pending", "outbox", ["next_attempt_at"],
                    postgresql_where=sa.text("sent_at IS NULL"), sqlite_where=sa.text("sent_at IS NULL"))


def downgrade() -> None:
    op.drop_table("outbox")
    op.drop_table("utterances")
    op.drop_table("jobs")
