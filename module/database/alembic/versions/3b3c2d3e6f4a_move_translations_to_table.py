"""
move translations to dedicated table
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from module.knowledge.type import Language


# revision identifiers, used by Alembic.
revision: str = "3b3c2d3e6f4a"
down_revision: Union[str, None] = "b42d90e32dfd"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "translations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.Column("language", sa.Enum("ENGLISH", "VIETNAMESE", "JAPANESE", "KOREAN", "CHINESE", "FRENCH", name="language", create_type=False, native_enum=False), nullable=False),
        sa.Column("transcript_line_id", sa.Integer(), nullable=False),
        sa.Column("translated", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["transcript_line_id"],
            ["transcript_lines.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_translations_transcript_line_id",
        "translations",
        ["transcript_line_id"],
    )

    op.execute(
        """
        INSERT INTO translations (language, transcript_line_id, translated, created_at, updated_at)
        SELECT 'vietnamese', id, translation, COALESCE(updated_at, NOW()), COALESCE(updated_at, NOW())
        FROM transcript_lines
        WHERE translation IS NOT NULL
        """
    )

    op.drop_column("transcript_lines", "translation")


def downgrade() -> None:
    op.add_column(
        "transcript_lines",
        sa.Column("translation", sa.TEXT(), autoincrement=False, nullable=True),
    )

    op.execute(
        """
        UPDATE transcript_lines tl
        SET translation = sub.translated
        FROM (
            SELECT DISTINCT ON (transcript_line_id) transcript_line_id, translated
            FROM translations
            ORDER BY transcript_line_id, updated_at DESC NULLS LAST
        ) AS sub
        WHERE tl.id = sub.transcript_line_id
        """
    )

    op.drop_index("ix_translations_transcript_line_id", table_name="translations")
    op.drop_table("translations")

