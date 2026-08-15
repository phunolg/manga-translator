"""

Revision ID: b42d90e32dfd
Revises: 440ebebfb75f
Create Date: 2025-11-26 18:39:51.333184

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b42d90e32dfd'
down_revision: Union[str, None] = '440ebebfb75f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Thêm cột mới ở trạng thái nullable để có thể migrate dữ liệu cũ
    op.add_column('mapping_names', sa.Column('story_id', sa.Integer(), nullable=True))
    op.add_column('mapping_names', sa.Column('source', sa.Text(), nullable=True))
    op.add_column('mapping_names', sa.Column('translation', sa.Text(), nullable=True))

    # Copy dữ liệu từ bảng characters sang các cột mới
    op.execute(
        """
        UPDATE mapping_names mn
        SET story_id = c.story_id,
            source = c.source_name,
            translation = mn.new_name
        FROM characters c
        WHERE mn.character_id = c.id
        """
    )

    # Nếu còn bản ghi orphan (không tìm được story), xóa để tránh vi phạm NOT NULL
    op.execute(
        "DELETE FROM mapping_names WHERE story_id IS NULL OR source IS NULL OR translation IS NULL"
    )

    # Ràng buộc NOT NULL sau khi dữ liệu đã được migrate
    op.alter_column('mapping_names', 'story_id', nullable=False)
    op.alter_column('mapping_names', 'source', nullable=False)
    op.alter_column('mapping_names', 'translation', nullable=False)

    # Cập nhật index & foreign key
    op.drop_index(op.f('ix_mapping_names_character_id'), table_name='mapping_names')
    op.create_index(op.f('ix_mapping_names_story_id'), 'mapping_names', ['story_id'], unique=False)
    op.drop_constraint(op.f('mapping_names_character_id_fkey'), 'mapping_names', type_='foreignkey')
    op.create_foreign_key(None, 'mapping_names', 'stories', ['story_id'], ['id'], ondelete='CASCADE')

    # Loại bỏ các cột cũ
    op.drop_column('mapping_names', 'character_id')
    op.drop_column('mapping_names', 'new_name')


def downgrade() -> None:
    # Thêm lại cột cũ ở trạng thái nullable để copy dữ liệu ngược
    op.add_column('mapping_names', sa.Column('new_name', sa.TEXT(), autoincrement=False, nullable=True))
    op.add_column('mapping_names', sa.Column('character_id', sa.INTEGER(), autoincrement=False, nullable=True))

    # Copy dữ liệu trở lại character_id/new_name dựa trên story_id + source
    op.execute(
        """
        UPDATE mapping_names mn
        SET character_id = c.id,
            new_name = mn.translation
        FROM characters c
        WHERE c.story_id = mn.story_id
          AND c.source_name = mn.source
        """
    )

    # Có thể có bản ghi không tìm được character_id -> xóa để tránh NOT NULL
    op.execute(
        "DELETE FROM mapping_names WHERE character_id IS NULL OR new_name IS NULL"
    )

    # Ràng buộc lại NOT NULL
    op.alter_column('mapping_names', 'new_name', nullable=False)
    op.alter_column('mapping_names', 'character_id', nullable=False)

    # Khôi phục index & foreign key
    op.drop_constraint(None, 'mapping_names', type_='foreignkey')
    op.create_foreign_key(op.f('mapping_names_character_id_fkey'), 'mapping_names', 'characters', ['character_id'], ['id'], ondelete='CASCADE')
    op.drop_index(op.f('ix_mapping_names_story_id'), table_name='mapping_names')
    op.create_index(op.f('ix_mapping_names_character_id'), 'mapping_names', ['character_id'], unique=False)

    # Xóa các cột mới
    op.drop_column('mapping_names', 'translation')
    op.drop_column('mapping_names', 'source')
    op.drop_column('mapping_names', 'story_id')

