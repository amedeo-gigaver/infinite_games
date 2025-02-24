"""Vacuum

Revision ID: 7f56835fca6a
Revises: 9a9ffede6f53
Create Date: 2025-02-20 23:16:33.979074

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7f56835fca6a"
down_revision: Union[str, None] = "9a9ffede6f53"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    raw_conn = conn.connection

    raw_conn.executescript(
        """
            PRAGMA auto_vacuum = INCREMENTAL;
            VACUUM;
        """
    )


def downgrade() -> None:
    pass
