"""Clean old events

Revision ID: 19b6be55ae16
Revises: 7f56835fca6a
Create Date: 2025-03-12 22:55:11.379227

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "19b6be55ae16"
down_revision: Union[str, None] = "7f56835fca6a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
            DELETE FROM
                events
            WHERE
                status = 2
                AND processed = FALSE
                AND datetime(cutoff) < datetime(CURRENT_TIMESTAMP, '-21 days')
        """
    )
