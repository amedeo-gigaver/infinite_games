"""Events deleted_at

Revision ID: 057a9fe8f6af
Revises: 90f3ec827a37
Create Date: 2025-05-30 20:21:22.050054

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "057a9fe8f6af"
down_revision: Union[str, None] = "90f3ec827a37"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
            ALTER TABLE events ADD COLUMN deleted_at DATETIME
        """
    )
