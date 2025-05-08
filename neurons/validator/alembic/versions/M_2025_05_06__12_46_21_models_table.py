"""models_table

Revision ID: 2bfc5096631d
Revises: 311e40121462
Create Date: 2025-05-06 12:46:21.512417

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "2bfc5096631d"
down_revision: Union[str, None] = "311e40121462"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
            CREATE TABLE IF NOT EXISTS models (
                name TEXT NOT NULL,
                model_blob TEXT NOT NULL,
                other_data JSON,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (name, created_at)
            )
        """
    )
