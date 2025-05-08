"""Reasonings

Revision ID: 311e40121462
Revises: 19b6be55ae16
Create Date: 2025-05-02 09:48:51.098041

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "311e40121462"
down_revision: Union[str, None] = "19b6be55ae16"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
            CREATE TABLE IF NOT EXISTS reasoning (
                event_id TEXT NOT NULL,
                miner_uid INTEGER NOT NULL,
                miner_hotkey TEXT NOT NULL,
                reasoning TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                exported BOOLEAN NOT NULL DEFAULT false,
                PRIMARY KEY (event_id, miner_uid, miner_hotkey)
            )
        """
    )
