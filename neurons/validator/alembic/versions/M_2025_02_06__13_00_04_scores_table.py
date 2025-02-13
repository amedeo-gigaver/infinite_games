"""Scores table

Revision ID: 9a9ffede6f53
Revises: 389ec9305bde
Create Date: 2025-02-04 13:00:04.801019

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9a9ffede6f53"
down_revision: Union[str, None] = "389ec9305bde"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
            CREATE TABLE IF NOT EXISTS scores (
                event_id TEXT NOT NULL,
                miner_uid INTEGER NOT NULL,
                miner_hotkey TEXT NOT NULL,
                prediction REAL NOT NULL,
                event_score REAL NOT NULL,
                metagraph_score REAL,
                other_data JSON,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                spec_version INTEGER NOT NULL,
                processed BOOLEAN NOT NULL DEFAULT false,
                exported BOOLEAN NOT NULL DEFAULT false,
                PRIMARY KEY (event_id, miner_uid, miner_hotkey)
            )
        """
    )
