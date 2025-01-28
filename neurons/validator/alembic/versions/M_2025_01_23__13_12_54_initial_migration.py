"""Initial migration

Revision ID: a0ebd4593388
Revises:
Create Date: 2025-01-23 13:12:54.963622

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a0ebd4593388"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
            CREATE TABLE IF NOT EXISTS events (
                unique_event_id PRIMARY KEY,
                event_id TEXT,
                market_type TEXT,
                registered_date DATETIME,
                description TEXT,
                starts DATETIME,
                resolve_date DATETIME,
                outcome TEXT,
                local_updated_at DATETIME,
                status TEXT,
                metadata TEXT,
                processed BOOLEAN DEFAULT false,
                exported INTEGER DEFAULT 0
            )
        """
    )

    op.execute(
        """
            CREATE TABLE IF NOT EXISTS miners (
                miner_hotkey TEXT,
                miner_uid TEXT,
                node_ip TEXT,
                registered_date DATETIME,
                last_updated DATETIME,
                blocktime INTEGER,
                blocklisted boolean DEFAULT false,
                PRIMARY KEY (miner_hotkey, miner_uid)
            )
        """
    )

    op.execute(
        """
            CREATE TABLE IF NOT EXISTS predictions (
                unique_event_id TEXT,
                minerHotkey TEXT,
                minerUid TEXT,
                predictedOutcome TEXT,
                canOverwrite BOOLEAN,
                outcome TEXT,
                interval_start_minutes INTEGER,
                interval_agg_prediction REAL,
                interval_count INTEGER,
                submitted DATETIME,
                blocktime INTEGER,
                exported INTEGER DEFAULT 0,
                PRIMARY KEY (unique_event_id, interval_start_minutes, minerUid)
            )
        """
    )
