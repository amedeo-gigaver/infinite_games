"""Rebuild predictions table

Revision ID: 90f3ec827a37
Revises: 2bfc5096631d
Create Date: 2025-05-12 22:59:45.684589

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "90f3ec827a37"
down_revision: Union[str, None] = "2bfc5096631d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
            PRAGMA foreign_keys = ON
          """
    )

    op.execute(
        """
            PRAGMA cache_size = 1000000
          """
    )

    op.execute(
        """
            DELETE FROM
                predictions
            WHERE
                -- Orphan predictions
                unique_event_id NOT IN (
                    SELECT unique_event_id FROM events
                )

                -- Any nulls
                OR (
                    unique_event_id IS NULL
                    OR minerHotkey IS NULL
                    OR minerUid IS NULL
                    OR predictedOutcome IS NULL
                    OR interval_start_minutes IS NULL
                    OR interval_agg_prediction IS NULL
                    OR interval_count IS NULL
                )
          """
    )

    op.execute(
        """
            ALTER TABLE predictions RENAME TO _predictions_old
          """
    )

    op.execute(
        """
            CREATE TABLE predictions (
                unique_event_id            TEXT NOT NULL,
                miner_uid                  INTEGER NOT NULL,
                miner_hotkey               TEXT NOT NULL,
                latest_prediction          REAL NOT NULL,
                interval_start_minutes     INTEGER NOT NULL,
                interval_agg_prediction    REAL NOT NULL,
                interval_count             INTEGER NOT NULL,
                submitted                  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at                 DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                exported                   BOOLEAN NOT NULL DEFAULT FALSE,
                PRIMARY KEY (unique_event_id, miner_uid, miner_hotkey,  interval_start_minutes),
                FOREIGN KEY (unique_event_id)
                    REFERENCES events(unique_event_id)
                    ON UPDATE CASCADE
                    ON DELETE CASCADE
            )
          """
    )

    op.execute(
        """
            CREATE INDEX idx_predictions_exported ON predictions(exported)
          """
    )

    op.execute(
        """
            INSERT INTO predictions (
                unique_event_id,
                miner_uid,
                miner_hotkey,
                latest_prediction,
                interval_start_minutes,
                interval_agg_prediction,
                interval_count,
                submitted,
                exported
            )
            SELECT
                unique_event_id,
                minerUid,
                minerHotkey,
                predictedOutcome,
                interval_start_minutes,
                interval_agg_prediction,
                interval_count,
                submitted,
                exported
            FROM _predictions_old
          """
    )

    op.execute(
        """
            DROP TABLE _predictions_old
          """
    )
