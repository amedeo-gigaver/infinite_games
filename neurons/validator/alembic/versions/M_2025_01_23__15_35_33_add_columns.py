"""Add columns

Revision ID: b200ceaa4d5e
Revises: a0ebd4593388
Create Date: 2025-01-23 15:35:33.666692

"""

from typing import Sequence, Union

from alembic import op
from sqlalchemy.sql import text

# revision identifiers, used by Alembic.
revision: str = "b200ceaa4d5e"
down_revision: Union[str, None] = "a0ebd4593388"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    add_column_if_not_exists(
        table_name="events",
        column_name="created_at",
        column_type="DATETIME",
        default_value=None,
    )

    add_column_if_not_exists(
        table_name="events",
        column_name="cutoff",
        column_type="DATETIME",
        default_value=None,
    )

    op.execute(
        """
            UPDATE
                events
            SET
                cutoff = datetime(json_extract(metadata, '$.cutoff'), 'unixepoch')
            WHERE
                cutoff IS NULL
                AND json_extract(metadata, '$.cutoff') IS NOT NULL
        """
    )

    add_column_if_not_exists(
        table_name="events",
        column_name="end_date",
        column_type="DATETIME",
        default_value=None,
    )

    op.execute(
        """
            UPDATE
                events
            SET
                end_date = datetime(json_extract(metadata, '$.end_date'), 'unixepoch')
            WHERE
                end_date IS NULL
                AND json_extract(metadata, '$.end_date') IS NOT NULL
            """
    )

    add_column_if_not_exists(
        table_name="events",
        column_name="resolved_at",
        column_type="DATETIME",
        default_value=None,
    )

    add_column_if_not_exists(
        table_name="events",
        column_name="event_type",
        column_type="TEXT",
        default_value=None,
    )

    op.execute(
        """
            UPDATE
                events
            SET
                event_type = json_extract(metadata, '$.market_type')
            WHERE
                event_type IS NULL
                AND json_extract(metadata, '$.market_type') IS NOT NULL
            """
    )


def add_column_if_not_exists(
    table_name: str,
    column_name: str,
    column_type: str,
    default_value: str | int | float | None = None,
):
    connection = op.get_bind()

    result = connection.execute(text(f"PRAGMA table_info({table_name})"))

    columns = [row[1] for row in result]

    if column_name not in columns:
        alter_query = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"

        if default_value is not None:
            alter_query += f" DEFAULT {default_value}"

        op.execute(alter_query)
