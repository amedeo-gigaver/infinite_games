"""Track validators

Revision ID: 389ec9305bde
Revises: b200ceaa4d5e
Create Date: 2025-01-28 21:39:27.321997

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "389ec9305bde"
down_revision: Union[str, None] = "b200ceaa4d5e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("miners") as batch_op:
        batch_op.add_column(
            sa.Column(
                "is_validating",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("0"),
            )
        )

        batch_op.add_column(
            sa.Column(
                "validator_permit",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("0"),
            )
        )

    with op.batch_alter_table("miners") as batch_op:
        batch_op.alter_column("is_validating", existing_type=sa.Boolean(), server_default=None)

        batch_op.alter_column("validator_permit", existing_type=sa.Boolean(), server_default=None)
