import asyncio
from unittest.mock import ANY

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.db.tests.test_utils import TestDbOperationsBase
from neurons.validator.models.reasoning import ReasoningModel


class TestDbOperationsPart3(TestDbOperationsBase):
    async def test_upsert_reasonings(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test basic upsert of reasonings"""
        reasonings = [
            ReasoningModel(
                event_id="event1",
                miner_uid=1,
                miner_hotkey="hotkey1",
                reasoning="Test reasoning 1",
                exported=False,
            ),
            ReasoningModel(
                event_id="event2",
                miner_uid=2,
                miner_hotkey="hotkey2",
                reasoning="Test reasoning 2",
                exported=True,
            ),
        ]

        await db_operations.upsert_reasonings(reasonings)

        # Verify the reasonings were inserted
        rows = await db_client.many(
            """
                SELECT
                    event_id,
                    miner_uid,
                    miner_hotkey,
                    reasoning,
                    exported,
                    created_at,
                    updated_at
                FROM
                    reasoning
                ORDER BY
                    ROWID ASC
            """
        )

        assert len(rows) == 2
        assert rows[0] == (
            "event1",
            1,
            "hotkey1",
            "Test reasoning 1",
            0,
            ANY,
            ANY,
        )
        assert rows[1] == ("event2", 2, "hotkey2", "Test reasoning 2", 1, ANY, ANY)

    async def test_upsert_reasonings_update_existing(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test updating existing reasonings"""
        # First insert
        initial_reasonings = [
            ReasoningModel(
                event_id="event1",
                miner_uid=1,
                miner_hotkey="hotkey1",
                reasoning="Initial reasoning",
                exported=False,
            )
        ]

        await db_operations.upsert_reasonings(initial_reasonings)

        initial_rows = await db_client.many(
            """
                SELECT
                    event_id,
                    miner_uid,
                    miner_hotkey,
                    reasoning,
                    exported,
                    created_at,
                    updated_at
                FROM
                    reasoning
            """
        )

        assert len(initial_rows) == 1
        assert initial_rows[0] == ("event1", 1, "hotkey1", "Initial reasoning", 0, ANY, ANY)

        # Update the same reasoning
        updated_reasonings = [
            ReasoningModel(
                event_id="event1",
                miner_uid=1,
                miner_hotkey="hotkey1",
                reasoning="Updated reasoning",
                exported=True,
            )
        ]

        # Delay upsert for updated_at to be different
        await asyncio.sleep(1)

        await db_operations.upsert_reasonings(updated_reasonings)

        # Verify the reasoning was updated
        final_rows = await db_client.many(
            """
                SELECT
                    event_id,
                    miner_uid,
                    miner_hotkey,
                    reasoning,
                    exported,
                    created_at,
                    updated_at
                FROM
                    reasoning
            """
        )

        assert len(final_rows) == 1
        # Exported is not updated
        assert final_rows[0] == ("event1", 1, "hotkey1", "Updated reasoning", 0, ANY, ANY)

        # Verify created_at and updated_at timestamps
        assert initial_rows[0][5] == final_rows[0][5]
        assert initial_rows[0][6] < final_rows[0][6]

    async def test_upsert_reasonings_empty_list(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test upserting an empty list of reasonings"""
        # Attempt to upsert an empty list
        await db_operations.upsert_reasonings([])

        # Verify no reasonings were inserted
        rows = await db_client.many(
            """
                SELECT COUNT(*) FROM reasoning
            """
        )

        assert rows[0][0] == 0
