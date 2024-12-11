import tempfile
from unittest.mock import MagicMock

import pytest

from infinite_games.sandbox.validator.db.client import Client
from infinite_games.sandbox.validator.db.operations import DatabaseOperations
from infinite_games.sandbox.validator.utils.logger.logger import AbstractLogger


class TestDbOperations:
    @pytest.fixture(scope="function")
    async def db_client(self):
        temp_db = tempfile.NamedTemporaryFile(delete=False)
        db_path = temp_db.name
        temp_db.close()

        logger = MagicMock(spec=AbstractLogger)

        db_client = Client(db_path, logger)

        await db_client.migrate()

        return db_client

    @pytest.fixture
    async def db_operations(self, db_client):
        db_operations = DatabaseOperations(db_client=db_client)

        return db_operations

    async def test_get_last_event_from_no_events(self, db_operations):
        result = await db_operations.get_last_event_from()

        assert result is None

    async def test_upsert_events(self, db_operations, db_client):
        events = [
            (
                "unique1",
                "event1",
                "market1",
                "desc1",
                "2024-12-02",
                "2024-12-03",
                "outcome1",
                "status1",
                '{"key": "value"}',
                "2000-12-02T14:30:00+01:00",
            ),
            (
                "unique2",
                "event2",
                "market2",
                "desc2",
                "2024-12-03",
                "2024-12-04",
                "outcome2",
                "status2",
                '{"key": "value"}',
                "2012-12-02T14:30:00+00:00",
            ),
        ]

        await db_operations.upsert_events(events)

        result = await db_client.many(
            """
                SELECT * FROM events
            """
        )

        assert len(result) == 2
        assert result[0][0] == "unique1"
        assert result[1][0] == "unique2"

        event_from = await db_operations.get_last_event_from()

        assert event_from == "2012-12-02T14:30:00+00:00"

    async def test_upsert_no_events(self, db_operations, db_client):
        """Test the run method when there are no events."""
        events = []

        await db_operations.upsert_events(events)

        # En
        result = await db_client.many(
            """
                SELECT * FROM events
            """
        )

        assert len(result) == 0
