import json
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
from aioresponses import aioresponses

from infinite_games.sandbox.validator.db.client import Client
from infinite_games.sandbox.validator.db.operations import DatabaseOperations
from infinite_games.sandbox.validator.if_games.client import IfGamesClient
from infinite_games.sandbox.validator.models.event import EventStatus
from infinite_games.sandbox.validator.tasks.resolve_events import ResolveEvents
from infinite_games.sandbox.validator.utils.logger.logger import AbstractLogger


class TestResolveEventsTask:
    @pytest.fixture
    def db_operations_mock(self):
        db_operations = AsyncMock(spec=DatabaseOperations)

        return db_operations

    @pytest.fixture
    def api_client_mock(self):
        api_client = AsyncMock(spec=IfGamesClient)

        return api_client

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
    def db_operations(self, db_client: Client):
        return DatabaseOperations(db_client=db_client)

    @pytest.fixture
    def resolve_events_task(self, db_operations: DatabaseOperations):
        api_client = IfGamesClient(env="test")

        return ResolveEvents(
            interval_seconds=60.0,
            db_operations=db_operations,
            api_client=api_client,
        )

    async def test_run_no_pending_events(
        self,
        db_operations_mock: DatabaseOperations,
    ):
        """Test the run method when there are no pending events."""
        # Arrange
        api_client = IfGamesClient(env="test")

        resolve_events_task = ResolveEvents(
            interval_seconds=60.0,
            db_operations=db_operations_mock,
            api_client=api_client,
        )
        db_operations_mock.get_pending_events.return_value = []

        # Act
        await resolve_events_task.run()

        # Assert
        db_operations_mock.resolve_event.assert_not_called()

    async def test_resolve_events(
        self,
        db_client: Client,
        db_operations: DatabaseOperations,
        resolve_events_task: ResolveEvents,
    ):
        """Test that task resolves correctly pending events."""
        # Arrange
        events = [
            (
                "unique_1",
                "event_id_1",
                "market1",
                "desc1",
                "2024-12-02",
                "2024-12-03",
                None,
                EventStatus.DISCARDED,
                '{"key": "value"}',
                "2000-12-02T14:30:00+00:00",
                "2000-12-30T14:30:00+00:00",
                "2000-12-31T14:30:00+00:00",
            ),
            (
                "unique_2",
                "event_id_2",
                "market1",
                "desc1",
                "2024-12-02",
                "2024-12-03",
                None,
                EventStatus.PENDING,
                '{"key": "value"}',
                "2000-12-02T14:30:00+00:00",
                "2000-12-30T14:30:00+00:00",
                "2000-12-31T14:30:00+00:00",
            ),
            (
                "unique_3",
                "event_id_3",
                "market1",
                "desc1",
                "2024-12-02",
                "2024-12-03",
                None,
                EventStatus.SETTLED,
                '{"key": "value"}',
                "2000-12-02T14:30:00+00:00",
                "2000-12-30T14:30:00+00:00",
                "2000-12-31T14:30:00+00:00",
            ),
            (
                "unique_4",
                "event_id_4",
                "market1",
                "desc1",
                "2024-12-02",
                "2024-12-03",
                None,
                EventStatus.PENDING,
                '{"key": "value"}',
                "2000-12-02T14:30:00+00:00",
                "2000-12-30T14:30:00+00:00",
                "2000-12-31T14:30:00+00:00",
            ),
        ]

        await db_operations.upsert_events(events)

        mock_response_event_2 = {
            "event_id": events[1][1],
            "answer": None,
        }

        mock_response_event_4 = {
            "event_id": events[3][1],
            "answer": 1,  # Event is settled now
        }

        # Mock API response
        with aioresponses() as mocked:
            mocked.get(
                f"/api/v2/events/{mock_response_event_2['event_id']}",
                status=200,
                body=json.dumps(mock_response_event_2).encode("utf-8"),
            )

            mocked.get(
                f"/api/v2/events/{mock_response_event_4['event_id']}",
                status=200,
                body=json.dumps(mock_response_event_4).encode("utf-8"),
            )

            # Act
            await resolve_events_task.run()

        # Assert
        response = await db_client.many(
            """
                SELECT status from events
            """
        )

        assert len(response) == 4
        assert response[0][0] == str(EventStatus.DISCARDED.value)
        assert response[1][0] == str(EventStatus.PENDING.value)
        assert response[2][0] == str(EventStatus.SETTLED.value)
        assert response[3][0] == str(EventStatus.SETTLED.value)

    async def test_resolve_events_404_410_errors(
        self,
        db_client: Client,
        db_operations: DatabaseOperations,
        resolve_events_task: ResolveEvents,
    ):
        """Test that task handles 404 & 410 on get event."""
        # Arrange
        events = [
            (
                "unique_1",
                "event_id_1",
                "market1",
                "desc1",
                "2024-12-02",
                "2024-12-03",
                None,
                EventStatus.PENDING,
                '{"key": "value"}',
                "2000-12-02T14:30:00+00:00",
                "2000-12-30T14:30:00+00:00",
                "2000-12-31T14:30:00+00:00",
            ),
            (
                "unique_2",
                "event_id_2",
                "market1",
                "desc1",
                "2024-12-02",
                "2024-12-03",
                None,
                EventStatus.PENDING,
                '{"key": "value"}',
                "2000-12-02T14:30:00+00:00",
                "2000-12-30T14:30:00+00:00",
                "2000-12-31T14:30:00+00:00",
            ),
            (
                "unique_3",
                "event_id_3",
                "market1",
                "desc1",
                "2024-12-02",
                "2024-12-03",
                None,
                EventStatus.PENDING,
                '{"key": "value"}',
                "2000-12-02T14:30:00+00:00",
                "2000-12-30T14:30:00+00:00",
                "2000-12-31T14:30:00+00:00",
            ),
        ]

        await db_operations.upsert_events(events)

        mock_response_event_1 = {"detail": "EVENT_NOT_FOUND"}

        mock_response_event_2 = {"detail": "EVENT_NO_LONGER_ACTIVE"}

        mock_response_event_3 = {
            "event_id": events[2][1],
            "answer": 1,  # Event is settled now
        }

        # Mock API response
        with aioresponses() as mocked:
            mocked.get(
                f"/api/v2/events/{events[0][1]}",
                status=404,
                body=json.dumps(mock_response_event_1).encode("utf-8"),
            )

            mocked.get(
                f"/api/v2/events/{events[1][1]}",
                status=410,
                body=json.dumps(mock_response_event_2).encode("utf-8"),
            )

            mocked.get(
                f"/api/v2/events/{events[2][1]}",
                status=200,
                body=json.dumps(mock_response_event_3).encode("utf-8"),
            )

            # Act
            await resolve_events_task.run()

        # Assert
        response = await db_client.many(
            """
                SELECT event_id, status from events
            """
        )

        assert len(response) == 1
        assert response[0][0] == events[2][1]
        assert response[0][1] == str(EventStatus.SETTLED.value)
