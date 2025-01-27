import json
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
from aioresponses import aioresponses
from bittensor_wallet import Wallet

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.if_games.client import IfGamesClient
from neurons.validator.models.event import EventStatus
from neurons.validator.sandbox.tasks.delete_events import DeleteEvents
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class TestDeleteEventsTask:
    @pytest.fixture
    def db_operations_mock(self):
        db_operations = AsyncMock(spec=DatabaseOperations)

        return db_operations

    @pytest.fixture
    def api_client_mock(self):
        api_client = AsyncMock(spec=IfGamesClient)

        return api_client

    @pytest.fixture
    def bt_wallet(self):
        hotkey_mock = MagicMock()
        hotkey_mock.sign = MagicMock(side_effect=lambda x: x.encode("utf-8"))
        hotkey_mock.ss58_address = "ss58_address"

        bt_wallet = MagicMock(spec=Wallet)
        bt_wallet.get_hotkey = MagicMock(return_value=hotkey_mock)

        return bt_wallet

    @pytest.fixture(scope="function")
    async def db_client(self):
        temp_db = tempfile.NamedTemporaryFile(delete=False)
        db_path = temp_db.name
        temp_db.close()

        logger = MagicMock(spec=InfiniteGamesLogger)

        db_client = DatabaseClient(db_path, logger)

        await db_client.migrate()

        return db_client

    @pytest.fixture
    def db_operations(self, db_client: DatabaseClient):
        return DatabaseOperations(db_client=db_client)

    @pytest.fixture
    def logger_mock(self):
        return MagicMock(spec=InfiniteGamesLogger)

    @pytest.fixture
    def delete_events_task(
        self, db_operations: DatabaseOperations, bt_wallet: Wallet, logger_mock: InfiniteGamesLogger
    ):
        api_client = IfGamesClient(env="test", logger=logger_mock, bt_wallet=bt_wallet)

        return DeleteEvents(
            interval_seconds=60.0,
            db_operations=db_operations,
            api_client=api_client,
            logger=logger_mock,
            page_size=10,
        )

    async def test_run_no_pending_events(
        self,
        db_operations: DatabaseOperations,
        api_client_mock: MagicMock,
        logger_mock: InfiniteGamesLogger,
    ):
        """Test the run method when there are no pending events."""

        # Arrange
        resolve_events_task = DeleteEvents(
            interval_seconds=60.0,
            db_operations=db_operations,
            api_client=api_client_mock,
            logger=logger_mock,
            page_size=10,
        )

        # Act
        await resolve_events_task.run()

        # Assert
        api_client_mock.get_events_deleted.assert_not_called()

    async def test_resolve_events(
        self,
        db_client: DatabaseClient,
        db_operations: DatabaseOperations,
        delete_events_task: DeleteEvents,
    ):
        """Test that task deletes correctly deleted events."""
        # Arrange
        delete_events_task.page_size = 1

        events = [
            (
                "unique_1",
                "event_id_1",
                "truncated_market1",
                "market_1",
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
                "truncated_market1",
                "market_1",
                "desc1",
                "2024-12-02",
                "2024-12-03",
                None,
                EventStatus.PENDING,
                '{"key": "value"}',
                "1950-12-02T14:30:00+00:00",
                "2000-12-30T14:30:00+00:00",
                "2000-12-31T14:30:00+00:00",
            ),
            (
                "unique_3",
                "event_id_3",
                "truncated_market1",
                "market_1",
                "desc1",
                "2024-12-02",
                "2024-12-03",
                None,
                EventStatus.PENDING,
                '{"key": "value"}',
                "1900-12-02T14:30:00+00:00",
                "2000-12-30T14:30:00+00:00",
                "2000-12-31T14:30:00+00:00",
            ),
            (
                "unique_4",
                "event_id_4",
                "truncated_market1",
                "market_1",
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

        api_response_1 = {
            "count": 1,
            "items": [
                {
                    "event_id": events[3][1],
                    "deleted_at": "2012-09-10T20:43:02Z",
                }
            ],
        }

        api_response_2 = {
            "count": 1,
            "items": [
                {
                    "event_id": events[2][1],
                    "deleted_at": "2024-09-11T20:43:02Z",
                }
            ],
        }

        api_response_3 = {
            "count": 0,
            "items": [],
        }

        api_response_4 = {
            "count": 1,
            "items": [
                {
                    "event_id": events[1][1],
                    "deleted_at": "2024-09-10T20:43:02Z",
                }
            ],
        }

        api_response_5 = {
            "count": 0,
            "items": [],
        }

        # Mock API response
        with aioresponses() as mocked:
            first_request_deleted_since = "1900-12-02T14:30:00+00:00"

            second_request_deleted_since = "1950-12-02T14:30:00+00:00"

            mocked.get(
                f"/api/v2/events/deleted?deleted_since={first_request_deleted_since}&offset=0&limit=1",
                status=200,
                body=json.dumps(api_response_1).encode("utf-8"),
            )

            mocked.get(
                f"/api/v2/events/deleted?deleted_since={first_request_deleted_since}&offset=1&limit=1",
                status=200,
                body=json.dumps(api_response_2).encode("utf-8"),
            )

            mocked.get(
                f"/api/v2/events/deleted?deleted_since={first_request_deleted_since}&offset=2&limit=1",
                status=200,
                body=json.dumps(api_response_3).encode("utf-8"),
            )

            mocked.get(
                f"/api/v2/events/deleted?deleted_since={second_request_deleted_since}&offset=0&limit=1",
                status=200,
                body=json.dumps(api_response_4).encode("utf-8"),
            )

            mocked.get(
                f"/api/v2/events/deleted?deleted_since={second_request_deleted_since}&offset=1&limit=1",
                status=200,
                body=json.dumps(api_response_5).encode("utf-8"),
            )

            # Act
            await delete_events_task.run()

            # Assert
            response = await db_client.many(
                """
                    SELECT event_id from events
                """
            )

            # Event 3 and 4 are deleted, events 1 and 2 are left
            assert len(response) == 2
            assert response[0][0] == events[0][1]
            assert response[1][0] == events[1][1]

            # Act run again
            await delete_events_task.run()

            # Assert
            response = await db_client.many(
                """
                    SELECT event_id from events
                """
            )

            # Event 2 is deleted, event 1 is left
            assert len(response) == 1
            assert response[0][0] == events[0][1]
