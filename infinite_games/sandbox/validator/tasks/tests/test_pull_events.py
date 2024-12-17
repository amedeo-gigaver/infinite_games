import json
import tempfile
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from aioresponses import aioresponses

from infinite_games.sandbox.validator.db.client import Client
from infinite_games.sandbox.validator.db.operations import DatabaseOperations
from infinite_games.sandbox.validator.if_games.client import IfGamesClient
from infinite_games.sandbox.validator.models.event import EventStatus
from infinite_games.sandbox.validator.tasks.pull_events import PullEvents
from infinite_games.sandbox.validator.utils.logger.logger import AbstractLogger


class TestPullEventsTask:
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

    @pytest.fixture(scope="function")
    def pull_events_task(self, db_client):
        db_operations = DatabaseOperations(db_client=db_client)
        api_client = IfGamesClient(env="test")

        return PullEvents(
            interval_seconds=60.0,
            db_operations=db_operations,
            api_client=api_client,
            page_size=1,
        )

    def test_parse_event(self, db_operations_mock, api_client_mock):
        """Test the parse_event method for correctness."""
        # Arrange
        event = {
            "event_id": "123",
            "title": "Test Event",
            "description": "This is a test.",
            "start_date": 1700000000,
            "end_date": 1700003600,
            "created_at": 1699996400,
            "answer": None,
            "market_type": "type1",
            "cutoff": 1699996800,
        }

        pull_events_task = PullEvents(
            interval_seconds=60.0,
            db_operations=db_operations_mock,
            api_client=api_client_mock,
            page_size=100,
        )

        # Act
        parsed_event = pull_events_task.parse_event(event)

        # Assert
        assert parsed_event[0] == "ifgames-123"  # unique_event_id
        assert parsed_event[1] == "123"  # event_id
        assert parsed_event[2] == "ifgames"  # truncated market_type
        assert parsed_event[3] == "Test EventThis is a test."  # description
        assert parsed_event[4] == datetime.fromtimestamp(
            event["start_date"], tz=timezone.utc
        )  # starts
        assert parsed_event[5] is None  # resolve_date
        assert parsed_event[6] is None  # outcome
        assert parsed_event[7] == EventStatus.PENDING  # status
        assert json.loads(parsed_event[8])["market_type"] == "type1"  # metadata
        assert parsed_event[9] == datetime.fromtimestamp(
            event["created_at"], tz=timezone.utc
        )  # created_at

    async def test_run_no_events(self, db_operations_mock, api_client_mock):
        """Test the run method when there are no events."""
        # Arrange
        page_size = 100

        pull_events_task = PullEvents(
            interval_seconds=60.0,
            db_operations=db_operations_mock,
            api_client=api_client_mock,
            page_size=page_size,
        )

        db_operations_mock.get_last_event_from.return_value = None

        api_client_mock.get_events.return_value = {"items": []}

        # Act
        await pull_events_task.run()

        # Assert
        api_client_mock.get_events.assert_called_with(0, 0, page_size)
        db_operations_mock.upsert_events.assert_not_called()

    async def test_start_from_empty_integration(
        self, db_client: Client, pull_events_task: PullEvents
    ):
        """Test that pulls events from 0 and iterates until the end."""
        # Arrange
        mock_response_data_1 = {
            "count": -5,
            "items": [
                {
                    "event_id": "0x123456789abcdef123456789abcdef123456789abcdef123456789abcdef1234",
                    "cutoff": 1733616000,
                    "title": "Will Tesla stock price reach $2000 by 2025?",
                    "description": (
                        "This market will resolve to 'Yes' if the closing stock price of Tesla reaches or exceeds $2000 "
                        "on any trading day in 2025. Otherwise, this market will resolve to 'No'.\n\n"
                        "Resolution source: NASDAQ official data."
                    ),
                    "market_type": "BINARY",
                    "start_date": 1733600000,
                    "created_at": 1733200000,
                    "end_date": 1733620000,
                    "answer": None,
                },
            ],
        }

        mock_response_data_2 = {
            "count": -5,
            "items": [
                {
                    "event_id": "0xabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef",
                    "cutoff": 1733617000,
                    "title": "Will AI surpass human intelligence by 2030?",
                    "description": (
                        "This market will resolve to 'Yes' if credible sources, including major AI researchers, "
                        "announce that AI has surpassed general human intelligence before December 31, 2030."
                    ),
                    "market_type": "POLYMARKET",
                    "start_date": 1733610000,
                    "created_at": 1733210000,
                    "end_date": 1733621000,
                    "answer": None,
                },
            ],
        }

        mock_response_data_3 = {
            "count": -5,
            "items": [],
        }

        # Mock API response
        with aioresponses() as mocked:
            mocked.get(
                "/api/v2/events?from_date=0&offset=0&limit=1",
                status=200,
                body=json.dumps(mock_response_data_1).encode("utf-8"),
            )

            mocked.get(
                "/api/v2/events?from_date=0&offset=1&limit=1",
                status=200,
                body=json.dumps(mock_response_data_2).encode("utf-8"),
            )

            mocked.get(
                "/api/v2/events?from_date=0&offset=2&limit=1",
                status=200,
                body=json.dumps(mock_response_data_3).encode("utf-8"),
            )

            # Act
            await pull_events_task.run()

        # Assert
        response = await db_client.many(
            """
                SELECT * from events
            """
        )

        assert len(response) == 2
        assert response[0][1] == mock_response_data_1["items"][0]["event_id"]
        assert response[1][1] == mock_response_data_2["items"][0]["event_id"]

    async def test_start_from_last_integration(
        self, db_client: Client, pull_events_task: PullEvents
    ):
        """Test that pulls events from where it left and iterates until the end."""
        # Arrange
        mock_response_data_1 = {
            "count": -5,
            "items": [
                {
                    "event_id": "0x123456789abcdef123456789abcdef123456789abcdef123456789abcdef1234",
                    "cutoff": 1733616000,
                    "title": "Will Tesla stock price reach $2000 by 2025?",
                    "description": (
                        "This market will resolve to 'Yes' if the closing stock price of Tesla reaches or exceeds $2000 "
                        "on any trading day in 2025. Otherwise, this market will resolve to 'No'.\n\n"
                        "Resolution source: NASDAQ official data."
                    ),
                    "market_type": "BINARY",
                    "start_date": 1733600000,
                    "created_at": 1733200000,
                    "end_date": 1733620000,
                    "answer": None,
                },
            ],
        }

        mock_response_data_2 = {
            "count": -5,
            "items": [],
        }

        mock_response_data_3 = {
            "count": -5,
            "items": [
                {
                    "event_id": "0xabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef",
                    "cutoff": 1733617000,
                    "title": "Will AI surpass human intelligence by 2030?",
                    "description": (
                        "This market will resolve to 'Yes' if credible sources, including major AI researchers, "
                        "announce that AI has surpassed general human intelligence before December 31, 2030."
                    ),
                    "market_type": "POLYMARKET",
                    "start_date": 1733610000,
                    "created_at": 1733210000,
                    "end_date": 1733621000,
                    "answer": None,
                },
            ],
        }

        mock_response_data_4 = {
            "count": -5,
            "items": [],
        }

        # Mock API response
        with aioresponses() as mocked:
            mocked.get(
                "/api/v2/events?from_date=0&offset=0&limit=1",
                status=200,
                body=json.dumps(mock_response_data_1).encode("utf-8"),
            )

            mocked.get(
                "/api/v2/events?from_date=0&offset=1&limit=1",
                status=200,
                body=json.dumps(mock_response_data_2).encode("utf-8"),
            )

            mocked.get(
                "/api/v2/events?from_date=1733196400&offset=0&limit=1",
                status=200,
                body=json.dumps(mock_response_data_3).encode("utf-8"),
            )

            mocked.get(
                "/api/v2/events?from_date=1733196400&offset=1&limit=1",
                status=200,
                body=json.dumps(mock_response_data_4).encode("utf-8"),
            )

            # Act
            await pull_events_task.run()
            # Run a second pass
            await pull_events_task.run()

        # Assert
        response = await db_client.many(
            """
                SELECT * from events
            """
        )

        assert len(response) == 2
        assert response[0][1] == mock_response_data_1["items"][0]["event_id"]
        assert response[1][1] == mock_response_data_3["items"][0]["event_id"]
