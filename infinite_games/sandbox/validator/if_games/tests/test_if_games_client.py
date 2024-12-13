import json

import pytest
from aiohttp import ClientResponseError
from aioresponses import aioresponses

from infinite_games.sandbox.validator.if_games.client import IfGamesClient


class TestIfGamesClient:
    @pytest.fixture
    def client_test_env(self):
        return IfGamesClient(env="test")

    @pytest.fixture
    def client_prod_env(self):
        return IfGamesClient(env="prod")

    async def test_base_url_and_timeout(self, client_test_env, client_prod_env):
        assert client_test_env._IfGamesClient__base_url == "https://stage.ifgames.win"
        assert client_prod_env._IfGamesClient__base_url == "https://ifgames.win"

        # Verify that the timeout value was set to 30 seconds
        assert client_test_env._IfGamesClient__timeout.total == 30
        assert client_prod_env._IfGamesClient__timeout.total == 30

    @pytest.mark.parametrize(
        "from_date,offset,limit",
        [
            (None, 0, 10),  # Missing from_date
            (1234567890, None, 10),  # Missing offset
            (1234567890, 0, None),  # Missing limit
        ],
    )
    async def test_get_events_invalid_params(self, client_test_env, from_date, offset, limit):
        with pytest.raises(ValueError, match="Invalid parameters"):
            await client_test_env.get_events(from_date=from_date, offset=offset, limit=limit)

    async def test_get_events_response(self, client_test_env):
        # Define mock response data
        mock_response_data = {
            "count": 2,
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

        # Use aioresponses context manager to mock HTTP requests
        with aioresponses() as mocked:
            mocked.get(
                "/api/v2/events?from_date=1234567890&offset=0&limit=10",
                status=200,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            result = await client_test_env.get_events(from_date=1234567890, offset=0, limit=10)

            mocked.assert_called_once()

            # Verify the response matches the mock data
            assert result == mock_response_data

    async def test_get_events_error_raised(self, client_test_env):
        # Define mock response data
        mock_response_data = {"message": "Internal error"}
        status_code = 500

        # Use aioresponses context manager to mock HTTP requests
        with aioresponses() as mocked:
            url_path = "/api/v2/events?from_date=1234567890&offset=0&limit=10"
            mocked.get(
                url_path,
                status=status_code,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            with pytest.raises(ClientResponseError) as e:
                await client_test_env.get_events(from_date=1234567890, offset=0, limit=10)

            mocked.assert_called_with(url_path)

            # Assert the exception
            assert e.value.status == status_code

    @pytest.mark.parametrize(
        "event_id",
        [
            None,  # Missing event id
            1,  # Non str event id
        ],
    )
    async def test_get_event_invalid_params(self, client_test_env, event_id):
        with pytest.raises(ValueError, match="Invalid parameter"):
            await client_test_env.get_event(event_id=event_id)

    async def test_get_event_response(self, client_test_env):
        # Define mock response data
        mock_response_data = {
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
        }

        with aioresponses() as mocked:
            event_id = "26d4322e-f2ac-4611-b8ed-e448d84975f1"
            url_path = f"/api/v2/events/{event_id}"

            mocked.get(
                url_path,
                status=200,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            result = await client_test_env.get_event(event_id)

            mocked.assert_called_once()

            # Verify the response matches the mock data
            assert result == mock_response_data

    async def test_get_event_error_raised(self, client_test_env):
        # Define mock response data
        mock_response_data = {"detail": "EVENT_NOT_FOUND"}
        status_code = 404

        # Use aioresponses context manager to mock HTTP requests
        with aioresponses() as mocked:
            event_id = "26d4322e-f2ac-4611-b8ed-e448d84975f1"
            url_path = f"/api/v2/events/{event_id}"

            mocked.get(
                url_path,
                status=status_code,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            with pytest.raises(ClientResponseError) as e:
                await client_test_env.get_event(event_id)

            mocked.assert_called_with(url_path)

            # Assert the exception
            assert e.value.status == status_code
