import json

import pytest
from aiohttp import ClientResponseError
from aioresponses import aioresponses

from infinite_games.sandbox.validator.if_games.client import IfGamesClient


@pytest.mark.asyncio
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

    async def test_api_error_raised(self, client_test_env):
        # Define mock response data
        mock_response_data = ({"error_message": "not found"},)

        status_code = 404

        # Use aioresponses context manager to mock HTTP requests
        with aioresponses() as mocked:
            mocked.get(
                "/api/v2/events?from_date=1234567890&offset=0&limit=10",
                status=status_code,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            with pytest.raises(ClientResponseError) as e:
                result = await client_test_env.get_events(from_date=1234567890, offset=0, limit=10)

                mocked.assert_called_once()

                # Assert the exception
                assert e.status == 200

                # Verify the response matches the mock data
                assert result == mock_response_data
