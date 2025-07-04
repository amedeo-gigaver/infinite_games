import asyncio
import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError
from aiohttp.web import Response
from aioresponses import aioresponses
from bittensor_wallet import Wallet
from yarl import URL

from neurons.validator.if_games.client import IfGamesClient, IfgamesEnvType
from neurons.validator.utils.git import commit_short_hash
from neurons.validator.utils.logger.logger import InfiniteGamesLogger
from neurons.validator.version import __version__


def make_client_test_env(env: IfgamesEnvType = "test"):
    logger = MagicMock(spec=InfiniteGamesLogger)

    hotkey_mock = MagicMock(
        sign=MagicMock(side_effect=lambda x: x.encode("utf-8")),
        ss58_address="ss58_address",
        public_key=b"public_key",
    )

    bt_wallet = MagicMock(
        spec=Wallet, get_hotkey=MagicMock(return_value=hotkey_mock), hotkey=hotkey_mock
    )

    return IfGamesClient(env=env, logger=logger, bt_wallet=bt_wallet)


class TestIfGamesClient:
    @pytest.fixture
    def client_test_env(self):
        return make_client_test_env(env="test")

    @pytest.mark.parametrize(
        "client,expected_base_url",
        [
            (
                make_client_test_env(env="test"),
                "https://stg.ifgames.win",
            ),
            (
                make_client_test_env(env="prod"),
                "https://ifgames.win",
            ),
        ],
    )
    async def test_default_session_config(self, client: IfGamesClient, expected_base_url: str):
        session = client.create_session()

        assert session._base_url == URL(expected_base_url)
        assert session._timeout.total == 90

        # Verify that the default headers were set correctly
        assert session.headers["Validator-Version"] == __version__
        assert session.headers["Validator-Hash"] == commit_short_hash
        assert session.headers["Validator-Public-Key"] == b"public_key".hex()

    async def test_logger_interceptors_success(self, client_test_env: IfGamesClient):
        logger = client_test_env._IfGamesClient__logger

        context = SimpleNamespace()
        method = "GET"
        response_status = 200
        url = "/test"

        # Test success response
        await client_test_env.on_request_start(None, context, None)
        await client_test_env.on_request_end(
            None,
            context,
            MagicMock(
                response=Response(status=response_status),
                method=method,
                url=URL(url),
            ),
        )

        # Verify the debug call
        logger.debug.assert_called_once_with(
            "Http request finished",
            extra={
                "response_status": response_status,
                "method": method,
                "url": url,
                "elapsed_time_ms": pytest.approx(100, abs=100),
            },
        )

    async def test_logger_interceptors_error(self, client_test_env: IfGamesClient):
        logger = client_test_env._IfGamesClient__logger

        context = SimpleNamespace()
        method = "GET"
        response_status = 500
        response_message = '{"message": "error"}'
        url = "/test"

        response = MagicMock(status=response_status, text=AsyncMock(return_value=response_message))
        params = MagicMock(response=response, method=method, url=URL(url))

        # Test success response
        await client_test_env.on_request_start(None, context, None)
        await client_test_env.on_request_end(None, context, params)

        # Verify the error call
        logger.error.assert_called_once_with(
            "Http request failed",
            extra={
                "response_status": response_status,
                "response_message": response_message,
                "method": method,
                "url": url,
                "elapsed_time_ms": pytest.approx(100, abs=100),
            },
        )

    async def test_logger_interceptors_exception(self, client_test_env: IfGamesClient):
        logger = client_test_env._IfGamesClient__logger

        context = SimpleNamespace()
        method = "GET"
        url = "/test"

        # Test success response
        await client_test_env.on_request_start(None, context, None)
        await client_test_env.on_request_exception(
            None,
            context,
            MagicMock(method=method, url=URL(url)),
        )

        # Verify the debug call
        logger.exception.assert_called_once_with(
            "Http request exception",
            extra={
                "method": method,
                "url": url,
                "elapsed_time_ms": pytest.approx(100, abs=100),
            },
        )

    async def test_logger_interceptors_cancelled_error_exception(
        self, client_test_env: IfGamesClient
    ):
        logger = client_test_env._IfGamesClient__logger

        context = SimpleNamespace()
        method = "GET"
        url = "/test"

        await client_test_env.on_request_start(None, context, None)
        await client_test_env.on_request_exception(
            None,
            context,
            MagicMock(method=method, url=URL(url), exception=asyncio.exceptions.CancelledError()),
        )

        # Verify no log call
        logger.exception.assert_not_called()

    @pytest.mark.parametrize(
        "from_date,offset,limit",
        [
            (None, 0, 10),  # Missing from_date
            (1234567890, None, 10),  # Missing offset
            (1234567890, 0, None),  # Missing limit
        ],
    )
    async def test_get_events_invalid_params(
        self, client_test_env: IfGamesClient, from_date, offset, limit
    ):
        with pytest.raises(ValueError, match="Invalid parameters"):
            await client_test_env.get_events(from_date=from_date, offset=offset, limit=limit)

    async def test_get_events_response(self, client_test_env: IfGamesClient):
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

    async def test_get_events_error_raised(self, client_test_env: IfGamesClient):
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
        "resolved_since,offset,limit",
        [
            (None, 0, 10),  # Missing resolved_since
            (1, 0, 10),  # Not str resolved_since
            ("2025-01-23T16:10:15Z", None, 10),  # Missing offset
            ("2025-01-23T16:10:15Z", 0, None),  # Missing limit
        ],
    )
    async def test_get_resolved_events_invalid_params(
        self, client_test_env: IfGamesClient, resolved_since, offset, limit
    ):
        with pytest.raises(ValueError, match="Invalid parameters"):
            await client_test_env.get_resolved_events(
                resolved_since=resolved_since, offset=offset, limit=limit
            )

    async def test_get_resolved_events_response(self, client_test_env: IfGamesClient):
        # Define mock response data

        mock_response_data = {
            "count": 2,
            "items": [
                {
                    "event_id": "21a1578e-705b-4935-9dd1-5138bf279ad0",
                    "answer": 0,
                    "resolved_at": "2025-01-23T16:10:15Z",
                },
                {
                    "event_id": "2837d80d-6c90-4b10-9dda-44ee0db617a3",
                    "answer": 1,
                    "resolved_at": "2025-01-23T16:10:15Z",
                },
            ],
        }

        # Use aioresponses context manager to mock HTTP requests
        with aioresponses() as mocked:
            mocked.get(
                "/api/v2/events/resolved?resolved_since=2000-12-30T14:30&offset=0&limit=10",
                status=200,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            result = await client_test_env.get_resolved_events(
                resolved_since="2000-12-30T14:30", offset=0, limit=10
            )

            mocked.assert_called_once()

            # Verify the response matches the mock data
            assert result == mock_response_data

    async def test_get_resolved_events_error_raised(self, client_test_env: IfGamesClient):
        # Define mock response data
        mock_response_data = {"message": "Internal error"}
        status_code = 500

        # Use aioresponses context manager to mock HTTP requests
        with aioresponses() as mocked:
            url_path = "/api/v2/events/resolved?resolved_since=2000-12-30T14:30&offset=0&limit=10"
            mocked.get(
                url_path,
                status=status_code,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            with pytest.raises(ClientResponseError) as e:
                await client_test_env.get_resolved_events(
                    resolved_since="2000-12-30T14:30", offset=0, limit=10
                )

            mocked.assert_called_with(url_path)

            # Assert the exception
            assert e.value.status == status_code

    @pytest.mark.parametrize(
        "deleted_since,offset,limit",
        [
            (None, 0, 10),  # Missing deleted_since
            (1, 0, 10),  # Not str deleted_since
            ("2025-01-23T16:10:15Z", None, 10),  # Missing offset
            ("2025-01-23T16:10:15Z", 0, None),  # Missing limit
        ],
    )
    async def test_get_events_deleted_invalid_params(
        self, client_test_env: IfGamesClient, deleted_since, offset, limit
    ):
        with pytest.raises(ValueError, match="Invalid parameters"):
            await client_test_env.get_events_deleted(
                deleted_since=deleted_since, offset=offset, limit=limit
            )

    async def test_get_events_deleted_response(self, client_test_env: IfGamesClient):
        # Define mock response data

        mock_response_data = {
            "count": 2,
            "items": [
                {
                    "event_id": "21a1578e-705b-4935-9dd1-5138bf279ad0",
                    "deleted_at": "2025-01-23T16:10:15Z",
                },
                {
                    "event_id": "2837d80d-6c90-4b10-9dda-44ee0db617a3",
                    "deleted_at": "2025-01-23T16:10:15Z",
                },
            ],
        }

        # Use aioresponses context manager to mock HTTP requests
        with aioresponses() as mocked:
            mocked.get(
                "/api/v2/events/deleted?deleted_since=2000-12-30T14:30&offset=0&limit=10",
                status=200,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            result = await client_test_env.get_events_deleted(
                deleted_since="2000-12-30T14:30", offset=0, limit=10
            )

            mocked.assert_called_once()

            # Verify the response matches the mock data
            assert result == mock_response_data

    async def test_get_events_deleted_error_raised(self, client_test_env: IfGamesClient):
        # Define mock response data
        mock_response_data = {"message": "Internal error"}
        status_code = 500

        # Use aioresponses context manager to mock HTTP requests
        with aioresponses() as mocked:
            url_path = "/api/v2/events/deleted?deleted_since=2000-12-30T14:30&offset=0&limit=10"
            mocked.get(
                url_path,
                status=status_code,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            with pytest.raises(ClientResponseError) as e:
                await client_test_env.get_events_deleted(
                    deleted_since="2000-12-30T14:30", offset=0, limit=10
                )

            mocked.assert_called_with(url_path)

            # Assert the exception
            assert e.value.status == status_code

    def test_make_auth_headers(self, client_test_env: IfGamesClient):
        body = {"fake": "body"}

        auth_headers = client_test_env.make_auth_headers(body=body)

        encoded = base64.b64encode(json.dumps(body).encode("utf-8")).decode("utf-8")

        assert auth_headers == {
            "Authorization": f"Bearer {encoded}",
            "Validator": "ss58_address",
        }

    async def test_post_predictions(self, client_test_env: IfGamesClient):
        # Define mock response data
        mock_response_data = {"fake_response": "ok"}

        predictions = {"submissions": [{"fake_data": "fake_data"}], "events": None}

        with aioresponses() as mocked:
            url_path = "/api/v1/validators/data"

            mocked.post(
                url_path,
                status=200,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            result = await client_test_env.post_predictions(predictions=predictions)

            mocked.assert_called_with(url=url_path, method="POST", json=predictions)

            # Verify the response matches
            assert result == mock_response_data

    async def test_post_predictions_error_raised(self, client_test_env: IfGamesClient):
        # Define mock response data
        mock_response_data = {"fake_response": "ok"}

        predictions = {"submissions": [{"fake_data": "fake_data"}], "events": None}

        status_code = 500

        with aioresponses() as mocked:
            url_path = "/api/v1/validators/data"

            mocked.post(
                url_path,
                status=status_code,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            with pytest.raises(ClientResponseError) as e:
                await client_test_env.post_predictions(predictions=predictions)

            mocked.assert_called_with(url=url_path, method="POST", json=predictions)

            # Assert the exception
            assert e.value.status == status_code

    async def test_post_scores(self, client_test_env: IfGamesClient):
        # Define mock response data
        mock_response_data = {"fake_response": "ok"}

        scores = {"results": [{"fake_data": "fake_data"}]}

        with aioresponses() as mocked:
            url_path = "/api/v1/validators/results"

            mocked.post(
                url_path,
                status=200,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            result = await client_test_env.post_scores(scores=scores)

            mocked.assert_called_with(url=url_path, method="POST", json=scores)

            # Verify the response matches
            assert result == mock_response_data

    async def test_post_scores_error_raised(self, client_test_env: IfGamesClient):
        # Define mock response data
        mock_response_data = {"fake_response": "ok"}

        scores = {"results": [{"fake_data": "fake_data"}]}

        status_code = 500

        with aioresponses() as mocked:
            url_path = "/api/v1/validators/results"

            mocked.post(
                url_path,
                status=status_code,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            with pytest.raises(ClientResponseError) as e:
                await client_test_env.post_scores(scores=scores)

            mocked.assert_called_with(
                url=url_path,
                method="POST",
                json=scores,
            )

            # Assert the exception
            assert e.value.status == status_code
