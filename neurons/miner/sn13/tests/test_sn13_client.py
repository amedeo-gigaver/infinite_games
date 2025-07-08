import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from aiohttp import ClientResponseError
from aiohttp.web import Response
from aioresponses import aioresponses
from yarl import URL

from neurons.miner.models.event import MinerEvent, MinerEventStatus
from neurons.miner.models.sn13 import SN13Response
from neurons.miner.sn13.client import Subnet13Client
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


@pytest.fixture
def mock_api_key():
    os.environ["SN13_API_KEY"] = "mock_api_key"
    return os.getenv("SN13_API_KEY")


@pytest.fixture
def test_client(mock_api_key: str):
    logger = MagicMock(spec=InfiniteGamesLogger)

    return Subnet13Client(logger=logger)


@pytest.fixture
def mock_event():
    return MinerEvent(
        event_id="123",
        market_type="crypto",
        description="Some",
        cutoff=datetime.now(timezone.utc) + timedelta(days=1),
        metadata={},
        probability=0.5,
        reasoning=None,
        miner_answered=False,
        status=MinerEventStatus.UNRESOLVED,
    )


@pytest.fixture(scope="function")
def openai_client():
    with patch("neurons.miner.sn13.client.AsyncOpenAI", Mock()) as mocker:
        yield mocker


def test_client_init_without_api_key():
    if "SN13_API_KEY" in os.environ:
        del os.environ["SN13_API_KEY"]

    with pytest.raises(ValueError):
        Subnet13Client(logger=MagicMock(spec=InfiniteGamesLogger))


@pytest.mark.asyncio
async def test_default_session_config(test_client: Subnet13Client, mock_api_key: str):
    session = test_client.create_session()

    assert session._base_url == URL("https://sn13.api.macrocosmos.ai")
    assert session._timeout.total == 90

    # Verify that the default headers were set correctly
    assert session.headers["X-API-Key"] == mock_api_key


@pytest.mark.asyncio
async def test_logger_interceptors_success(test_client: Subnet13Client):
    logger = test_client._Subnet13Client__logger

    context = SimpleNamespace()
    method = "POST"
    response_status = 200
    url = "/test"

    await test_client.on_request_start(None, context, None)
    await test_client.on_request_end(
        None,
        context,
        MagicMock(
            response=Response(status=response_status),
            method=method,
            url=URL(url),
        ),
    )

    logger.debug.assert_called_once_with(
        "Http request finished",
        extra={
            "response_status": response_status,
            "method": method,
            "url": url,
            "elapsed_time_ms": pytest.approx(100, abs=100),
        },
    )


@pytest.mark.asyncio
async def test_logger_interceptors_error(test_client: Subnet13Client):
    logger = test_client._Subnet13Client__logger

    context = SimpleNamespace()
    method = "GET"
    response_status = 500
    response_message = '{"message": "error"}'
    url = "/test"

    response = MagicMock(status=response_status, text=AsyncMock(return_value=response_message))
    params = MagicMock(response=response, method=method, url=URL(url))

    await test_client.on_request_start(None, context, None)
    await test_client.on_request_end(None, context, params)

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


async def test_logger_interceptors_exception(test_client: Subnet13Client):
    logger = test_client._Subnet13Client__logger

    context = SimpleNamespace()
    method = "GET"
    url = "/test"

    await test_client.on_request_start(None, context, None)
    await test_client.on_request_exception(
        None,
        context,
        MagicMock(method=method, url=URL(url)),
    )

    logger.exception.assert_called_once_with(
        "Http request exception",
        extra={
            "method": method,
            "url": url,
            "elapsed_time_ms": pytest.approx(100, abs=100),
        },
    )


async def test_logger_interceptors_cancelled_error_exception(test_client: Subnet13Client):
    logger = test_client._Subnet13Client__logger

    context = SimpleNamespace()
    method = "GET"
    url = "/test"

    await test_client.on_request_start(None, context, None)
    await test_client.on_request_exception(
        None,
        context,
        MagicMock(method=method, url=URL(url), exception=asyncio.exceptions.CancelledError()),
    )

    # Verify no log call
    logger.exception.assert_not_called()


@pytest.mark.asyncio
async def test_get_on_demand_data(test_client: Subnet13Client, mock_event: MinerEvent):
    mock_response_data = {
        "status": "success",
        "data": [
            {
                "uri": "https://example.com/data",
                "datetime": "2025-01-01",
                "source": "X",
                "label": "test",
                "content": "test",
            }
        ],
        "meta": {},
    }

    with aioresponses() as m:
        m.post(
            "/api/v1/on_demand_data_request",
            status=200,
            body=json.dumps(mock_response_data).encode("utf-8"),
        )
        result = await test_client.get_on_demand_data(
            mock_event,
            source="X",
            keywords=["bitcoin"],
        )

        m.assert_called_once()

        assert result == SN13Response(**mock_response_data)


@pytest.mark.asyncio
async def test_get_on_demand_data_error(test_client: Subnet13Client, mock_event: MinerEvent):
    mock_response_data = {"message": "Internal error"}
    status_code = 500

    with aioresponses() as m:
        m.post(
            "/api/v1/on_demand_data_request",
            status=status_code,
            body=json.dumps(mock_response_data).encode("utf-8"),
        )

        with pytest.raises(ClientResponseError) as e:
            await test_client.get_on_demand_data(mock_event)

        assert e.value.status == status_code


@pytest.mark.asyncio
async def test_get_on_demand_data_with_gpt(
    test_client: Subnet13Client, mock_event: MinerEvent, openai_client: Mock
):
    openai_client.return_value.chat.completions.create = AsyncMock(
        return_value=Mock(
            choices=[
                Mock(message=Mock(content="keywords: [bitcoin, crypto, bull run]")),
            ]
        )
    )
    mock_response_data = {
        "status": "success",
        "data": [
            {
                "uri": "https://example.com/data",
                "datetime": "2025-01-01",
                "source": "X",
                "label": "test",
                "content": "test",
            }
        ],
        "meta": {},
    }

    with aioresponses() as m:
        start_date = datetime.now(timezone.utc) - timedelta(days=1)
        end_date = datetime.now(timezone.utc)
        m.post(
            "/api/v1/on_demand_data_request",
            status=200,
            body=json.dumps(mock_response_data).encode("utf-8"),
        )
        result = await test_client.get_on_demand_data_with_gpt(
            mock_event,
            source="X",
            start_date=start_date,
            end_date=end_date,
            limit=10,
        )

        m.assert_called_once()
        m.assert_called_with(
            "/api/v1/on_demand_data_request",
            method="POST",
            json={
                "source": "X",
                "limit": 10,
                "keywords": ["bitcoin", "crypto", "bull run"],
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )
        assert result == SN13Response(**mock_response_data)


@pytest.mark.asyncio
async def test_get_on_demand_data_with_gpt_failed(
    test_client: Subnet13Client, mock_event: MinerEvent, openai_client: Mock
):
    openai_client.return_value.chat.completions.create = AsyncMock(
        side_effect=Exception("Test exception")
    )

    with pytest.raises(Exception) as e:
        await test_client.get_on_demand_data_with_gpt(mock_event)

    assert str(e.value) == "Test exception"
