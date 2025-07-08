from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from neurons.miner.forecasters.base import BaseForecaster, DummyForecaster
from neurons.miner.models.event import MinerEvent, MinerEventStatus


@pytest.fixture
def mock_event():
    return MinerEvent(
        event_id="123",
        market_type="crypto",
        description="Some",
        cutoff=datetime.now(timezone.utc) + timedelta(days=1),
        metadata={},
        probability=None,
        reasoning=None,
        miner_answered=False,
        status=MinerEventStatus.UNRESOLVED,
    )


@pytest.fixture
def mock_forecaster():
    class BaseForecasterForTesting(BaseForecaster):
        _run = AsyncMock()

    return BaseForecasterForTesting


def test_constructor(mock_event, mock_forecaster):
    obj = mock_forecaster(mock_event, logger=Mock())
    assert obj.event == mock_event
    assert obj.extremize is False


@pytest.mark.asyncio
async def test_run(mock_event, mock_forecaster):
    obj = mock_forecaster(mock_event, logger=Mock())
    obj._run.return_value = 0.3, "Reasoning x"
    await obj.run()
    obj._run.assert_called_once()
    assert obj.event.get_probability() == 0.3
    assert obj.event.get_reasoning() == "Reasoning x"
    assert obj.event.get_status() == MinerEventStatus.RESOLVED


@pytest.mark.asyncio
async def test_run_extremize(mock_event, mock_forecaster):
    obj = mock_forecaster(mock_event, logger=Mock(), extremize=True)
    obj._run.return_value = 0.3, "Reasoning x"
    await obj.run()
    obj._run.assert_called_once()
    assert obj.event.get_probability() == 0
    assert obj.event.get_reasoning() == "Reasoning x"
    assert obj.event.get_status() == MinerEventStatus.RESOLVED


@pytest.mark.asyncio
async def test_run_failed(mock_event, mock_forecaster):
    obj = mock_forecaster(mock_event, logger=Mock())
    obj._run.side_effect = Exception("Test exception")
    await obj.run()
    obj._run.assert_called_once()
    assert obj.event.get_probability() is None
    assert obj.event.get_reasoning() is None
    assert obj.event.get_status() == MinerEventStatus.UNRESOLVED


@pytest.mark.asyncio
async def test_run_no_reasoning(mock_event, mock_forecaster):
    obj = mock_forecaster(mock_event, logger=Mock())
    obj._run.return_value = 0.3, None

    await obj.run()

    obj._run.assert_called_once()

    assert obj.event.get_probability() == 0.3
    assert obj.event.get_reasoning() is None
    assert obj.event.get_status() == MinerEventStatus.RESOLVED


@pytest.mark.asyncio
async def test_run_sets_probability(mock_event):
    resolver = DummyForecaster(mock_event, logger=Mock())
    prob, reasoning = await resolver._run()

    assert prob in [0, 1]
    assert reasoning == "Dummy forecaster reasoning"
