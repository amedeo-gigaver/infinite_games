from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from neurons.miner.main import Miner
from neurons.miner.models.event import MinerEvent, MinerEventStatus
from neurons.protocol import EventPrediction, EventPredictionSynapse


class MockMinerWithActualForwardMethod(Miner):
    def __init__(self):
        self.metagraph = Mock()
        self.subtensor = Mock()
        self.subtensor.network = "mock"
        self.is_testnet = False
        self.uid = 42
        self.storage = AsyncMock()
        self.storage.get = AsyncMock()
        self.storage.set = AsyncMock()
        self.task_executor = AsyncMock()
        self.assign_resolver = AsyncMock()
        self.logger = Mock()


@pytest.fixture
async def miner():
    miner = MockMinerWithActualForwardMethod()
    return miner


@pytest.mark.asyncio
async def test_forward_expired_event(miner):
    past_time = datetime.now(timezone.utc) - timedelta(hours=24)

    event_data = EventPrediction(
        event_id="test_event_2",
        market_type="test_type",
        description="Will BTC reach 100k by 2024?",
        cutoff=int(past_time.timestamp()),
        metadata={},
        probability=None,
        reasoning=None,
        miner_answered=False,
    )

    synapse = EventPredictionSynapse(events={"test_event_2": event_data})

    miner.storage.get.return_value = None

    result = await miner.forward(synapse)

    assert result.events["test_event_2"].miner_answered is False
    assert result.events["test_event_2"].probability is None

    miner.storage.get.assert_called_once_with("test_event_2")
    miner.storage.set.assert_not_called()


@pytest.mark.asyncio
async def test_forward_existing_event(miner):
    future_time = datetime.now(timezone.utc) + timedelta(hours=24)

    validator_event = EventPrediction(
        event_id="test_event_3",
        market_type="test_type",
        description="Will BTC reach 100k by 2024?",
        cutoff=int(future_time.timestamp()),
        metadata={},
        probability=None,
        reasoning=None,
        miner_answered=False,
    )

    synapse = EventPredictionSynapse(events={"test_event_3": validator_event})

    existing_event = MinerEvent.model_validate(validator_event.model_dump())
    existing_event.status = MinerEventStatus.RESOLVED
    existing_event.probability = 0
    miner.storage.get.return_value = existing_event

    result = await miner.forward(synapse)

    assert result.events["test_event_3"].miner_answered is True
    assert result.events["test_event_3"].probability == 0

    miner.storage.get.assert_called_once_with("test_event_3")
    miner.storage.set.assert_not_called()
