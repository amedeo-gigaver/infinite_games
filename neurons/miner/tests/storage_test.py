import os
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from neurons.miner.models.event import MinerEvent
from neurons.miner.utils.storage import STORAGE_FILE, MinerStorage


@pytest.fixture
def storage():
    storage = MinerStorage(logger=Mock())
    yield storage
    if os.path.exists(STORAGE_FILE):
        os.remove(STORAGE_FILE)


@pytest.fixture
def sample_event():
    return MinerEvent(
        event_id="test_id",
        market_type="test_market_type",
        description="test_description",
        cutoff=datetime.now(timezone.utc) + timedelta(days=1),
        metadata={},
        probability=None,
        reasoning=None,
        miner_answered=False,
    )


@pytest.mark.asyncio
async def test_set_and_get(storage, sample_event):
    await storage.set("test_id", sample_event)
    retrieved_event = await storage.get("test_id")
    assert retrieved_event == sample_event


@pytest.mark.asyncio
async def test_get_nonexistent(storage):
    retrieved_event = await storage.get("nonexistent")
    assert retrieved_event is None


@pytest.mark.asyncio
async def test_store_and_load(storage, sample_event):
    await storage.set("test_id", sample_event)
    await storage._store()

    new_storage = MinerStorage(logger=Mock())
    await new_storage.load()

    loaded_event = await new_storage.get("test_id")
    assert loaded_event.event_id == sample_event.event_id


@pytest.mark.asyncio
async def test_load_with_condition(storage, sample_event):
    await storage.set("test_id", sample_event)
    await storage._store()

    new_storage = MinerStorage(logger=Mock())
    condition = lambda event: event.event_id == "not_test_id"  # noqa: E731
    await new_storage.load(condition=condition)

    assert await new_storage.get("test_id") is None
