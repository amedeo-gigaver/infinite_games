import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from neurons.miner.forecasters.base import BaseForecaster
from neurons.miner.utils.task_executor import TaskExecutor


class MockResolver(BaseForecaster):
    def __init__(self, event=None):
        self.event = Mock()
        self.event.cutoff = datetime.now(timezone.utc)
        self.run = AsyncMock()
        self.logger = Mock()

    def _run(self):
        pass


@pytest.fixture
def task_executor():
    return TaskExecutor(logger=Mock())


@pytest.mark.asyncio
async def test_add_task(task_executor):
    mock_task = MockResolver()

    await task_executor.add_task(mock_task)

    assert len(task_executor.tasks) == 1
    assert task_executor.tasks[0] == mock_task


@pytest.mark.asyncio
async def test_execute_on_empty_queue(task_executor):
    execute_task = asyncio.create_task(task_executor.execute())
    await asyncio.sleep(0.2)
    execute_task.cancel()

    try:
        await execute_task
    except asyncio.CancelledError:
        pass

    assert len(task_executor.tasks) == 0


@pytest.mark.asyncio
async def test_execute_single_task(task_executor):
    mock_task = MockResolver()
    await task_executor.add_task(mock_task)

    execute_task = asyncio.create_task(task_executor.execute())
    await asyncio.sleep(0.1)
    execute_task.cancel()

    try:
        await execute_task
    except asyncio.CancelledError:
        pass

    mock_task.run.assert_called_once()
    assert len(task_executor.tasks) == 0


@pytest.mark.asyncio
async def test_execute_run_error(task_executor):
    mock_task = MockResolver()
    mock_task.run = AsyncMock(side_effect=Exception("Test error"))
    await task_executor.add_task(mock_task)

    execute_task = asyncio.create_task(task_executor.execute())
    await asyncio.sleep(0.2)
    execute_task.cancel()

    try:
        await execute_task
    except asyncio.CancelledError:
        pass

    assert len(task_executor.tasks) == 0
