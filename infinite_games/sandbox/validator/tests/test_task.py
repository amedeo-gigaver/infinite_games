import asyncio

import pytest

from infinite_games.sandbox.validator.tasks_scheduler import Task


class TestTask:
    async def test_task_initialization_valid(self):
        async def dummy_function():
            pass

        task = Task(name="Test Task", interval_seconds=5, task_function=dummy_function)

        assert task.name == "Test Task"
        assert task.interval_seconds == 5
        assert task.status == "unscheduled"
        assert callable(task.task_function)

    def test_task_initialization_invalid(self):
        with pytest.raises(ValueError):
            Task(name=None, interval_seconds=5, task_function=lambda: asyncio.Future())

        with pytest.raises(ValueError):
            Task(name="Test Task", interval_seconds=-1, task_function=lambda: asyncio.Future())

        with pytest.raises(ValueError):
            Task(name="Test Task", interval_seconds=5, task_function=None)

        with pytest.raises(ValueError):
            Task(name="Test Task", interval_seconds=None, task_function=lambda: asyncio.Future())
