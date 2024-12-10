import asyncio

import pytest

from infinite_games.sandbox.validator.tasks_scheduler import Task, TasksScheduler


class TestTasksScheduler:
    @pytest.fixture(scope="class")
    def await_start_with_timeout(self):
        async def _await_start_with_timeout(start_future, timeout):
            try:
                return await asyncio.wait_for(start_future, timeout)  # Run with a timeout
            except asyncio.TimeoutError:
                pass

        return _await_start_with_timeout

    @pytest.fixture(scope="function")
    def scheduler(self):
        class TestLogger:
            def info(self, _):
                pass

            def error(self, _):
                pass

        return TasksScheduler(logger=TestLogger())

    async def test_scheduler_no_tasks(self, scheduler):
        # Test that the scheduler doesn't crash when no tasks are added
        await scheduler.start()

    async def test_add_task(self, scheduler):
        async def dummy_task():
            pass

        task = Task(name="Test Task", interval_seconds=5, task_function=dummy_task)
        scheduler.add(task)

        assert len(scheduler._TasksScheduler__tasks) == 1
        assert scheduler._TasksScheduler__tasks[0].name == "Test Task"

    async def test_schedule_task_execution(self, scheduler, await_start_with_timeout):
        runs = 0

        interval_seconds = 0.5

        async def dummy_task():
            nonlocal runs
            runs += 1

        task = Task(name="Test Task", interval_seconds=interval_seconds, task_function=dummy_task)
        scheduler.add(task)

        # Run the scheduler for a short time to allow task to execute
        await await_start_with_timeout(
            start_future=scheduler.start(), timeout=interval_seconds * 2.1
        )

        # Ensure that the task was executed
        assert runs == 3

    async def test_tasks_run_concurrently_at_intervals(self, scheduler, await_start_with_timeout):
        interval_task = 0.5

        runs_task_1 = 0

        async def dummy_task_1():
            nonlocal runs_task_1
            runs_task_1 += 1

        runs_task_2 = 0

        async def dummy_task_2():
            nonlocal runs_task_2
            runs_task_2 += 1

        task_1 = Task(
            name="Test Task 1", interval_seconds=interval_task, task_function=dummy_task_1
        )
        task_2 = Task(
            name="Test Task 2", interval_seconds=interval_task, task_function=dummy_task_2
        )

        scheduler.add(task_1)
        scheduler.add(task_2)

        # Run the scheduler for a short time to allow tasks to execute
        await await_start_with_timeout(start_future=scheduler.start(), timeout=interval_task * 2.1)

        # Ensure that the tasks were executed N times
        assert runs_task_1 == 3
        assert runs_task_2 == 3

    async def test_task_execution_error(self, scheduler, await_start_with_timeout):
        runs = 0

        interval_seconds = 0.5

        # Create an async function that will simulate an error
        async def dummy_task_with_error():
            nonlocal runs
            runs += 1

            raise Exception("Simulated error")

        task = Task(
            name="Test Task", interval_seconds=interval_seconds, task_function=dummy_task_with_error
        )
        scheduler.add(task)

        # Run the scheduler for a short time to allow task to execute
        await await_start_with_timeout(
            start_future=scheduler.start(), timeout=interval_seconds * 1.5
        )

        # Ensure that the task was executed by checking the event
        assert runs == 2

        # Check that the task status is "idle" after execution
        assert task.status == "idle"

    async def test_task_with_invalid_status(self, scheduler, await_start_with_timeout):
        # Verify the task is not executed because its status is not "unscheduled"
        runs = 0

        async def dummy_task():
            nonlocal runs
            runs += 1

        # Create a task and manually set it to "idle"
        task = Task(name="Test Task", interval_seconds=0.01, task_function=dummy_task)
        task.status = "idle"
        scheduler.add(task)

        # Run the scheduler asynchronously
        await await_start_with_timeout(start_future=scheduler.start(), timeout=1)

        # Verify that the task function is NOT executed because it is already started
        assert runs == 0
        assert task.status == "idle"
