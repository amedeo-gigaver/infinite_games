import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Literal


class Logger(ABC):
    @abstractmethod
    def info(self, message: str) -> None:
        pass

    @abstractmethod
    def error(self, message: str) -> None:
        pass


@dataclass
class Task:
    """
    Represents a scheduled task with an execution interval and a callable function.
    """

    name: str  # Name of the task
    interval_seconds: int  # Interval between task executions
    task_function: Callable[[], asyncio.Future]  # Async function representing the task
    status: Literal["unscheduled", "idle", "running"] = field(
        init=False, default="unscheduled"
    )  # Task status

    def __post_init__(self):
        """
        Perform validation after the dataclass initialization.
        :raises ValueError: If any argument is invalid (e.g., None or negative interval).
        """
        if (
            not self.name
            or not callable(self.task_function)
            or not self.interval_seconds
            or self.interval_seconds < 0
        ):
            raise ValueError("Invalid arguments.")


class TasksScheduler:
    """
    Scheduler for managing and running multiple asynchronous tasks.
    """

    def __init__(self, logger: Logger):
        self.__tasks: list[Task] = []  # List to store tasks
        self.__logger = logger  # Logger

    async def __schedule_task(self, task: Task):
        """
        Private method to manage the execution of a single task.
        :param task: The task to be scheduled and executed.
        """
        while True:  # Continuously execute the task at the specified interval
            start_time = time.time()

            task.status = "running"  # Mark the task as running

            self.__logger.info(f"Task started: {task.name}")

            try:
                # Execute the task's async function
                await task.task_function()
            except Exception as e:
                # Log any exceptions that occur during task execution
                self.__logger.error(f"Task errored: {task.name}. {e}")

            task.status = "idle"  # Mark the task as idle after completion

            elapsed_time = time.time() - start_time

            self.__logger.info(f"Task finished: {task.name}. {elapsed_time:.3f} seconds")

            # Wait for the specified interval before the next execution
            await asyncio.sleep(task.interval_seconds)

    def add(self, task: Task):
        """
        Add a new task to the scheduler.
        :param task: The Task object to add.
        """
        tasks_names = {item.name for item in self.__tasks}

        if task.name in tasks_names:
            raise ValueError(f"Task '{task.name}' already added")

        self.__tasks.append(task)  # Append the task to the internal list

    async def start(self):
        """
        Start all tasks that are in the "unscheduled" state.
        This method creates a new asyncio task for each task in the scheduler.
        """
        scheduled_tasks = []  # List to hold scheduled asyncio tasks

        for task in self.__tasks:
            if task.status != "unscheduled":  # Skip tasks that are already scheduled
                continue

            # Schedule and start the task
            scheduled_tasks.append(self.__schedule_task(task))

        # Await all tasks
        await asyncio.gather(*scheduled_tasks)
