import asyncio
import heapq
import typing

from neurons.miner.forecasters.base import BaseForecaster
from neurons.validator.utils.logger.logger import InfiniteGamesLogger

WAIT_TIME_SECONDS = 5


class TaskExecutor:
    def __init__(self, logger: InfiniteGamesLogger):
        self.lock = asyncio.Lock()
        self.tasks = []
        self.logger = logger

    async def execute(self):
        self.logger.debug("Task executor started")
        while True:
            try:
                async with self.lock:
                    if self.tasks:
                        task = heapq.heappop(self.tasks)
                    else:
                        task = None

                if task:
                    await task.run()
                    self.logger.debug(f"Task {task.event.event_id} completed")
            except Exception as e:
                self.logger.error(f"Error executing task: {str(e)}")
            finally:
                await asyncio.sleep(WAIT_TIME_SECONDS)

    async def add_task(self, task: typing.Type[BaseForecaster]):
        async with self.lock:
            heapq.heappush(self.tasks, task)
            self.logger.debug(
                f"Added task {task.event.event_id} to executor, size of heap: {len(self.tasks)}"
            )
