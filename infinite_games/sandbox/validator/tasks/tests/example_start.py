import asyncio

from infinite_games.sandbox.validator.db.client import Client
from infinite_games.sandbox.validator.db.operations import DatabaseOperations
from infinite_games.sandbox.validator.if_games.client import IfGamesClient
from infinite_games.sandbox.validator.scheduler.tasks_scheduler import TasksScheduler
from infinite_games.sandbox.validator.tasks.pull_events import PullEvents
from infinite_games.sandbox.validator.utils.logger.logger import logger


async def main():
    db_client = Client(db_path="my_db.db", logger=logger)
    db_operations = DatabaseOperations(db_client=db_client)
    api_client = IfGamesClient(env="test", logger=logger)

    await db_client.migrate()

    task = PullEvents(
        interval_seconds=60.0, page_size=50, db_operations=db_operations, api_client=api_client
    )
    scheduler = TasksScheduler(logger=logger)

    scheduler.add(task=task)

    # await scheduler.start()


if __name__ == "__main__":
    asyncio.run(main())
