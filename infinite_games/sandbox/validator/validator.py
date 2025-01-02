import argparse
import asyncio

import bittensor as bt

from infinite_games.sandbox.validator.db.client import DatabaseClient
from infinite_games.sandbox.validator.db.operations import DatabaseOperations
from infinite_games.sandbox.validator.if_games.client import IfGamesClient
from infinite_games.sandbox.validator.scheduler.tasks_scheduler import TasksScheduler
from infinite_games.sandbox.validator.tasks.pull_events import PullEvents
from infinite_games.sandbox.validator.tasks.resolve_events import ResolveEvents
from infinite_games.sandbox.validator.utils.logger.logger import logger


def get_config():
    parser = argparse.ArgumentParser()

    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.axon.add_args(parser)

    return bt.config(parser)


async def main():
    # Start session id
    logger.start_session()

    # Set dependencies
    config = get_config()

    db_client = DatabaseClient(db_path="new_validator.db", logger=logger)
    await db_client.migrate()

    bt_wallet = bt.wallet(config=config)
    db_operations = DatabaseOperations(db_client=db_client)
    api_client = IfGamesClient(env="test", logger=logger, bt_wallet=bt_wallet)

    # Set tasks
    pull_events_task = PullEvents(
        interval_seconds=50.0, page_size=50, db_operations=db_operations, api_client=api_client
    )

    resolve_events_task = ResolveEvents(
        interval_seconds=1800.0, db_operations=db_operations, api_client=api_client, logger=logger
    )

    # Set scheduler and add tasks
    scheduler = TasksScheduler(logger=logger)

    scheduler.add(task=pull_events_task)
    scheduler.add(task=resolve_events_task)

    # Start tasks
    await scheduler.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        exit(0)
