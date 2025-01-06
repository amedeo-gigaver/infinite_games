import asyncio
import os

from bittensor import Dendrite, Subtensor
from bittensor_wallet import Wallet

from infinite_games.sandbox.validator.db.client import DatabaseClient
from infinite_games.sandbox.validator.db.operations import DatabaseOperations
from infinite_games.sandbox.validator.if_games.client import IfGamesClient
from infinite_games.sandbox.validator.scheduler.tasks_scheduler import TasksScheduler
from infinite_games.sandbox.validator.tasks.export_predictions import ExportPredictions
from infinite_games.sandbox.validator.tasks.pull_events import PullEvents
from infinite_games.sandbox.validator.tasks.query_miners import QueryMiners
from infinite_games.sandbox.validator.tasks.resolve_events import ResolveEvents
from infinite_games.sandbox.validator.tasks.score_predictions import ScorePredictions
from infinite_games.sandbox.validator.utils.config import get_config
from infinite_games.sandbox.validator.utils.logger.logger import logger


async def main():
    # Start session id
    logger.start_session()

    # Force torch
    os.environ["USE_TORCH"] = "1"

    # Set dependencies
    config = get_config()

    # Bittensor stuff
    bt_netuid = config.get("netuid")
    bt_network = config.get("subtensor").get("network")
    bt_wallet = Wallet(config=config)
    bt_dendrite = Dendrite(wallet=bt_wallet)
    bt_subtensor = Subtensor(config=config)
    bt_metagraph = bt_subtensor.metagraph(netuid=bt_netuid, lite=True)

    validator_hotkey = bt_wallet.hotkey.ss58_address
    validator_uid = bt_metagraph.hotkeys.index(validator_hotkey)

    db_client = DatabaseClient(db_path="new_validator.db", logger=logger)

    db_operations = DatabaseOperations(db_client=db_client)

    env: IfGamesClient.EnvType = "prod" if bt_network == "finney" else "test"
    api_client = IfGamesClient(env=env, logger=logger, bt_wallet=bt_wallet)

    # Migrate db
    await db_client.migrate()

    # Set tasks
    pull_events_task = PullEvents(
        interval_seconds=50.0, page_size=50, db_operations=db_operations, api_client=api_client
    )

    resolve_events_task = ResolveEvents(
        interval_seconds=1800.0, db_operations=db_operations, api_client=api_client, logger=logger
    )

    query_miners_task = QueryMiners(
        interval_seconds=180.0,
        db_operations=db_operations,
        dendrite=bt_dendrite,
        metagraph=bt_metagraph,
        logger=logger,
    )

    export_predictions_task = ExportPredictions(
        interval_seconds=180.0,
        db_operations=db_operations,
        api_client=api_client,
        batch_size=200,
        validator_uid=validator_uid,
        validator_hotkey=validator_hotkey,
    )

    # TODO: add the logger to the ScorePredictions object
    score_predictions_task = ScorePredictions(
        interval_seconds=300.0,
        db_operations=db_operations,
        api_client=api_client,
        metagraph=bt_metagraph,
        config=config,
        subtensor=bt_subtensor,
        wallet=bt_wallet,
    )

    # Set scheduler and add tasks
    scheduler = TasksScheduler(logger=logger)

    scheduler.add(task=pull_events_task)
    scheduler.add(task=resolve_events_task)
    scheduler.add(task=query_miners_task)
    scheduler.add(task=export_predictions_task)
    scheduler.add(task=score_predictions_task)

    # Start tasks
    await scheduler.start()

    logger.info(
        "Validator started",
        extra={"validator_uid": validator_uid, "validator_hotkey": validator_hotkey},
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        exit(0)
    except Exception:
        logger.exception("Unexpected error")
        exit(1)
