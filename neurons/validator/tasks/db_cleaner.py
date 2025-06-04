import asyncio

from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class DbCleaner(AbstractTask):
    interval: float
    db_operations: DatabaseOperations
    batch_size: int
    logger: InfiniteGamesLogger

    def __init__(
        self,
        interval_seconds: float,
        db_operations: DatabaseOperations,
        batch_size: int,
        logger: InfiniteGamesLogger,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        # Validate db_operations
        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        # Validate batch_size
        max_batch_size = 4000

        if not isinstance(batch_size, int) or batch_size <= 0 or batch_size > max_batch_size:
            raise ValueError(
                f"batch_size must be a positive integer equal or less than {max_batch_size}."
            )

        # Validate logger
        if not isinstance(logger, InfiniteGamesLogger):
            raise TypeError("logger must be an instance of InfiniteGamesLogger.")

        self.interval = interval_seconds
        self.db_operations = db_operations
        self.batch_size = batch_size
        self.logger = logger

    @property
    def name(self):
        return "db-cleaner"

    @property
    def interval_seconds(self):
        return self.interval

    async def run(self):
        # Delete predictions
        deleted_predictions = await self.db_operations.delete_predictions(self.batch_size)

        if len(deleted_predictions) > 0:
            self.logger.debug(
                "Predictions deleted", extra={"deleted_count": len(deleted_predictions)}
            )

        await asyncio.sleep(1)

        # Delete scores
        deleted_scores = await self.db_operations.delete_scores(self.batch_size)

        if len(deleted_scores) > 0:
            self.logger.debug("Scores deleted", extra={"deleted_count": len(deleted_scores)})

        await asyncio.sleep(1)

        # Delete reasonings
        deleted_reasonings = await self.db_operations.delete_reasonings(self.batch_size)

        if len(deleted_reasonings) > 0:
            self.logger.debug(
                "Reasonings deleted", extra={"deleted_count": len(deleted_reasonings)}
            )
