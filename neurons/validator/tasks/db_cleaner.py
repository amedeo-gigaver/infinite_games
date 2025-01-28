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
        if not isinstance(batch_size, int) or batch_size <= 0 or batch_size > 2000:
            raise ValueError("batch_size must be a positive integer.")

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
        deleted = await self.db_operations.delete_predictions(self.batch_size)

        if len(deleted) > 0:
            self.logger.debug("Predictions deleted", extra={"deleted_count": len(deleted)})
