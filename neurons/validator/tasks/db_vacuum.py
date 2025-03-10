from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class DbVacuum(AbstractTask):
    interval: float
    db_operations: DatabaseOperations
    logger: InfiniteGamesLogger
    pages: int
    _first_run: bool

    def __init__(
        self,
        interval_seconds: float,
        db_operations: DatabaseOperations,
        logger: InfiniteGamesLogger,
        pages: int,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        # Validate db_operations
        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        # Validate logger
        if not isinstance(logger, InfiniteGamesLogger):
            raise TypeError("logger must be an instance of InfiniteGamesLogger.")

        # Validate pages
        if not isinstance(pages, int) or pages <= 0:
            raise ValueError("pages must be a positive integer.")

        self.interval = interval_seconds
        self.db_operations = db_operations
        self.logger = logger
        self.pages = pages
        self._first_run = True

    @property
    def name(self):
        return "vacuum-task"

    @property
    def interval_seconds(self):
        return self.interval

    async def run(self):
        if self._first_run:
            self._first_run = False

            return

        await self.db_operations.vacuum_database(self.pages)
