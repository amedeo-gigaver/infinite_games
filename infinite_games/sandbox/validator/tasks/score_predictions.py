from infinite_games.sandbox.validator.db.operations import DatabaseOperations
from infinite_games.sandbox.validator.if_games.client import IfGamesClient
from infinite_games.sandbox.validator.scheduler.task import AbstractTask


class ScorePredictions(AbstractTask):
    interval: float
    page_size: int
    api_client: IfGamesClient
    db_operations: DatabaseOperations

    def __init__(
        self,
        interval_seconds: float,
        db_operations: DatabaseOperations,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        # Validate db_operations
        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        self.interval = interval_seconds
        self.db_operations = db_operations

    @property
    def name(self):
        return "score-predictions"

    @property
    def interval_seconds(self):
        return self.interval

    async def run(self):
        # pull predictions

        # score them

        # store scores
        pass
