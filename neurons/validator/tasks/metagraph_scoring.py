from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.logger.logger import InfiniteGamesLogger

# how many previous events to consider for the moving average
MOVING_AVERAGE_EVENTS = 99


class MetagraphScoring(AbstractTask):
    interval: float
    page_size: int
    db_operations: DatabaseOperations
    logger: InfiniteGamesLogger

    def __init__(
        self,
        interval_seconds: float,
        page_size: int,
        db_operations: DatabaseOperations,
        logger: InfiniteGamesLogger,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        # Validate db_operations
        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        self.interval = interval_seconds
        self.page_size = page_size
        self.db_operations = db_operations

        self.errors_count = 0
        self.logger = logger

    @property
    def name(self):
        return "metagraph-scoring"

    @property
    def interval_seconds(self):
        return self.interval

    async def run(self):
        events_to_score = await self.db_operations.get_events_for_metagraph_scoring(
            max_events=self.page_size
        )
        if not events_to_score:
            self.logger.debug("No events to calculate metagraph scores.")
        else:
            self.logger.debug(
                "Found events to calculate metagraph scores.",
                extra={"n_events": len(events_to_score)},
            )

            for event in events_to_score:
                self.logger.debug(
                    "Processing event for metagraph scoring.",
                    extra={"event_id": event["event_id"]},
                )

                try:
                    res = await self.db_operations.set_metagraph_peer_scores(
                        event["event_id"], n_events=MOVING_AVERAGE_EVENTS
                    )
                    if res == []:
                        self.logger.debug(
                            "Metagraph scores calculated successfully.",
                            extra={"event_id": event["event_id"]},
                        )
                    else:
                        raise Exception("Error calculating metagraph scores.")
                except Exception:
                    self.errors_count += 1
                    self.logger.exception(
                        "Error calculating metagraph scores.",
                        extra={"event_id": event["event_id"]},
                    )

        self.logger.debug(
            "Metagraph scoring task completed.",
            extra={"errors_count": self.errors_count},
        )

        self.errors_count = 0
