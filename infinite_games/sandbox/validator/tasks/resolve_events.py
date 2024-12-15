import aiohttp

from infinite_games.sandbox.validator.db.operations import DatabaseOperations
from infinite_games.sandbox.validator.if_games.client import IfGamesClient
from infinite_games.sandbox.validator.scheduler.task import AbstractTask


class ResolveEvents(AbstractTask):
    interval: float
    api_client: IfGamesClient
    db_operations: DatabaseOperations

    def __init__(
        self,
        interval_seconds: float,
        db_operations: DatabaseOperations,
        api_client: IfGamesClient,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        # Validate db_operations
        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        # Validate api_client
        if not isinstance(api_client, IfGamesClient):
            raise TypeError("api_client must be an instance of IfGamesClient.")

        self.interval = interval_seconds
        self.db_operations = db_operations
        self.api_client = api_client

    @property
    def name(self):
        return "resolve-events"

    @property
    def interval_seconds(self):
        return self.interval

    async def run(self):
        # Read pending events from db
        pending_events = await self.db_operations.get_pending_events()

        for event in pending_events:
            event_id = event[0]

            try:
                # Query
                event = await self.api_client.get_event(event_id=event_id)

                resolved = True if event.get("answer") is not None else False

                # Mark resolved
                if resolved:
                    await self.db_operations.resolve_event(event_id)

            except aiohttp.ClientResponseError as error:
                # Clear deleted events
                if error.status in [404, 410]:
                    await self.db_operations.delete_event(event_id=event_id)

                    continue

                raise error
