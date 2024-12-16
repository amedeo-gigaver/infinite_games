import json
import math
from datetime import datetime, timedelta, timezone

from infinite_games.sandbox.validator.db.operations import DatabaseOperations
from infinite_games.sandbox.validator.if_games.client import IfGamesClient
from infinite_games.sandbox.validator.models.event import EventStatus
from infinite_games.sandbox.validator.scheduler.task import AbstractTask


class PullEvents(AbstractTask):
    interval: float
    page_size: int
    api_client: IfGamesClient
    db_operations: DatabaseOperations

    def __init__(
        self,
        interval_seconds: float,
        db_operations: DatabaseOperations,
        api_client: IfGamesClient,
        page_size: int,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        # Validate db_operations
        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        # Validate api_client
        if not isinstance(api_client, IfGamesClient):
            raise TypeError("api_client must be an instance of IfGamesClient.")

        # Validate page_size
        if not isinstance(page_size, int) or page_size <= 0 or page_size > 500:
            raise ValueError("page_size must be a positive integer.")

        self.interval = interval_seconds
        self.db_operations = db_operations
        self.api_client = api_client
        self.page_size = page_size

    @property
    def name(self):
        return "pull-events"

    @property
    def interval_seconds(self):
        return self.interval

    async def run(self):
        # Pick up from where it left, get 'from' from latest db events
        events_from = await self.db_operations.get_last_event_from()

        if events_from:
            # Back track 1h
            events_from = (datetime.fromisoformat(events_from) - timedelta(hours=1)).timestamp()
            events_from = math.floor(events_from)
        else:
            events_from = 0

        offset = 0

        while True:
            # Query events in batches
            response = await self.api_client.get_events(events_from, offset, self.page_size)
            # TODO: data validation

            # Parse events
            items = response.get("items")
            parsed_events_for_insertion = [(self.parse_event(e)) for e in items]

            # Batch insert in the db
            if len(parsed_events_for_insertion) > 0:
                await self.db_operations.upsert_events(parsed_events_for_insertion)

            if len(items) < self.page_size:
                # Break if no more events
                break

            offset += self.page_size

    def parse_event(self, event: any):
        end_date_ts = event.get("end_date")
        start_date_ts = event.get("start_date")
        start_date = datetime.fromtimestamp(start_date_ts, tz=timezone.utc)

        created_at = datetime.fromtimestamp(event.get("created_at"), tz=timezone.utc)
        status = EventStatus.SETTLED if event.get("answer") is not None else EventStatus.PENDING
        truncated_market_type = "ifgames"

        cutoff = datetime.fromtimestamp(event.get("cutoff"), tz=timezone.utc)

        return (
            # unique_event_id
            f"{truncated_market_type}-{event['event_id']}",
            # event_id
            event["event_id"],
            # market_type
            truncated_market_type,
            # description
            event.get("title", "") + event.get("description", ""),
            # starts
            start_date,
            # resolve_date
            None,
            # outcome
            event["answer"],
            # status
            status,
            # metadata
            json.dumps(
                {
                    "market_type": event.get("market_type", "").lower(),
                    "cutoff": event.get("cutoff"),
                    "end_date": end_date_ts,
                }
            ),
            # created_at
            created_at,
            # cutoff
            cutoff,
        )
