from typing import Iterable

from infinite_games.sandbox.validator.db.client import Client
from infinite_games.sandbox.validator.models.event import EventStatus


class DatabaseOperations:
    __db_client: Client

    def __init__(self, db_client: Client):
        if not isinstance(db_client, Client):
            raise ValueError("Invalid db_client arg")

        self.__db_client = db_client

    async def get_last_event_from(self) -> str | None:
        row = await self.__db_client.one(
            """
                SELECT MAX(created_at) FROM events
            """
        )

        if row is not None:
            return row[0]

    async def get_pending_events(self) -> Iterable[tuple[str]]:
        # TODO limit to events that passed cutoff
        return await self.__db_client.many(
            """
                SELECT event_id FROM events WHERE status = ?
            """,
            parameters=[EventStatus.PENDING],
        )

    async def delete_event(self, event_id: str) -> Iterable[tuple[str]]:
        return await self.__db_client.delete(
            """
                DELETE FROM events WHERE event_id = ? RETURNING event_id
            """,
            [event_id],
        )

    async def resolve_event(self, event_id: str) -> Iterable[tuple[str]]:
        return await self.__db_client.update(
            """
                UPDATE events SET status = ? WHERE event_id = ? RETURNING event_id
            """,
            [EventStatus.SETTLED, event_id],
        )

    async def upsert_events(self, events: list[any]) -> None:
        return await self.__db_client.insert_many(
            """
                INSERT INTO events
                    (
                        unique_event_id,
                        event_id,
                        market_type,
                        description,
                        starts,
                        resolve_date,
                        outcome,
                        status,
                        metadata,
                        created_at,
                        registered_date,
                        local_updated_at
                    )
                VALUES
                    (
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        CURRENT_TIMESTAMP,
                        CURRENT_TIMESTAMP
                    )
                ON CONFLICT
                    (unique_event_id)
                DO UPDATE
                    set outcome = EXCLUDED.market_type
            """,
            events,
        )
