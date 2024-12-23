from typing import Iterable

from infinite_games.sandbox.validator.db.client import Client
from infinite_games.sandbox.validator.models.event import EVENTS_FIELDS, EventsModel, EventStatus
from infinite_games.sandbox.validator.models.miner import MINERS_FIELDS, MinersModel
from infinite_games.sandbox.validator.models.prediction import PREDICTION_FIELDS, PredictionsModel
from infinite_games.sandbox.validator.utils.logger.logger import db_logger


class DatabaseOperations:
    __db_client: Client

    def __init__(self, db_client: Client):
        if not isinstance(db_client, Client):
            raise ValueError("Invalid db_client arg")

        self.__db_client = db_client

    async def delete_event(self, event_id: str) -> Iterable[tuple[str]]:
        return await self.__db_client.delete(
            """
                DELETE FROM events WHERE event_id = ? RETURNING event_id
            """,
            [event_id],
        )

    async def get_events_to_predict(self) -> Iterable[tuple[str]]:
        return await self.__db_client.many(
            """
                SELECT
                    event_id,
                    market_type,
                    description,
                    cutoff,
                    resolve_date,
                    end_date,
                    metadata
                FROM
                    events
                WHERE
                    status = ?
                    AND datetime(CURRENT_TIMESTAMP) < datetime(cutoff)
            """,
            parameters=[EventStatus.PENDING],
        )

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

    async def resolve_event(
        self, event_id: str, outcome: str, resolved_at: str
    ) -> Iterable[tuple[str]]:
        return await self.__db_client.update(
            """
                UPDATE
                    events
                SET
                    status = ?,
                    outcome = ?,
                    resolved_at = ?
                WHERE
                    event_id = ?
                RETURNING
                    event_id
            """,
            [EventStatus.SETTLED, outcome, resolved_at, event_id],
        )

    async def upsert_events(self, events: list[list[any]]) -> None:
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
                        cutoff,
                        end_date,
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
                        ?,
                        ?,
                        CURRENT_TIMESTAMP,
                        CURRENT_TIMESTAMP
                    )
                ON CONFLICT
                    (unique_event_id)
                DO NOTHING
            """,
            events,
        )

    async def upsert_predictions(self, predictions: list[list[any]]):
        return await self.__db_client.insert_many(
            """
                INSERT INTO predictions (
                    unique_event_id,
                    minerHotkey,
                    minerUid,
                    predictedOutcome,
                    interval_start_minutes,
                    interval_agg_prediction,
                    blocktime,
                    interval_count,
                    submitted
                )
                VALUES (
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    1,
                    CURRENT_TIMESTAMP
                )
                ON CONFLICT(unique_event_id,  interval_start_minutes, minerUid)
                DO UPDATE SET
                    interval_agg_prediction = (interval_agg_prediction * interval_count + ?) / (interval_count + 1),
                    interval_count = interval_count + 1
            """,
            predictions,
        )

    async def upsert_pydantic_events(self, events: list[EventsModel]) -> None:
        """Same as upsert_events but with pydantic models"""

        fields_to_insert = [
            field_name
            for field_name in EVENTS_FIELDS
            if field_name not in ("registered_date", "local_updated_at")
        ]
        placeholders = ", ".join(["?"] * len(fields_to_insert))
        columns = ", ".join(fields_to_insert)

        # Convert each event into a tuple of values in the same order as fields_to_insert
        event_tuples = [
            tuple(getattr(event, field_name) for field_name in fields_to_insert) for event in events
        ]

        sql = f"""
                INSERT INTO events
                    ({columns}, registered_date, local_updated_at)
                VALUES
                    ({placeholders}, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT
                    (unique_event_id)
                DO NOTHING
        """
        return await self.__db_client.insert_many(
            sql=sql,
            parameters=event_tuples,
        )

    async def get_events_for_scoring(self) -> list[EventsModel]:
        """
        Returns all events that were recently resolved and need to be scored
        """

        rows = await self.__db_client.many(
            f"""
                SELECT
                    {', '.join(EVENTS_FIELDS)}
                FROM events
                WHERE status = ?
                    AND outcome IS NOT NULL
                    AND processed = false
            """,
            parameters=[EventStatus.SETTLED],
            use_row_factory=True,
        )

        events = []
        for row in rows:
            try:
                event = EventsModel(**dict(row))
                events.append(event)
            except Exception:
                db_logger.exception("Error parsing event", extra={"row": row[0]})

        return events

    async def get_predictions_for_scoring(self, event_id: str) -> list[PredictionsModel]:
        rows = await self.__db_client.many(
            f"""
                SELECT
                    {', '.join(PREDICTION_FIELDS)}
                FROM predictions
                WHERE unique_event_id = ?
            """,
            parameters=(event_id,),
            use_row_factory=True,
        )

        predictions = []
        for row in rows:
            try:
                prediction = PredictionsModel(**dict(row))
                predictions.append(prediction)
            except Exception:
                db_logger.exception("Error parsing prediction", extra={"row": row[0]})

        return predictions

    async def get_miners_last_registration(self) -> list:
        rows = await self.__db_client.many(
            f"""
                WITH ranked AS (
                    SELECT
                        {', '.join(MINERS_FIELDS)},
                        ROW_NUMBER() OVER (
                            PARTITION BY miner_uid
                            ORDER BY registered_date DESC
                        ) AS rn
                    FROM miners t
                )
                SELECT
                    {', '.join(MINERS_FIELDS)}
                FROM ranked
                WHERE rn = 1
                ORDER BY miner_uid
            """,
            use_row_factory=True,
        )
        miners = []
        for row in rows:
            try:
                miner = MinersModel(**dict(row))
                miners.append(miner)
            except Exception:
                db_logger.exception("Error parsing miner", extra={"row": row[0]})

        return miners
