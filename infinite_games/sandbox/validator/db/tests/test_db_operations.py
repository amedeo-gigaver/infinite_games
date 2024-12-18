import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from infinite_games.sandbox.validator.db.client import Client
from infinite_games.sandbox.validator.db.operations import DatabaseOperations
from infinite_games.sandbox.validator.models.event import EventsModel, EventStatus
from infinite_games.sandbox.validator.utils.logger.logger import AbstractLogger


class TestDbOperations:
    @pytest.fixture(scope="function")
    async def db_client(self):
        temp_db = tempfile.NamedTemporaryFile(delete=False)
        db_path = temp_db.name
        temp_db.close()

        logger = MagicMock(spec=AbstractLogger)

        db_client = Client(db_path, logger)

        await db_client.migrate()

        return db_client

    @pytest.fixture
    async def db_operations(self, db_client):
        db_operations = DatabaseOperations(db_client=db_client)

        return db_operations

    async def test_delete_event(self, db_operations: DatabaseOperations, db_client: Client):
        event_id_to_keep = "event1"
        event_id_to_delete = "event2"

        events = [
            (
                "unique1",
                event_id_to_keep,
                "market1",
                "desc1",
                "2024-12-02",
                "2024-12-03",
                "outcome1",
                "status1",
                '{"key": "value"}',
                "2000-12-02T14:30:00+00:00",
                "2000-12-02T14:30:00+00:00",
                "2000-12-03T14:30:00+00:00",
            ),
            (
                "unique2",
                event_id_to_delete,
                "market1",
                "desc1",
                "2024-12-02",
                "2024-12-03",
                "outcome1",
                "status1",
                '{"key": "value"}',
                "2000-12-02T14:30:00+00:00",
                "2000-12-02T14:30:00+00:00",
                "2000-12-03T14:30:00+00:00",
            ),
        ]

        await db_operations.upsert_events(events)

        result = await db_operations.delete_event(event_id=event_id_to_delete)

        assert len(result) == 1
        assert result[0][0] == event_id_to_delete

        result = await db_client.many(
            """
                SELECT event_id FROM events
            """
        )

        assert len(result) == 1
        assert result[0][0] == event_id_to_keep

    async def test_get_last_event_from(self, db_operations: DatabaseOperations):
        created_at = "2000-12-02T14:30:00+00:00"

        events = [
            (
                "unique1",
                "event1",
                "market1",
                "desc1",
                "2024-12-02",
                "2024-12-03",
                "outcome1",
                "status1",
                '{"key": "value"}',
                created_at,
                "2024-12-03",
                "2024-12-04",
            )
        ]

        await db_operations.upsert_events(events)

        result = await db_operations.get_last_event_from()

        assert result == created_at

    async def test_get_last_event_from_no_events(self, db_operations: DatabaseOperations):
        result = await db_operations.get_last_event_from()

        assert result is None

    async def test_get_pending_events(self, db_operations: DatabaseOperations):
        pending_event_id = "event1"

        events = [
            (
                "unique1",
                pending_event_id,
                "market1",
                "desc1",
                "2024-12-02",
                "2024-12-03",
                None,
                EventStatus.PENDING,
                '{"key": "value"}',
                "2000-12-02T14:30:00+00:00",
                "2000-12-29T14:30:00+00:00",
                "2000-12-30T14:30:00+00:00",
            ),
            (
                "unique2",
                "event2",
                "market2",
                "desc2",
                "2024-12-03",
                "2024-12-04",
                "outcome2",
                EventStatus.SETTLED,
                '{"key": "value"}',
                "2012-12-02T14:30:00+00:00",
                "2000-12-29T14:30:00+00:00",
                "2000-12-30T14:30:00+00:00",
            ),
        ]

        await db_operations.upsert_events(events)

        result = await db_operations.get_pending_events()

        assert len(result) == 1
        assert result[0][0] == pending_event_id

    async def test_get_events_to_predict(self, db_operations: DatabaseOperations):
        event_to_predict_id = "event3"

        cutoff_now = datetime.now(timezone.utc).isoformat()
        cutoff_future = (datetime.now(timezone.utc) + timedelta(seconds=2)).isoformat()

        events = [
            (
                "unique1",
                "event1",
                "market1",
                "desc1",
                "2024-12-02",
                "2024-12-03",
                None,
                EventStatus.DISCARDED,
                '{"key": "value"}',
                "2000-12-02T14:30:00+00:00",
                "2000-12-30T14:30:00+00:00",
                "2000-12-31T14:30:00+00:00",
            ),
            (
                "unique2",
                "event2",
                "market2",
                "desc2",
                "2024-12-03",
                "2024-12-04",
                "outcome2",
                EventStatus.PENDING,
                '{"key": "value"}',
                "2012-12-02T14:30:00+00:00",
                cutoff_now,
                "2000-12-31T14:30:00+00:00",
            ),
            (
                "unique3",
                event_to_predict_id,
                "market3",
                "desc3",
                "2024-12-03",
                "2024-12-04",
                "outcome2",
                EventStatus.PENDING,
                '{"key": "value"}',
                "2012-12-02T14:30:00+00:00",
                cutoff_future,
                "2000-12-31T14:30:00+00:00",
            ),
            (
                "unique4",
                "event4",
                "market4",
                "desc2",
                "2024-12-03",
                "2024-12-04",
                "outcome2",
                EventStatus.SETTLED,
                '{"key": "value"}',
                "2012-12-02T14:30:00+00:00",
                "2000-12-30T14:30:00+00:00",
                "2000-12-31T14:30:00+00:00",
            ),
        ]

        await db_operations.upsert_events(events)

        result = await db_operations.get_events_to_predict()

        assert len(result) == 1
        assert result[0][0] == event_to_predict_id

    async def test_resolve_event(self, db_operations: DatabaseOperations):
        event_to_resolve_id = "event1"
        event_pending_id = "event2"

        events = [
            (
                "unique1",
                event_to_resolve_id,
                "market1",
                "desc1",
                "2024-12-02",
                "2024-12-03",
                None,
                EventStatus.PENDING,
                '{"key": "value"}',
                "2000-12-02T14:30:00+00:00",
                "2000-12-30T14:30:00+00:00",
                "2000-12-31T14:30:00+00:00",
            ),
            (
                "unique2",
                event_pending_id,
                "market2",
                "desc2",
                "2024-12-03",
                "2024-12-04",
                "outcome2",
                EventStatus.PENDING,
                '{"key": "value"}',
                "2012-12-02T14:30:00+00:00",
                "2000-12-30T14:30:00+00:00",
                "2000-12-31T14:30:00+00:00",
            ),
        ]

        await db_operations.upsert_events(events)

        await db_operations.resolve_event(event_to_resolve_id)

        result = await db_operations.get_pending_events()

        assert len(result) == 1
        assert result[0][0] == event_pending_id

    async def test_upsert_events(self, db_operations: DatabaseOperations, db_client: Client):
        events = [
            (
                "unique1",
                "event1",
                "market1",
                "desc1",
                "2024-12-02",
                "2024-12-03",
                "outcome1",
                "status1",
                '{"key": "value"}',
                "2000-12-02T14:30:00+00:00",
                "2000-01-01T14:30:00+00:00",
                "2000-01-02T14:30:00+00:00",
            ),
            (
                "unique2",
                "event2",
                "market2",
                "desc2",
                "2024-12-03",
                "2024-12-04",
                "outcome2",
                "status2",
                '{"key": "value"}',
                "2012-12-02T14:30:00+00:00",
                "2001-01-01T14:30:00+00:00",
                "2001-01-02T14:30:00+00:00",
            ),
        ]

        await db_operations.upsert_events(events)

        result = await db_client.many(
            """
                SELECT event_id, cutoff FROM events
            """
        )

        assert len(result) == 2

        # Assert event id
        assert result[0][0] == "event1"
        assert result[1][0] == "event2"

        # Assert cutoff
        assert result[0][1] == events[0][10]
        assert result[1][1] == events[1][10]

    async def test_upsert_no_events(self, db_operations: DatabaseOperations):
        """Test upsert not failing with empty list."""
        events = []

        await db_operations.upsert_events(events)

    async def test_upsert_predictions(self, db_operations: DatabaseOperations, db_client: Client):
        predictions = [
            (
                "unique_event_id_1",
                "minerHotkey_1",
                "minerUid_1",
                "1",
                10,
                1,
                1,
                "2000-12-02T14:30:00+00:00",
                1,
                1,
            ),
            (
                "unique_event_id_2",
                "minerHotkey_2",
                "minerUid_2",
                "1",
                10,
                1,
                1,
                "2000-12-02T14:30:00+00:00",
                1,
                1,
            ),
        ]

        # Insert
        await db_operations.upsert_predictions(predictions)

        result = await db_client.many(
            """
                SELECT unique_event_id, minerHotkey FROM predictions
            """
        )

        assert len(result) == 2

        # Assert event id
        assert result[0][0] == predictions[0][0]
        assert result[1][0] == predictions[1][0]

        # Assert minerHotkey
        assert result[0][1] == predictions[0][1]
        assert result[1][1] == predictions[1][1]

        # Upsert
        await db_operations.upsert_predictions(predictions)

        result = await db_client.many(
            """
                SELECT interval_count FROM predictions
            """
        )

        assert len(result) == 2

        # Assert interval_count
        assert result[0][0] == 2
        assert result[1][0] == 2

    async def test_get_events_for_scoring(self, db_operations: DatabaseOperations):
        expected_event_id = "event1"

        events = [
            EventsModel(
                unique_event_id="unique1",
                event_id=expected_event_id,
                market_type="market1",
                description="desc1",
                starts="2024-12-02",
                resolve_date="2024-12-03",
                outcome="outcome1",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
                end_date="2000-12-31T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="unique2",
                event_id="event2",
                market_type="market2",
                description="desc2",
                starts="2024-12-03",
                resolve_date="2024-12-04",
                outcome=None,
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
                created_at="2012-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
                end_date="2000-12-31T14:30:00+00:00",
            ),
        ]

        await db_operations.upsert_pydantic_events(events)

        result = await db_operations.get_events_for_scoring()

        assert len(result) == 1
        assert result[0].event_id == expected_event_id
        assert result[0].status == EventStatus.SETTLED

    async def test_get_predictions_for_scoring(
        self, db_operations: DatabaseOperations, db_client: Client
    ):
        expected_event_id = "_event1"

        predictions = [
            (
                "unique_event_id_1",
                "minerHotkey_1",
                "minerUid_1",
                "1",
                10,
                1,
                1,
                "2000-12-02T14:30:00+00:00",
                1,
                1,
            ),
            (
                expected_event_id,
                "minerHotkey_2",
                "minerUid_2",
                "1",
                10,
                1,
                1,
                "2000-12-02T14:30:00+00:00",
                1,
                1,
            ),
        ]

        await db_operations.upsert_predictions(predictions)

        all_predictions = await db_client.many(
            """
                SELECT unique_event_id FROM predictions ORDER BY unique_event_id
            """
        )
        assert len(all_predictions) == 2
        assert all_predictions[0][0] == expected_event_id
        assert all_predictions[1][0] == "unique_event_id_1"

        result = await db_operations.get_predictions_for_scoring(event_id=expected_event_id)

        assert len(result) == 1
        assert result[0].unique_event_id == expected_event_id
