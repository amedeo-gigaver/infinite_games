import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import ANY, MagicMock

import pytest

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.event import EventsModel, EventStatus
from neurons.validator.models.miner import MinersModel
from neurons.validator.models.prediction import PredictionExportedStatus
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class TestDbOperations:
    @pytest.fixture(scope="function")
    async def db_client(self):
        temp_db = tempfile.NamedTemporaryFile(delete=False)
        db_path = temp_db.name
        temp_db.close()

        logger = MagicMock(spec=InfiniteGamesLogger)

        db_client = DatabaseClient(db_path, logger)

        await db_client.migrate()

        return db_client

    @pytest.fixture
    async def db_operations(self, db_client):
        db_operations = DatabaseOperations(db_client=db_client)

        return db_operations

    async def test_delete_event(self, db_operations: DatabaseOperations, db_client: DatabaseClient):
        event_id_to_keep = "event1"
        event_id_to_delete = "event2"

        events = [
            (
                "unique1",
                event_id_to_keep,
                "truncated_market1",
                "market_1",
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
                "truncated_market1",
                "market_1",
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

    async def test_delete_predictions_orphan_prediction(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        events = [
            # No events - orphan prediction
        ]

        predictions = [
            (
                "no_event_orphan_prediction_id_1",
                "neuronHotkey_1",
                "neuronUid_1",
                "1",
                10,
                "1",
                1,
                "1",
            ),
            (
                "no_event_orphan_prediction_id_2",
                "neuronHotkey_1",
                "neuronUid_1",
                "1",
                10,
                "1",
                1,
                "1",
            ),
        ]

        await db_operations.upsert_events(events=events)
        await db_operations.upsert_predictions(predictions=predictions)

        # Mark first prediction as exported
        await db_client.update(
            """
                UPDATE
                    predictions
                SET
                    exported = ?
                WHERE
                    unique_event_id = ?
                """,
            [PredictionExportedStatus.EXPORTED, predictions[0][0]],
        )

        # delete predictions
        result = await db_operations.delete_predictions(batch_size=100)

        assert len(result) == 1
        assert result[0][0] == 1

        result = await db_client.many(
            """
                SELECT unique_event_id FROM predictions ORDER BY ROWID ASC
            """
        )

        # Should have deleted the orphan exported prediction
        assert len(result) == 1
        assert result[0][0] == predictions[1][0]

    async def test_delete_predictions_processed_unprocessed_events(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        events = [
            (
                "processed_event_prediction_id",
                "event1",
                "truncated_market1",
                "market_1",
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
                "unprocessed_event_prediction_id",
                "event2",
                "truncated_market2",
                "market_2",
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

        predictions = [
            (
                "processed_event_prediction_id",
                "neuronHotkey_2",
                "neuronUid_2",
                "1",
                10,
                "1",
                1,
                "1",
            ),
            (
                "unprocessed_event_prediction_id",
                "neuronHotkey_2",
                "neuronUid_2",
                "1",
                10,
                "1",
                1,
                "1",
            ),
        ]

        await db_operations.upsert_events(events=events)
        await db_operations.upsert_predictions(predictions=predictions)

        # Mark processed event as processed
        await db_client.update(
            """
                UPDATE
                    events
                SET
                    processed = ?
                WHERE
                    unique_event_id = ?
            """,
            [True, events[0][0]],
        )
        # Mark all predictions as exported but not exported one
        await db_client.update(
            """
                UPDATE
                    predictions
                SET
                    exported = ?
                """,
            [PredictionExportedStatus.EXPORTED],
        )

        # delete
        result = await db_operations.delete_predictions(batch_size=100)

        assert len(result) == 1
        assert result[0][0] == 1

        result = await db_client.many(
            """
                SELECT unique_event_id FROM predictions ORDER BY ROWID ASC
            """
        )

        # Should have deleted the processed event prediction
        assert len(result) == 1
        assert result[0][0] == "unprocessed_event_prediction_id"

    async def test_delete_predictions_discarded_event(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        events = [
            (
                "discarded_event_prediction_id",
                "event3",
                "truncated_market3",
                "market_3",
                "desc2",
                "2024-12-03",
                "2024-12-04",
                "outcome3",
                EventStatus.DISCARDED,
                '{"key": "value"}',
                "2012-12-02T14:30:00+00:00",
                "2001-01-01T14:30:00+00:00",
                "2001-01-02T14:30:00+00:00",
            ),
            (
                "pending_event_prediction_id",
                "event3",
                "truncated_market3",
                "market_3",
                "desc2",
                "2024-12-03",
                "2024-12-04",
                "outcome3",
                EventStatus.PENDING,
                '{"key": "value"}',
                "2012-12-02T14:30:00+00:00",
                "2001-01-01T14:30:00+00:00",
                "2001-01-02T14:30:00+00:00",
            ),
        ]

        predictions = [
            (
                "discarded_event_prediction_id",
                "neuronHotkey_2",
                "neuronUid_2",
                "1",
                10,
                "1",
                1,
                "1",
            ),
            (
                "pending_event_prediction_id",
                "neuronHotkey_2",
                "neuronUid_2",
                "1",
                10,
                "1",
                1,
                "1",
            ),
        ]

        await db_operations.upsert_events(events=events)
        await db_operations.upsert_predictions(predictions=predictions)

        # Mark all predictions as exported
        await db_client.update(
            """
                UPDATE
                    predictions
                SET
                    exported = ?
                """,
            [PredictionExportedStatus.EXPORTED],
        )

        # delete
        result = await db_operations.delete_predictions(batch_size=100)

        assert len(result) == 1
        assert result[0][0] == 1

        result = await db_client.many(
            """
                SELECT unique_event_id FROM predictions ORDER BY ROWID ASC
            """
        )

        assert len(result) == 1
        assert result[0][0] == "pending_event_prediction_id"

    async def test_delete_predictions_exported_unexported(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        events = [
            (
                "exported_prediction_event_id",
                "event1",
                "truncated_market1",
                "market_1",
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
                "not_exported_prediction_event_id",
                "event4",
                "truncated_market4",
                "market_4",
                "desc4",
                "2024-12-03",
                "2024-12-04",
                "outcome4",
                "status4",
                '{"key": "value"}',
                "2012-12-02T14:30:00+00:00",
                "2001-01-01T14:30:00+00:00",
                "2001-01-02T14:30:00+00:00",
            ),
        ]

        predictions = [
            (
                "exported_prediction_event_id",
                "neuronHotkey_1",
                "neuronUid_1",
                "1",
                10,
                "1",
                1,
                "1",
            ),
            (
                "not_exported_prediction_event_id",
                "neuronHotkey_2",
                "neuronUid_2",
                "1",
                10,
                "1",
                1,
                "1",
            ),
        ]

        await db_operations.upsert_events(events=events)
        await db_operations.upsert_predictions(predictions=predictions)

        # Mark all events as processed but unprocessed and discarded ones
        await db_client.update(
            """
                UPDATE
                    events
                SET
                    processed = ?
            """,
            [
                True,
            ],
        )

        # Mark all predictions as exported but not exported one
        await db_client.update(
            """
                UPDATE
                    predictions
                SET
                    exported = ?
                WHERE
                    unique_event_id NOT IN ('not_exported_prediction_event_id')
                """,
            [PredictionExportedStatus.EXPORTED],
        )

        # delete
        result = await db_operations.delete_predictions(batch_size=100)

        assert len(result) == 1
        assert result[0][0] == 1

        result = await db_client.many(
            """
                SELECT unique_event_id FROM predictions ORDER BY ROWID ASC
            """
        )

        # Should have unexported prediction left
        assert len(result) == 1
        assert result[0][0] == "not_exported_prediction_event_id"

    async def test_delete_predictions_batch_size(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        events = [
            (
                "event_id_1",
                "event1",
                "truncated_market1",
                "market_1",
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
                "event_id_2",
                "event2",
                "truncated_market2",
                "market_2",
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
            (
                "event_id_3",
                "event3",
                "truncated_market3",
                "market_3",
                "desc2",
                "2024-12-03",
                "2024-12-04",
                "outcome3",
                EventStatus.DISCARDED,
                '{"key": "value"}',
                "2012-12-02T14:30:00+00:00",
                "2001-01-01T14:30:00+00:00",
                "2001-01-02T14:30:00+00:00",
            ),
        ]

        predictions = [
            (
                "event_id_1",
                "neuronHotkey_1",
                "neuronUid_1",
                "1",
                10,
                "1",
                1,
                "1",
            ),
            (
                "event_id_2",
                "neuronHotkey_2",
                "neuronUid_2",
                "1",
                10,
                "1",
                1,
                "1",
            ),
            (
                "event_id_3",
                "neuronHotkey_2",
                "neuronUid_2",
                "1",
                10,
                "1",
                1,
                "1",
            ),
        ]

        await db_operations.upsert_events(events=events)
        await db_operations.upsert_predictions(predictions=predictions)

        # Mark all events as processed
        await db_client.update(
            """
                UPDATE
                    events
                SET
                    processed = ?
            """,
            [
                True,
            ],
        )
        # Mark all predictions as exported
        await db_client.update(
            """
                UPDATE
                    predictions
                SET
                    exported = ?
                """,
            [PredictionExportedStatus.EXPORTED],
        )

        # delete with batch size 2
        result = await db_operations.delete_predictions(batch_size=2)

        assert len(result) == 2

        # Check the rows are deleted in ASC order
        assert result[0][0] == 1
        assert result[1][0] == 2

        result = await db_client.many(
            """
                SELECT unique_event_id FROM predictions ORDER BY ROWID ASC
            """
        )

        # Should have 1 prediction left
        assert len(result) == 1
        assert result[0][0] == "event_id_3"

        # Delete the rest
        result = await db_operations.delete_predictions(batch_size=1000)

        assert len(result) == 1
        assert result[0][0] == 3

        result = await db_client.many(
            """
                SELECT unique_event_id FROM predictions ORDER BY ROWID ASC
            """
        )

        # Should have no predictions left
        assert len(result) == 0

    async def test_get_events_last_resolved_at(self, db_operations: DatabaseOperations):
        events = [
            (
                "unique1",
                "event1",
                "truncated_market1",
                "market_1",
                "desc1",
                "2024-12-02",
                "2024-12-03",
                "outcome1",
                EventStatus.PENDING,
                '{"key": "value"}',
                "1900-12-02T14:30:00+00:00",
                "2024-12-03",
                "2024-12-04",
            ),
            (
                "unique2",
                "event2",
                "truncated_market2",
                "market_2",
                "desc2",
                "2024-12-02",
                "2024-12-03",
                "outcome2",
                EventStatus.PENDING,
                '{"key": "value"}',
                "3000-12-02T14:30:00+00:00",
                "2024-12-03",
                "2024-12-04",
            ),
            (
                "unique3",
                "event3",
                "truncated_market3",
                "market_3",
                "desc3",
                "2024-12-02",
                "2024-12-03",
                "outcome3",
                EventStatus.PENDING,
                '{"key": "value"}',
                "1950-12-02T14:30:00+00:00",
                "2024-12-03",
                "2024-12-04",
            ),
        ]

        current_time = datetime.now(timezone.utc)
        current_time_iso = current_time.isoformat()
        future_time_iso = (current_time + timedelta(seconds=1)).isoformat()

        await db_operations.upsert_events(events)

        # Resolve events 1 and 3
        await db_operations.resolve_event(
            event_id="event1", outcome=1, resolved_at=current_time_iso
        )
        await db_operations.resolve_event(event_id="event3", outcome=1, resolved_at=future_time_iso)

        result = await db_operations.get_events_last_resolved_at()

        assert result == future_time_iso

    async def test_get_events_last_resolved_at_no_events(self, db_operations: DatabaseOperations):
        result = await db_operations.get_events_last_resolved_at()

        assert result is None

    async def test_get_events_pending_first_created_at(self, db_operations: DatabaseOperations):
        events = [
            (
                "unique1",
                "resolved_event",
                "truncated_market1",
                "market_1",
                "desc1",
                "2024-12-02",
                "2024-12-03",
                "outcome1",
                EventStatus.SETTLED,
                '{"key": "value"}',
                "1900-12-02T14:30:00+00:00",
                "2024-12-03",
                "2024-12-04",
            ),
            (
                "unique2",
                "pending_event_new",
                "truncated_market2",
                "market_2",
                "desc2",
                "2024-12-02",
                "2024-12-03",
                "outcome2",
                EventStatus.PENDING,
                '{"key": "value"}',
                "3000-12-02T14:30:00+00:00",
                "2024-12-03",
                "2024-12-04",
            ),
            (
                "unique3",
                "pending_event_old",
                "truncated_market3",
                "market_3",
                "desc3",
                "2024-12-02",
                "2024-12-03",
                "outcome3",
                EventStatus.PENDING,
                '{"key": "value"}',
                "1950-12-02T14:30:00+00:00",
                "2024-12-03",
                "2024-12-04",
            ),
        ]

        await db_operations.upsert_events(events)

        result = await db_operations.get_events_pending_first_created_at()

        assert result == "1950-12-02T14:30:00+00:00"

    async def test_get_events_pending_first_created_at_no_events(
        self, db_operations: DatabaseOperations
    ):
        result = await db_operations.get_events_pending_first_created_at()

        assert result is None

    async def test_get_last_event_from(self, db_operations: DatabaseOperations):
        created_at = "2000-12-02T14:30:00+00:00"

        events = [
            (
                "unique1",
                "event1",
                "truncated_market1",
                "market_1",
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

    async def test_get_events_to_predict(self, db_operations: DatabaseOperations):
        event_to_predict_id = "event3"

        cutoff_now = datetime.now(timezone.utc).isoformat()
        cutoff_future = (datetime.now(timezone.utc) + timedelta(seconds=2)).isoformat()

        events = [
            (
                "unique1",
                "event1",
                "truncated_market1",
                "market_1",
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
                "truncated_market2",
                "market_2",
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
                "truncated_market3",
                "market_3",
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
                "truncated_market4",
                "market_4",
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

    async def test_get_predictions_to_export(
        self, db_client: DatabaseClient, db_operations: DatabaseOperations
    ):
        events = [
            (
                "unique_event_id_1",
                "event_1",
                "truncated_market1",
                "market_1",
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
                "unique_event_id_2",
                "event_2",
                "truncated_market2",
                "market_2",
                "desc2",
                "2024-12-02",
                "2024-12-03",
                "outcome2",
                "status2",
                '{"key": "value"}',
                "2000-12-02T14:30:00+00:00",
                "2000-12-02T14:30:00+00:00",
                "2000-12-03T14:30:00+00:00",
            ),
            (
                "unique_event_id_3",
                "event_3",
                "truncated_market3",
                "market_3",
                "desc3",
                "2024-12-02",
                "2024-12-03",
                "outcome3",
                "status3",
                '{"key": "value"}',
                "2000-12-02T14:30:00+00:00",
                "2000-12-02T14:30:00+00:00",
                "2000-12-03T14:30:00+00:00",
            ),
            (
                "unique_event_id_4",
                "event_4",
                "truncated_market4",
                "market_4",
                "desc4",
                "2024-12-02",
                "2024-12-03",
                "outcome4",
                "status4",
                '{"key": "value"}',
                "2000-12-02T14:30:00+00:00",
                "2000-12-02T14:30:00+00:00",
                "2000-12-03T14:30:00+00:00",
            ),
        ]

        predictions = [
            (
                "unique_event_id_1",
                "neuronHotkey_1",
                "neuronUid_1",
                "1",
                # interval_start_minutes
                10,
                "1",
                1,
                "1",
            ),
            (
                "unique_event_id_2",
                "neuronHotkey_2",
                "neuronUid_2",
                "1",
                # interval_start_minutes
                10,
                "1",
                1,
                "1",
            ),
            (
                "unique_event_id_3",
                "neuronHotkey_3",
                "neuronUid_3",
                "1",
                # interval_start_minutes
                10,
                "1",
                1,
                "1",
            ),
            (
                "unique_event_id_4",
                "neuronHotkey_4",
                "neuronUid_4",
                "1",
                # interval_start_minutes
                # This prediction wont be ready to be exported
                11,
                "1",
                1,
                "1",
            ),
        ]

        await db_operations.upsert_events(events=events)

        await db_operations.upsert_predictions(predictions)

        # Mark one prediction as exported
        await db_client.update(
            """
                UPDATE predictions SET exported = ? WHERE unique_event_id = ?
            """,
            [PredictionExportedStatus.EXPORTED, "unique_event_id_2"],
        )

        current_interval_minutes = 11

        result = await db_operations.get_predictions_to_export(
            current_interval_minutes=current_interval_minutes, batch_size=1
        )

        assert len(result) == 1
        assert result[0][1] == "unique_event_id_1"

        result = await db_operations.get_predictions_to_export(
            current_interval_minutes=current_interval_minutes, batch_size=20
        )

        assert len(result) == 2

    async def test_mark_predictions_as_exported(
        self, db_client: DatabaseClient, db_operations: DatabaseOperations
    ):
        predictions = [
            (
                "unique_event_id_1",
                "neuronHotkey_1",
                "neuronUid_1",
                "1",
                10,
                "1",
                1,
                "1",
            ),
            (
                "unique_event_id_2",
                "neuronHotkey_2",
                "neuronUid_2",
                "1",
                10,
                "1",
                1,
                "1",
            ),
            (
                "unique_event_id_3",
                "neuronHotkey_3",
                "neuronUid_3",
                "1",
                10,
                "1",
                1,
                "1",
            ),
        ]

        await db_operations.upsert_predictions(predictions=predictions)

        result = await db_operations.mark_predictions_as_exported(ids=["2"])

        # Updated row id is correct
        assert result[0][0] == 2

        result = await db_client.many(
            """
                SELECT
                    ROWID,
                    unique_event_id,
                    exported
                FROM
                    predictions
            """
        )

        assert result[1][1] == "unique_event_id_2"
        assert result[0][2] == PredictionExportedStatus.NOT_EXPORTED
        assert result[1][2] == PredictionExportedStatus.EXPORTED
        assert result[2][2] == PredictionExportedStatus.NOT_EXPORTED

    async def test_resolve_event(
        self, db_client: DatabaseClient, db_operations: DatabaseOperations
    ):
        event_id = "event1"
        outcome = 1
        resolved_at = "2000-12-31T14:30:00+00:00"

        events = [
            (
                "unique1",
                event_id,
                "truncated_market1",
                "market_1",
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
        ]

        await db_operations.upsert_events(events)

        await db_operations.resolve_event(
            event_id=event_id,
            outcome=outcome,
            resolved_at=resolved_at,
        )

        result = await db_client.many(
            """
                SELECT event_id, status, outcome, resolved_at, local_updated_at FROM events
            """
        )

        assert len(result) == 1
        assert result[0][0] == event_id
        assert result[0][1] == str(EventStatus.SETTLED.value)
        assert result[0][2] == str(outcome)
        assert result[0][3] == resolved_at
        assert isinstance(result[0][4], str)

    async def test_resolve_event_not_resolving_again(
        self, db_client: DatabaseClient, db_operations: DatabaseOperations
    ):
        event_id = "event1"
        outcome = 1
        resolved_at = "2000-12-31T14:30:00+00:00"

        events = [
            (
                "unique1",
                event_id,
                "truncated_market1",
                "market_1",
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
        ]

        await db_operations.upsert_events(events)

        await db_operations.resolve_event(
            event_id=event_id,
            outcome=outcome,
            resolved_at=resolved_at,
        )

        outcome_2 = 0
        resolved_at_2 = "1900-12-31T14:30:00+00:00"

        await db_operations.resolve_event(
            event_id=event_id,
            outcome=outcome_2,
            resolved_at=resolved_at_2,
        )

        result = await db_client.many(
            """
                SELECT event_id, status, outcome, resolved_at, local_updated_at FROM events
            """
        )

        # Assert event not resolved with new values
        assert len(result) == 1
        assert result[0][0] == event_id
        assert result[0][2] == str(outcome)
        assert result[0][3] == resolved_at

    async def test_upsert_events(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        events = [
            (
                "unique1",
                "event1",
                "truncated_market1",
                "market_1",
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
                "truncated_market2",
                "market_2",
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
        assert result[0][1] == events[0][11]
        assert result[1][1] == events[1][11]

    async def test_upsert_no_events(self, db_operations: DatabaseOperations):
        """Test upsert not failing with empty list."""
        events = []

        await db_operations.upsert_events(events)

    async def test_upsert_miners(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        miners = [
            (
                "uid1",
                "hotkey1",
                "ip1",
                "2000-12-02T14:30:00+00:00",
                1,
                False,
                True,
                "ip1",
                1,
            ),
            (
                "uid2",
                "hotkey2",
                "ip2",
                "2000-12-02T14:30:00+00:00",
                2,
                True,
                False,
                "ip2",
                2,
            ),
        ]

        await db_operations.upsert_miners(miners)

        result = await db_client.many(
            """
                SELECT
                    miner_hotkey,
                    miner_uid,
                    registered_date,
                    last_updated,
                    is_validating,
                    validator_permit
                FROM
                    miners
            """
        )

        assert len(result) == 2

        print(result)
        # Assert miner hotkey
        assert result[0][0] == "hotkey1"
        assert result[1][0] == "hotkey2"

        # Assert miner uid
        assert result[0][1] == "uid1"
        assert result[1][1] == "uid2"

        # Assert registered_date
        assert result[0][2] == "2000-12-02T14:30:00+00:00"
        assert result[1][2] == "2000-12-02T14:30:00+00:00"

        # Assert last_registration
        assert result[0][3] == ANY
        assert result[1][3] == ANY

        # Assert is_validating
        assert result[0][4] == 0
        assert result[1][4] == 1

        # Assert validator_permit
        assert result[0][5] == 1
        assert result[1][5] == 0

        # TODO: Test update fields on conflict

    async def test_upsert_predictions(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        interval_start_minutes = 5
        block = 1

        neuron_1_answer = "1"
        neuron_2_answer = "0.5"

        predictions = [
            (
                "unique_event_id_1",
                "neuronHotkey_1",
                "neuronUid_1",
                neuron_1_answer,
                interval_start_minutes,
                neuron_1_answer,
                block,
                neuron_1_answer,
            ),
            (
                "unique_event_id_2",
                "neuronHotkey_2",
                "neuronUid_2",
                neuron_2_answer,
                interval_start_minutes,
                neuron_2_answer,
                block,
                neuron_2_answer,
            ),
        ]

        # Insert
        await db_operations.upsert_predictions(predictions)

        result = await db_client.many(
            """
                SELECT unique_event_id, minerHotkey, interval_agg_prediction, interval_count  FROM predictions
            """
        )

        assert len(result) == 2

        # Assert event id
        assert result[0][0] == predictions[0][0]
        assert result[1][0] == predictions[1][0]

        # Assert neuron hotkey
        assert result[0][1] == predictions[0][1]
        assert result[1][1] == predictions[1][1]

        # Assert interval_agg_prediction
        assert result[0][2] == float(neuron_1_answer)
        assert result[1][2] == float(neuron_2_answer)

        # Assert interval_count
        assert result[0][3] == 1
        assert result[1][3] == 1

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
                market_type="truncated_market1",
                event_type="market1",
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
                market_type="truncated_market2",
                event_type="market2",
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
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        expected_event_id = "_event1"

        predictions = [
            (
                "unique_event_id_1",
                "neuronHotkey_1",
                "neuronUid_1",
                "1",
                10,
                "1",
                1,
                "1",
            ),
            (
                expected_event_id,
                "neuronHotkey_2",
                "neuronUid_2",
                "1",
                10,
                "1",
                1,
                "1",
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

    async def test_get_miners_last_registration(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        miner_1 = MinersModel(
            miner_hotkey="hotkey1",
            miner_uid="uid1",
            registered_date=datetime(2024, 1, 1, 10, 0, 0),
            is_validating=False,
            validator_permit=False,
        )

        miner_2 = MinersModel(
            miner_hotkey="hotkey2",
            miner_uid="uid2",
            registered_date=datetime(2024, 1, 1, 10, 0, 0),
            is_validating=False,
            validator_permit=True,
        )

        miner_1_replaced = MinersModel(
            miner_hotkey="hotkey2",
            miner_uid="uid1",
            registered_date=datetime(2024, 1, 1, 11, 0, 1),
            is_validating=True,
            validator_permit=True,
        )

        miners = [miner_1, miner_2, miner_1_replaced]
        await db_client.insert_many(
            """
            INSERT INTO miners (miner_hotkey, miner_uid, registered_date, is_validating, validator_permit)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    m.miner_hotkey,
                    m.miner_uid,
                    m.registered_date,
                    m.is_validating,
                    m.validator_permit,
                )
                for m in miners
            ],
        )

        all_miners = await db_client.many(
            """
                SELECT miner_uid, miner_hotkey, registered_date FROM miners
                ORDER BY miner_uid, registered_date
            """
        )

        assert len(all_miners) == 3
        assert all_miners[0][0] == miner_1.miner_uid
        assert all_miners[0][1] == miner_1.miner_hotkey
        assert all_miners[1][1] == miner_1_replaced.miner_hotkey
        assert datetime.fromisoformat(all_miners[1][2]) == miner_1_replaced.registered_date
        assert all_miners[2][0] == miner_2.miner_uid

        result = await db_operations.get_miners_last_registration()
        assert len(result) == 2
        assert result[0].miner_uid == miner_1_replaced.miner_uid
        assert result[0].miner_hotkey == miner_1_replaced.miner_hotkey
        assert result[0].registered_date == miner_1_replaced.registered_date
        assert result[1].miner_uid == miner_2.miner_uid
        assert result[1].miner_hotkey == miner_2.miner_hotkey
        assert result[1].registered_date == miner_2.registered_date

    async def test_mark_event_as_processed(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        unique_event_id = "unique_event1"

        events = [
            (
                unique_event_id,
                "event1",
                "truncated_market1",
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
                "unique_event2",
                "event2",
                "truncated_market2",
                "market2",
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
        ]

        await db_operations.upsert_events(events)

        await db_operations.mark_event_as_processed(unique_event_id=unique_event_id)

        result = await db_client.many(
            """
                SELECT unique_event_id, processed, exported FROM events ORDER BY unique_event_id
            """
        )

        assert len(result) == 2
        assert result[0][0] == unique_event_id
        assert result[0][1] == 1
        assert result[0][2] == 0
        assert result[1][0] == "unique_event2"
        assert result[1][1] == 0
        assert result[1][2] == 0

    async def test_mark_event_as_exported(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        unique_event_id = "unique_event1"

        events = [
            (
                unique_event_id,
                "event1",
                "truncated_market1",
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
                "unique_event2",
                "event2",
                "truncated_market2",
                "market2",
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
        ]

        await db_operations.upsert_events(events)

        await db_operations.mark_event_as_exported(unique_event_id=unique_event_id)

        result = await db_client.many(
            """
                SELECT unique_event_id, exported FROM events ORDER BY unique_event_id
            """
        )

        assert len(result) == 2
        assert result[0][0] == unique_event_id
        assert result[0][1] == 1
        assert result[1][0] == "unique_event2"
        assert result[1][1] == 0
