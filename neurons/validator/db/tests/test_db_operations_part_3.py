import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import ANY

import pytest

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.db.tests.test_utils import TestDbOperationsBase
from neurons.validator.models.event import EventsModel, EventStatus
from neurons.validator.models.reasoning import ReasoningModel
from neurons.validator.models.score import ScoresModel


class TestDbOperationsPart3(TestDbOperationsBase):
    async def test_upsert_reasonings(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test basic upsert of reasonings"""
        reasonings = [
            ReasoningModel(
                event_id="event1",
                miner_uid=1,
                miner_hotkey="hotkey1",
                reasoning="Test reasoning 1",
                exported=False,
            ),
            ReasoningModel(
                event_id="event2",
                miner_uid=2,
                miner_hotkey="hotkey2",
                reasoning="Test reasoning 2",
                exported=True,
            ),
        ]

        await db_operations.upsert_reasonings(reasonings)

        # Verify the reasonings were inserted
        rows = await db_client.many(
            """
                SELECT
                    event_id,
                    miner_uid,
                    miner_hotkey,
                    reasoning,
                    exported,
                    created_at,
                    updated_at
                FROM
                    reasoning
                ORDER BY
                    ROWID ASC
            """
        )

        assert len(rows) == 2
        assert rows[0] == (
            "event1",
            1,
            "hotkey1",
            "Test reasoning 1",
            0,
            ANY,
            ANY,
        )
        assert rows[1] == ("event2", 2, "hotkey2", "Test reasoning 2", 1, ANY, ANY)

    async def test_upsert_reasonings_update_existing(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test updating existing reasonings"""
        # First insert
        initial_reasonings = [
            ReasoningModel(
                event_id="event1",
                miner_uid=1,
                miner_hotkey="hotkey1",
                reasoning="Initial reasoning",
                exported=False,
            )
        ]

        await db_operations.upsert_reasonings(initial_reasonings)

        initial_rows = await db_client.many(
            """
                SELECT
                    event_id,
                    miner_uid,
                    miner_hotkey,
                    reasoning,
                    exported,
                    created_at,
                    updated_at
                FROM
                    reasoning
            """
        )

        assert len(initial_rows) == 1
        assert initial_rows[0] == ("event1", 1, "hotkey1", "Initial reasoning", 0, ANY, ANY)

        # Update the same reasoning
        updated_reasonings = [
            ReasoningModel(
                event_id="event1",
                miner_uid=1,
                miner_hotkey="hotkey1",
                reasoning="Updated reasoning",
                exported=True,
            )
        ]

        # Delay upsert for updated_at to be different
        await asyncio.sleep(1)

        await db_operations.upsert_reasonings(updated_reasonings)

        # Verify the reasoning was updated
        final_rows = await db_client.many(
            """
                SELECT
                    event_id,
                    miner_uid,
                    miner_hotkey,
                    reasoning,
                    exported,
                    created_at,
                    updated_at
                FROM
                    reasoning
            """
        )

        assert len(final_rows) == 1
        # Exported is not updated
        assert final_rows[0] == ("event1", 1, "hotkey1", "Updated reasoning", 0, ANY, ANY)

        # Verify created_at and updated_at timestamps
        assert initial_rows[0][5] == final_rows[0][5]
        assert initial_rows[0][6] < final_rows[0][6]

    async def test_upsert_reasonings_empty_list(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test upserting an empty list of reasonings"""
        # Attempt to upsert an empty list
        await db_operations.upsert_reasonings([])

        # Verify no reasonings were inserted
        rows = await db_client.many(
            """
                SELECT COUNT(*) FROM reasoning
            """
        )

        assert rows[0][0] == 0

    async def test_get_community_train_dataset(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test getting community train dataset"""
        unique_event_id_1 = "unique_event_id_1"
        unique_event_id_2 = "unique_event_id_2"
        now = datetime.now(timezone.utc)
        now_minus_4h = now - timedelta(hours=4)

        scores = [
            ScoresModel(
                event_id=unique_event_id_1,
                miner_uid=i,
                miner_hotkey=f"hk{i}",
                prediction=1 - i * 0.01,
                event_score=0.5,
                spec_version=1,
            )
            for i in range(50)
        ] + [
            ScoresModel(
                event_id=unique_event_id_2,
                miner_uid=i,
                miner_hotkey=f"hk{i}",
                prediction=i * 0.01,
                event_score=0.5,
                spec_version=1,
            )
            for i in range(50)
        ]

        await db_operations.insert_peer_scores(scores)

        await db_client.update(
            "UPDATE scores SET processed = 1, metagraph_score = miner_uid * 0.01",
        )
        # move the first event to 4h ago to be in the previous batch
        await db_client.update(
            f"""UPDATE scores SET created_at = '{now_minus_4h.isoformat()}'"""
            + f""" WHERE event_id = '{unique_event_id_1}'""",
        )

        events = [
            EventsModel(
                unique_event_id=unique_event_id_1,
                event_id=unique_event_id_1,
                market_type="market_type1",
                event_type="type1",
                description="Some event",
                outcome="1",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                resolved_at=now.isoformat(),
            ),
            EventsModel(
                unique_event_id=unique_event_id_2,
                event_id=unique_event_id_2,
                market_type="market_type2",
                event_type="type2",
                description="Some event",
                outcome="0",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                resolved_at=now.isoformat(),
            ),
        ]

        await db_operations.upsert_pydantic_events(events)

        dataset = await db_operations.get_community_train_dataset()
        assert len(dataset) == 100

        for i in range(0, 100, 2):
            assert dataset[i]["event_id"] == unique_event_id_2
            assert dataset[i + 1]["event_id"] == unique_event_id_1
            assert dataset[i]["miner_uid"] == i // 2
            assert dataset[i + 1]["miner_uid"] == i // 2
            assert dataset[i]["miner_rank"] == 50 - i // 2
            assert dataset[i + 1]["miner_rank"] == 50 - i // 2
            assert dataset[i]["prev_batch_miner_rank"] == 50 - i // 2
            assert dataset[i + 1]["prev_batch_miner_rank"] == 256
            assert dataset[i]["prev_metagraph_score"] == pytest.approx(i // 2 * 0.01)
            assert dataset[i + 1]["prev_metagraph_score"] == pytest.approx(1e-6)
            assert dataset[i]["event_rank"] == 1
            assert dataset[i + 1]["event_rank"] == 2
            assert datetime.fromisoformat(dataset[i]["scoring_batch"]) - datetime.fromisoformat(
                dataset[i + 1]["scoring_batch"]
            ) == timedelta(hours=4)
            assert dataset[i]["batch_rank"] == 1
            assert dataset[i + 1]["batch_rank"] == 2
            assert dataset[i]["event_batch_rank"] == 1
            assert dataset[i + 1]["event_batch_rank"] == 1
            assert dataset[i]["outcome"] == 0
            assert dataset[i + 1]["outcome"] == 1
            assert dataset[i]["agg_prediction"] == i // 2 * 0.01
            assert dataset[i + 1]["agg_prediction"] == 1 - i // 2 * 0.01
            assert dataset[i]["metagraph_score"] == i // 2 * 0.01
            assert dataset[i + 1]["metagraph_score"] == i // 2 * 0.01

    async def test_save_community_predictions_model(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test saving community prediction model"""
        name = "test_model"
        model_blob = "dummy_model_blob"

        await db_operations.save_community_predictions_model(name=name, model_blob=model_blob)

        rows = await db_client.many(
            "SELECT * FROM models",
            use_row_factory=True,
        )

        assert len(rows) == 1
        assert rows[0]["name"] == name
        assert rows[0]["model_blob"] == model_blob
        assert rows[0]["other_data"] is None
        assert rows[0]["created_at"] is not None

    async def test_get_community_predictions_model(self, db_operations: DatabaseOperations):
        """Test getting community prediction model"""
        name = "test_model"
        model_blob = "dummy_model_blob"

        await db_operations.save_community_predictions_model(name=name, model_blob=model_blob)

        loaded_model_blob = await db_operations.get_community_predictions_model(name=name)

        assert loaded_model_blob == model_blob

    async def test_get_community_inference_dataset(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        unique_event_id_1 = "unique_event_id_1"
        unique_event_id_2 = "unique_event_id_2"
        unique_event_ids = [unique_event_id_1, unique_event_id_2, "fake_unique_event_id"]

        inference_data = await db_operations.get_community_inference_dataset(
            unique_event_ids=unique_event_ids, interval_start_minutes=1, top_n_ranks=100
        )

        # No predictions yet
        assert len(inference_data) == 0

        events = [
            EventsModel(
                unique_event_id=unique_event_id_1,
                event_id="event_id_1",
                market_type="truncated_market",
                event_type="market",
                description="desc",
                outcome="0",
                status=3,
                metadata='{"key": "value"}',
                created_at="2024-12-02T14:30:00+00:00",
                cutoff="2024-12-26T11:30:00+00:00",
                resolved_at="2024-12-30T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id=unique_event_id_2,
                event_id="event_id_2",
                market_type="truncated_market",
                event_type="market",
                description="desc",
                outcome="0",
                status=3,
                metadata='{"key": "value"}',
                created_at="2024-12-02T14:30:00+00:00",
                cutoff="2024-12-26T11:30:00+00:00",
                resolved_at="2024-12-30T14:30:00+00:00",
            ),
        ]

        await db_operations.upsert_pydantic_events(events=events)

        scores = [
            ScoresModel(
                event_id="event_id_1",
                miner_uid=i,
                miner_hotkey=f"hk{i}",
                prediction=0.5,
                event_score=0.5,
                spec_version=1,
            )
            for i in range(50)
        ] + [
            ScoresModel(
                event_id="event_id_1",
                miner_uid=i,
                miner_hotkey=f"hk{i}",
                prediction=0.5,
                event_score=0.5,
                spec_version=1,
            )
            for i in range(50)
        ]

        await db_operations.insert_peer_scores(scores)

        await db_client.update(
            "UPDATE scores SET processed = 1, metagraph_score = miner_uid * 0.01",
        )

        # insert predictions like 0.0, 0.01, 0.02, 0.03, ..., 0.99 for calculations
        predictions = [
            (
                unique_event_id_1,
                f"hk{i}",
                i,
                1.0,
                10,
                i * 0.01,
            )
            for i in range(100)
        ] + [
            (
                unique_event_id_2,
                f"hk{i}",
                i,
                1.0,
                10,
                i * 0.01,
            )
            for i in range(100)
        ]

        await db_operations.upsert_predictions(predictions=predictions)

        # No prediction for interval
        inference_data = await db_operations.get_community_inference_dataset(
            unique_event_ids=unique_event_ids, interval_start_minutes=11, top_n_ranks=100
        )

        assert len(inference_data) == 0

        inference_data = await db_operations.get_community_inference_dataset(
            unique_event_ids=unique_event_ids, interval_start_minutes=10, top_n_ranks=100
        )
        assert len(inference_data) == 100
        for j, row in enumerate(inference_data):
            i = j % 50
            if j >= 50:
                assert row["unique_event_id"] == unique_event_id_2
            else:
                assert row["unique_event_id"] == unique_event_id_1
            assert row["prev_batch_miner_rank"] == i + 1
            assert row["metagraph_score"] == pytest.approx(0.5 - (i + 1) * 0.01)
            assert row["agg_prediction"] == pytest.approx(0.5 - (i + 1) * 0.01)

        # check for metagraph scores all 0s
        await db_client.update(
            "UPDATE scores SET metagraph_score = 0.0",
        )

        predictions = [
            (
                unique_event_id_1,
                f"hk{i}",
                i,
                1.0,
                12,  # interval_start_minutes
                i * 0.01,
            )
            for i in range(10)
        ]
        await db_operations.upsert_predictions(predictions=predictions)

        inference_data = await db_operations.get_community_inference_dataset(
            unique_event_ids=[unique_event_id_1], interval_start_minutes=12, top_n_ranks=100
        )
        assert len(inference_data) == 10

        for i, row in enumerate(inference_data):
            assert row["unique_event_id"] == unique_event_id_1
            assert row["prev_batch_miner_rank"] == i + 1
            assert row["metagraph_score"] == 0.0
            assert row["agg_prediction"] == pytest.approx(i * 0.01)

    async def test_delete_reasonings_orphan(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        reasonings = [
            ReasoningModel(
                event_id="non_existent_event_1",
                miner_uid=1,
                miner_hotkey="hotkey1",
                reasoning="Test reasoning 1",
                # Not exported
                exported=False,
            ),
            ReasoningModel(
                event_id="non_existent_event_2",
                miner_uid=2,
                miner_hotkey="hotkey2",
                reasoning="Test reasoning 2",
                exported=True,
            ),
        ]

        await db_operations.upsert_reasonings(reasonings)

        result = await db_client.one("SELECT COUNT(*) FROM reasoning")
        assert result[0] == 2

        # Delete orphan reasonings
        deleted = await db_operations.delete_reasonings(batch_size=100)

        # Deleted in order
        assert deleted == [(1,), (2,)]

        result = await db_client.one("SELECT COUNT(*) FROM reasoning")

        assert result[0] == 0

    async def test_delete_reasonings_resolved_old(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        old_date = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()

        new_date = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()

        events = [
            EventsModel(
                unique_event_id="old_event",
                event_id="old_event",
                market_type="market_type",
                event_type="type",
                description="Old event",
                outcome="1",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                resolved_at=old_date,
                processed=True,
            ),
            EventsModel(
                unique_event_id="new_event",
                event_id="new_event",
                market_type="market_type",
                event_type="type",
                description="New event",
                outcome="1",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                resolved_at=new_date,
                processed=True,
            ),
        ]

        await db_operations.upsert_pydantic_events(events)

        reasonings = [
            ReasoningModel(
                event_id="old_event",
                miner_uid=1,
                miner_hotkey="hotkey1",
                reasoning="Test reasoning 1",
                exported=False,
            ),
            ReasoningModel(
                event_id="new_event",
                miner_uid=2,
                miner_hotkey="hotkey2",
                reasoning="Test reasoning 2",
                exported=False,
            ),
        ]

        await db_operations.upsert_reasonings(reasonings)

        result = await db_client.one("SELECT COUNT(*) FROM reasoning")
        assert result[0] == 2

        # Delete old reasonings
        deleted = await db_operations.delete_reasonings(batch_size=100)

        # Deleted in order
        assert deleted == [(1,)]

        # Verify reasoning for new_event remains
        result = await db_client.one("SELECT event_id FROM reasoning")
        assert result[0] == "new_event"

    async def test_delete_reasonings_discarded_and_deleted_events(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        events = [
            EventsModel(
                unique_event_id="discarded_event",
                event_id="discarded_event",
                market_type="market_type",
                event_type="type",
                description="Discarded event",
                outcome=None,
                status=EventStatus.DISCARDED,
                metadata='{"key": "value"}',
                resolved_at=None,
            ),
            EventsModel(
                unique_event_id="deleted_event",
                event_id="deleted_event",
                market_type="market_type",
                event_type="type",
                description="Deleted event",
                outcome=None,
                status=EventStatus.DELETED,
                metadata='{"key": "value"}',
                resolved_at=None,
            ),
        ]

        await db_operations.upsert_pydantic_events(events=events)

        # Insert reasonings for the discarded event
        reasonings = [
            ReasoningModel(
                event_id="discarded_event",
                miner_uid=1,
                miner_hotkey="hotkey1",
                reasoning="Test reasoning 1",
                exported=False,
            ),
            ReasoningModel(
                event_id="discarded_event",
                miner_uid=2,
                miner_hotkey="hotkey2",
                reasoning="Test reasoning 2",
                exported=False,
            ),
            ReasoningModel(
                event_id="deleted_event",
                miner_uid=2,
                miner_hotkey="hotkey2",
                reasoning="Test reasoning 3",
                exported=False,
            ),
        ]

        await db_operations.upsert_reasonings(reasonings=reasonings)

        result = await db_client.one("SELECT COUNT(*) FROM reasoning")

        assert result[0] == 3

        # Delete discarded reasonings
        deleted = await db_operations.delete_reasonings(batch_size=100)

        # Deleted in order
        assert deleted == [(1,), (2,), (3,)]

        result = await db_client.one("SELECT COUNT(*) FROM reasoning")
        assert result[0] == 0

    async def test_delete_reasonings_no_results(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        resolved_at_recently = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()

        event = EventsModel(
            unique_event_id="recent_event",
            event_id="recent_event",
            market_type="market_type",
            event_type="type",
            description="Recent event",
            outcome="1",
            status=EventStatus.PENDING,
            metadata='{"key": "value"}',
            resolved_at=resolved_at_recently,
            processed=True,
        )
        await db_operations.upsert_pydantic_events([event])

        reasonings = [
            ReasoningModel(
                event_id="recent_event",
                miner_uid=1,
                miner_hotkey="hotkey1",
                reasoning="Test reasoning 1",
                exported=False,
            ),
        ]
        await db_operations.upsert_reasonings(reasonings)

        # Verify reasonings were inserted
        result = await db_client.one("SELECT COUNT(*) FROM reasoning")

        assert result[0] == 1

        # Try to delete reasonings (should not delete any as they're too recent)
        deleted = await db_operations.delete_reasonings(batch_size=100)

        assert len(deleted) == 0

        # Verify reasonings were not deleted
        result = await db_client.one("SELECT COUNT(*) FROM reasoning")

        assert result[0] == 1

    async def test_get_events_last_deleted_at(self, db_operations: DatabaseOperations):
        events = [
            EventsModel(
                unique_event_id="unique1",
                event_id="event1",
                market_type="market_type",
                event_type="type",
                description="Recent event",
                outcome="1",
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
            ),
            EventsModel(
                unique_event_id="unique2",
                event_id="event2",
                market_type="market_type",
                event_type="type",
                description="Recent event",
                outcome="1",
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
            ),
            EventsModel(
                unique_event_id="unique3",
                event_id="event3",
                market_type="market_type",
                event_type="type",
                description="Recent event",
                outcome="1",
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
            ),
        ]

        current_time = datetime.now(timezone.utc)
        future_time = current_time + timedelta(days=1)

        await db_operations.upsert_pydantic_events(events=events)

        # Delete events 1 and 3
        await db_operations.delete_event(event_id="event1", deleted_at=current_time)
        await db_operations.delete_event(event_id="event3", deleted_at=future_time)

        result = await db_operations.get_events_last_deleted_at()

        assert result == future_time.isoformat().replace("T", " ")

    async def test_get_events_last_deleted_at_no_events(self, db_operations: DatabaseOperations):
        result = await db_operations.get_events_last_deleted_at()

        assert result is None
