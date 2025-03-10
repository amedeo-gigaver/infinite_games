from datetime import datetime, timedelta, timezone

from neurons.validator.db.tests.test_utils import TestDbOperationsBase
from neurons.validator.models.event import EventsModel, EventStatus
from neurons.validator.models.score import SCORE_FIELDS, ScoresModel


class TestDbOperationsPart2(TestDbOperationsBase):
    async def test_get_peer_scored_events_for_export_basic(self, db_operations, db_client):
        export_event_id = "export_event"
        non_export_event_id = "non_export_event"

        now = datetime.now(timezone.utc)

        # For export_event, we want processed=1 and exported=0.
        score_export = ScoresModel(
            event_id=export_event_id,
            miner_uid=1,
            miner_hotkey="hk1",
            prediction=0.85,
            event_score=0.90,
            spec_version=1,
        )
        # For non_export_event, set exported=1 so it should not be picked up.
        score_non_export = ScoresModel(
            event_id=non_export_event_id,
            miner_uid=2,
            miner_hotkey="hk2",
            prediction=0.75,
            event_score=0.80,
            spec_version=1,
        )

        await db_operations.insert_peer_scores([score_export, score_non_export])
        await db_client.update(
            "UPDATE scores SET processed = ?, exported = ?, created_at = ? WHERE event_id = ?",
            [1, 0, now.isoformat(), export_event_id],
        )
        await db_client.update(
            "UPDATE scores SET processed = ?, exported = ?, created_at = ? WHERE event_id = ?",
            [1, 1, now.isoformat(), non_export_event_id],
        )

        # Prepare corresponding events in the events table.
        export_event = EventsModel(
            unique_event_id="unique_export_event",
            event_id=export_event_id,
            market_type="market_type1",
            event_type="type1",
            description="Exportable event",
            outcome="1",
            status=EventStatus.SETTLED,
            metadata='{"key": "value"}',
            resolved_at=now.isoformat(),
        )
        non_export_event = EventsModel(
            unique_event_id="unique_non_export_event",
            event_id=non_export_event_id,
            market_type="market_type2",
            event_type="type2",
            description="Non-exportable event",
            outcome="0",
            status=EventStatus.SETTLED,
            metadata='{"key": "value"}',
            resolved_at=now.isoformat(),
        )

        await db_operations.upsert_pydantic_events([export_event, non_export_event])

        raw_scores = await db_client.many(
            "SELECT event_id, ROWID, processed, exported FROM scores ORDER BY ROWID ASC",
            use_row_factory=True,
        )
        assert len(raw_scores) >= 2
        assert raw_scores[0]["event_id"] == export_event_id
        assert raw_scores[0]["processed"] == 1
        assert raw_scores[0]["exported"] == 0
        assert raw_scores[1]["event_id"] == non_export_event_id
        assert raw_scores[1]["processed"] == 1
        assert raw_scores[1]["exported"] == 1

        result = await db_operations.get_peer_scored_events_for_export(max_events=1000)
        assert len(result) == 1
        event = result[0]
        assert event.event_id == export_event_id
        assert event.unique_event_id == "unique_export_event"
        assert event.market_type == "market_type1"
        assert event.event_type == "type1"
        assert event.description == "Exportable event"
        assert event.outcome == "1"
        assert event.status == EventStatus.SETTLED

    async def test_get_peer_scored_events_for_export_invalid_data(self, db_operations, db_client):
        # Create a fake row with missing required field(s) so that EventsModel parsing fails.
        invalid_row = {"unique_event_id": "unique_invalid", "market_type": "market_type_invalid"}

        async def fake_many(*args, **kwargs):
            return [invalid_row]

        db_client.many = fake_many

        logged_exceptions = []

        def fake_logger_exception(msg, extra):
            logged_exceptions.append((msg, extra))

        db_operations.logger.exception = fake_logger_exception

        result = await db_operations.get_peer_scored_events_for_export(max_events=1000)

        assert result == []
        assert len(logged_exceptions) >= 1
        for msg, extra in logged_exceptions:
            assert "Error parsing event" in msg
            assert "row" in extra

    async def test_get_peer_scores_for_export_basic(self, db_operations, db_client):
        event_id = "test_event"

        score1 = ScoresModel(
            event_id=event_id,
            miner_uid=1,
            miner_hotkey="hk1",
            prediction=0.8,
            event_score=0.85,
            spec_version=1,
        )
        score2 = ScoresModel(
            event_id=event_id,
            miner_uid=2,
            miner_hotkey="hk2",
            prediction=0.9,
            event_score=0.95,
            spec_version=1,
        )
        other_score = ScoresModel(
            event_id="other_event",
            miner_uid=3,
            miner_hotkey="hk3",
            prediction=0.7,
            event_score=0.75,
            spec_version=1,
        )
        await db_operations.insert_peer_scores([score1, score2, other_score])
        await db_client.update(
            "UPDATE scores SET processed = ? WHERE event_id = ?",
            [1, event_id],
        )
        await db_client.update(
            "UPDATE scores SET processed = ? WHERE event_id = ?",
            [0, "other_event"],
        )

        scores = await db_operations.get_peer_scores_for_export(event_id)

        assert isinstance(scores, list)
        assert len(scores) == 2
        for score in scores:
            assert score.event_id == event_id
            assert score.processed == 1

    async def test_get_peer_scores_for_export_invalid_data(self, db_operations, db_client):
        # Test that the method can handle invalid data.
        event_id = "test_event_invalid"
        invalid_row = {field: "invalid" for field in SCORE_FIELDS if field != "event_id"}

        async def fake_many(*args, **kwargs):
            return [invalid_row]

        db_client.many = fake_many

        logged_exceptions = []

        def fake_logger_exception(msg, extra):
            logged_exceptions.append((msg, extra))

        db_operations.logger.exception = fake_logger_exception

        scores = await db_operations.get_peer_scores_for_export(event_id)
        assert scores == []
        assert len(logged_exceptions) >= 1
        for msg, extra in logged_exceptions:
            assert "Error parsing score" in msg
            assert "row" in extra

    async def test_mark_peer_scores_as_exported(self, db_operations, db_client):
        events = ["event1", "event2", "event3"]
        scores = [
            ScoresModel(
                event_id=event_id,
                miner_uid=1,
                miner_hotkey="hk1",
                prediction=0.80,
                event_score=0.85,
                spec_version=1,
            )
            for event_id in events
        ]

        await db_operations.insert_peer_scores(scores)

        initial = await db_client.many(
            "SELECT event_id, exported FROM scores ORDER BY ROWID ASC",
            use_row_factory=True,
        )
        assert len(initial) == len(events)
        for row, ev in zip(initial, events):
            assert row["exported"] == 0
            assert row["event_id"] == ev

        # Call the method under test.
        update_result = await db_operations.mark_peer_scores_as_exported(events[1])
        assert update_result == []

        updated = await db_client.many(
            "SELECT event_id, exported FROM scores ORDER BY ROWID ASC",
            use_row_factory=True,
        )
        assert len(updated) == len(events)
        for row, ev in zip(updated, events):
            if ev == events[1]:
                assert row["exported"] == 1
            else:
                assert row["exported"] == 0

    async def test_get_last_metagraph_scores(self, db_operations, db_client):
        created_at = datetime.now(timezone.utc) - timedelta(days=1)
        scores_list = [
            ScoresModel(
                event_id="expected_event_id_1",
                miner_uid=3,
                miner_hotkey="hk3",
                prediction=0.75,
                event_score=0.80,
                metagraph_score=1.0,
                created_at=created_at,
                spec_version=1,
                processed=True,
            ),
            ScoresModel(
                event_id="expected_event_id_2",
                miner_uid=3,
                miner_hotkey="hk3",
                prediction=0.75,
                event_score=0.40,
                metagraph_score=0.9,
                created_at=created_at,
                spec_version=1,
                processed=True,
            ),
            ScoresModel(
                event_id="expected_event_id_3",
                miner_uid=3,
                miner_hotkey="hk3",
                prediction=0.75,
                event_score=0.60,
                metagraph_score=0.835,
                created_at=created_at,
                spec_version=1,
                processed=True,
            ),
            ScoresModel(
                event_id="expected_event_id_2",
                miner_uid=4,
                miner_hotkey="hk4",
                prediction=0.75,
                event_score=0.40,
                metagraph_score=0.1,
                created_at=created_at,
                spec_version=1,
                processed=True,
            ),
            ScoresModel(
                event_id="expected_event_id_1",
                miner_uid=4,
                miner_hotkey="hk4",
                prediction=0.75,
                event_score=0.40,
                metagraph_score=0.165,
                created_at=created_at,
                spec_version=1,
                processed=True,
            ),
            ScoresModel(
                event_id="expected_event_id_2",
                miner_uid=5,
                miner_hotkey="hk5",
                prediction=0.75,
                event_score=-0.40,
                metagraph_score=0.0,
                created_at=created_at,
                spec_version=1,
                processed=True,
            ),
        ]
        # insert scores
        sql = f"""
            INSERT INTO scores ({', '.join(SCORE_FIELDS)})
            VALUES ({', '.join(['?'] * len(SCORE_FIELDS))})
        """
        score_tuples = [
            tuple(getattr(score, field) for field in SCORE_FIELDS) for score in scores_list
        ]
        await db_client.insert_many(sql, score_tuples)

        inserted_scores = await db_client.many("SELECT * FROM scores")
        assert len(inserted_scores) == len(scores_list)

        # get last metagraph scores
        last_metagraph_scores = await db_operations.get_last_metagraph_scores()
        assert len(last_metagraph_scores) == 3

        assert last_metagraph_scores[0].event_id == "expected_event_id_3"
        assert last_metagraph_scores[0].miner_uid == 3
        assert last_metagraph_scores[0].miner_hotkey == "hk3"
        assert last_metagraph_scores[0].metagraph_score == 0.835

        assert last_metagraph_scores[1].event_id == "expected_event_id_1"
        assert last_metagraph_scores[1].miner_uid == 4
        assert last_metagraph_scores[1].miner_hotkey == "hk4"
        assert last_metagraph_scores[1].metagraph_score == 0.165

        assert last_metagraph_scores[2].event_id == "expected_event_id_2"
        assert last_metagraph_scores[2].miner_uid == 5
        assert last_metagraph_scores[2].miner_hotkey == "hk5"
        assert last_metagraph_scores[2].metagraph_score == 0.0

    async def test_mark_event_as_discarded(self, db_operations, db_client):
        now = datetime.now(timezone.utc)
        discarded_unique_event_id = "discarded_event"
        non_discarded_unique_event_id = "non_discarded_event"
        export_event = EventsModel(
            unique_event_id=discarded_unique_event_id,
            event_id="ev1",
            market_type="market_type1",
            event_type="type1",
            description="Bad event",
            outcome="1",
            status=EventStatus.SETTLED,
            metadata='{"key": "value"}',
            resolved_at=now.isoformat(),
        )
        non_export_event = EventsModel(
            unique_event_id=non_discarded_unique_event_id,
            event_id="ev2",
            market_type="market_type2",
            event_type="type2",
            description="Good event",
            outcome="0",
            status=EventStatus.SETTLED,
            metadata='{"key": "value"}',
            resolved_at=now.isoformat(),
        )

        await db_operations.upsert_pydantic_events([export_event, non_export_event])

        initial = await db_client.many(
            "SELECT unique_event_id, status FROM events ORDER BY ROWID ASC",
            use_row_factory=True,
        )
        assert len(initial) == 2
        assert initial[0]["unique_event_id"] == discarded_unique_event_id
        # status column is text type :|
        assert initial[0]["status"] == str(EventStatus.SETTLED.value)
        assert initial[1]["unique_event_id"] == non_discarded_unique_event_id
        assert initial[1]["status"] == str(EventStatus.SETTLED.value)

        update_result = await db_operations.mark_event_as_discarded(discarded_unique_event_id)
        assert update_result == []

        updated = await db_client.many(
            "SELECT unique_event_id, status FROM events ORDER BY ROWID ASC",
            use_row_factory=True,
        )
        assert len(updated) == 2
        assert updated[0]["unique_event_id"] == discarded_unique_event_id
        assert updated[0]["status"] == str(EventStatus.DISCARDED.value)
        assert updated[1]["unique_event_id"] == non_discarded_unique_event_id
        assert updated[1]["status"] == str(EventStatus.SETTLED.value)

    async def test_vacuum_database(self, db_operations):
        result = await db_operations.vacuum_database(500)

        assert result is None

    async def test_get_wa_prediction_event(self, db_operations, db_client):
        unique_event_id = "unique_event_id"

        wa_prediction = await db_operations.get_wa_prediction_event(unique_event_id, 1)

        # No predictions yet
        assert wa_prediction is None

        scores = [
            ScoresModel(
                event_id="event_id",
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
                unique_event_id,
                f"hk{i}",
                i,
                "1",
                10,
                i * 0.01,
                1,
                i * 0.01,
            )
            for i in range(100)
        ]
        await db_operations.upsert_predictions(predictions=predictions)
        # validate inserted predictions
        inserted_predictions = await db_client.many(
            "SELECT * FROM predictions", use_row_factory=True
        )
        assert [int(p["minerUid"]) for p in inserted_predictions] == list(range(100))
        assert [p["interval_agg_prediction"] for p in inserted_predictions] == [
            i * 0.01 for i in range(100)
        ]

        # No prediction for interval
        wa_prediction = await db_operations.get_wa_prediction_event(unique_event_id, 11)

        assert wa_prediction is None

        wa_prediction = await db_operations.get_wa_prediction_event(
            unique_event_id=unique_event_id,
            interval_start_minutes=10,
        )
        # top 10 miners should be 40-49 with metagraph scores 0.4-0.49
        # their prediction should be 0.40-0.49
        assert wa_prediction == sum([(i * 0.01) ** 2 for i in range(40, 50)]) / sum(
            [i * 0.01 for i in range(40, 50)]
        )
