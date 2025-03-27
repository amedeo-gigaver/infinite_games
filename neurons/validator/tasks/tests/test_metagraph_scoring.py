import json
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
from freezegun import freeze_time

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.score import SCORE_FIELDS, ScoresModel
from neurons.validator.tasks.metagraph_scoring import MOVING_AVERAGE_EVENTS, MetagraphScoring
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class TestMetagraphScoring:
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
    def db_operations(self, db_client: DatabaseClient):
        logger = MagicMock(spec=InfiniteGamesLogger)

        return DatabaseOperations(db_client=db_client, logger=logger)

    @pytest.fixture
    def metagraph_scoring_task(
        self,
        db_operations: DatabaseOperations,
    ):
        logger = MagicMock(spec=InfiniteGamesLogger)

        with freeze_time("2025-01-02 03:00:00"):
            return MetagraphScoring(
                interval_seconds=60.0,
                page_size=100,
                db_operations=db_operations,
                logger=logger,
            )

    def test_init(self, metagraph_scoring_task: MetagraphScoring):
        unit = metagraph_scoring_task

        assert isinstance(unit, MetagraphScoring)
        assert unit.interval_seconds == 60.0
        assert unit.page_size == 100
        assert unit.errors_count == 0

    @pytest.mark.parametrize(
        "scores_list, expected_result, log_calls",
        [
            # Case 1: No scores.
            (
                [],
                [],
                {
                    "debug": [
                        ("No events to calculate metagraph scores.", {}),
                    ]
                },
            ),
            # Case 2: All scores are processed.
            (
                [
                    ScoresModel(
                        event_id="processed_event_id_1",
                        miner_uid=3,
                        miner_hotkey="hk3",
                        prediction=0.75,
                        event_score=0.80,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=True,
                    ),
                    ScoresModel(
                        event_id="processed_event_id_2",
                        miner_uid=1,
                        miner_hotkey="hk1",
                        prediction=0.85,
                        event_score=0.90,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=True,
                    ),
                ],
                [],
                {
                    "debug": [
                        ("No events to calculate metagraph scores.", {}),
                    ]
                },
            ),
            # Case 3: Single miner single event.
            (
                [
                    ScoresModel(
                        event_id="expected_event_id",
                        miner_uid=3,
                        miner_hotkey="hk3",
                        prediction=0.75,
                        event_score=0.80,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                ],
                [
                    {
                        "event_id": "expected_event_id",
                        "processed": 1,
                        "metagraph_score": 1.0,
                        "other_data": {
                            "sum_peer_score": 0.80,
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": 1,
                            "avg_peer_score": 0.008 * (99 / MOVING_AVERAGE_EVENTS),
                            "sqmax_avg_peer_score": 0.00064,
                        },
                    },
                ],
                {
                    "debug": [
                        (
                            "Found events to calculate metagraph scores.",
                            {"n_events": 1},
                        ),
                        (
                            "Processing event for metagraph scoring.",
                            {"event_id": "expected_event_id"},
                        ),
                        (
                            "Metagraph scores calculated successfully.",
                            {"event_id": "expected_event_id"},
                        ),
                        ("Metagraph scoring task completed.", {"errors_count": 0}),
                    ]
                },
            ),
            # Case 3: Two miners single event, one miner processed -> reprocess.
            (
                [
                    ScoresModel(
                        event_id="expected_event_id",
                        miner_uid=3,
                        miner_hotkey="hk3",
                        prediction=0.75,
                        event_score=0.80,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=True,
                    ),
                    ScoresModel(
                        event_id="expected_event_id",
                        miner_uid=4,
                        miner_hotkey="hk4",
                        prediction=0.75,
                        event_score=0.40,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                ],
                [
                    {
                        "event_id": "expected_event_id",
                        "processed": 1,
                        "metagraph_score": 0.8,
                        "other_data": {
                            "sum_peer_score": 0.80,
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": 1,
                            "avg_peer_score": 0.008 * (99 / MOVING_AVERAGE_EVENTS),
                            "sqmax_avg_peer_score": 0.000064,
                        },
                    },
                    {
                        "event_id": "expected_event_id",
                        "processed": 1,
                        "metagraph_score": 0.2,
                        "other_data": {
                            "sum_peer_score": 0.40,
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": 1,
                            "avg_peer_score": 0.004 * (99 / MOVING_AVERAGE_EVENTS),
                            "sqmax_avg_peer_score": 0.000016,
                        },
                    },
                ],
                {
                    "debug": [
                        (
                            "Found events to calculate metagraph scores.",
                            {"n_events": 1},
                        ),
                        (
                            "Processing event for metagraph scoring.",
                            {"event_id": "expected_event_id"},
                        ),
                        (
                            "Metagraph scores calculated successfully.",
                            {"event_id": "expected_event_id"},
                        ),
                        ("Metagraph scoring task completed.", {"errors_count": 0}),
                    ]
                },
            ),
            # Case 4: Exception during set metagraph peer scores.
            (
                [
                    ScoresModel(
                        event_id="expected_event_id",
                        miner_uid=3,
                        miner_hotkey="hk3",
                        prediction=0.75,
                        event_score=0.80,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                ],
                [],
                {
                    "debug": [
                        (
                            "Found events to calculate metagraph scores.",
                            {"n_events": 1},
                        ),
                        (
                            "Processing event for metagraph scoring.",
                            {"event_id": "expected_event_id"},
                        ),
                    ],
                    "exception": [
                        (
                            "Error calculating metagraph scores.",
                            {"event_id": "expected_event_id"},
                        ),
                    ],
                },
            ),
            # Case 5: Three miners single event, one has negative peer score.
            (
                [
                    ScoresModel(
                        event_id="expected_event_id",
                        miner_uid=3,
                        miner_hotkey="hk3",
                        prediction=0.75,
                        event_score=0.80,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id",
                        miner_uid=4,
                        miner_hotkey="hk4",
                        prediction=0.75,
                        event_score=0.40,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id",
                        miner_uid=5,
                        miner_hotkey="hk5",
                        prediction=0.75,
                        event_score=-0.40,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                ],
                [
                    {
                        "event_id": "expected_event_id",
                        "processed": 1,
                        "metagraph_score": 0.8,
                        "other_data": {
                            "sum_peer_score": 0.80,
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": 1,
                            "avg_peer_score": 0.008 * (99 / MOVING_AVERAGE_EVENTS),
                            "sqmax_avg_peer_score": 0.000064,
                        },
                    },
                    {
                        "event_id": "expected_event_id",
                        "processed": 1,
                        "metagraph_score": 0.2,
                        "other_data": {
                            "sum_peer_score": 0.40,
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": 1,
                            "avg_peer_score": 0.004 * (99 / MOVING_AVERAGE_EVENTS),
                            "sqmax_avg_peer_score": 0.000016,
                        },
                    },
                    {
                        "event_id": "expected_event_id",
                        "processed": 1,
                        "metagraph_score": 0.0,
                        "other_data": {
                            "sum_peer_score": -0.40,
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": 1,
                            "avg_peer_score": -0.004 * (99 / MOVING_AVERAGE_EVENTS),
                            "sqmax_avg_peer_score": 0.0,
                        },
                    },
                ],
                {
                    "debug": [
                        (
                            "Found events to calculate metagraph scores.",
                            {"n_events": 1},
                        ),
                        (
                            "Processing event for metagraph scoring.",
                            {"event_id": "expected_event_id"},
                        ),
                        (
                            "Metagraph scores calculated successfully.",
                            {"event_id": "expected_event_id"},
                        ),
                        ("Metagraph scoring task completed.", {"errors_count": 0}),
                    ]
                },
            ),
            # Case 6: One miner for 2 events, one miner for 1 event.
            (
                [
                    ScoresModel(
                        event_id="expected_event_id_1",
                        miner_uid=3,
                        miner_hotkey="hk3",
                        prediction=0.75,
                        event_score=0.80,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id_2",
                        miner_uid=3,
                        miner_hotkey="hk3",
                        prediction=0.75,
                        event_score=0.40,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id_2",
                        miner_uid=4,
                        miner_hotkey="hk4",
                        prediction=0.75,
                        event_score=0.40,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                ],
                [
                    {
                        "event_id": "expected_event_id_1",
                        "processed": 1,
                        "metagraph_score": 1.0,
                        "other_data": {
                            "sum_peer_score": 0.80,
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": 1,
                            "avg_peer_score": 0.008 * (99 / MOVING_AVERAGE_EVENTS),
                            "sqmax_avg_peer_score": 0.000064,
                        },
                    },
                    {
                        "event_id": "expected_event_id_2",
                        "processed": 1,
                        "metagraph_score": 0.9,
                        "other_data": {
                            "sum_peer_score": 1.2,
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": 2,
                            "avg_peer_score": 0.012 * (99 / MOVING_AVERAGE_EVENTS),
                            "sqmax_avg_peer_score": 0.000144,
                        },
                    },
                    {
                        "event_id": "expected_event_id_2",
                        "processed": 1,
                        "metagraph_score": 0.1,
                        "other_data": {
                            "sum_peer_score": 0.40,
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": 1,
                            "avg_peer_score": 0.004 * (99 / MOVING_AVERAGE_EVENTS),
                            "sqmax_avg_peer_score": 0.000016,
                        },
                    },
                ],
                {
                    "debug": [
                        (
                            "Found events to calculate metagraph scores.",
                            {"n_events": 2},
                        ),
                        (
                            "Processing event for metagraph scoring.",
                            {"event_id": "expected_event_id_1"},
                        ),
                        (
                            "Metagraph scores calculated successfully.",
                            {"event_id": "expected_event_id_1"},
                        ),
                        (
                            "Processing event for metagraph scoring.",
                            {"event_id": "expected_event_id_2"},
                        ),
                        (
                            "Metagraph scores calculated successfully.",
                            {"event_id": "expected_event_id_2"},
                        ),
                        ("Metagraph scoring task completed.", {"errors_count": 0}),
                    ]
                },
            ),
            # Case 7: 1 miner 3 events, 1 miner 2 events, 1 miner 1 event with negative peer score.
            (
                [
                    ScoresModel(
                        event_id="expected_event_id_1",
                        miner_uid=3,
                        miner_hotkey="hk3",
                        prediction=0.75,
                        event_score=0.80,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id_2",
                        miner_uid=3,
                        miner_hotkey="hk3",
                        prediction=0.75,
                        event_score=0.40,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id_3",
                        miner_uid=3,
                        miner_hotkey="hk3",
                        prediction=0.75,
                        event_score=0.60,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id_2",
                        miner_uid=4,
                        miner_hotkey="hk4",
                        prediction=0.75,
                        event_score=0.40,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id_3",
                        miner_uid=4,
                        miner_hotkey="hk4",
                        prediction=0.75,
                        event_score=0.40,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id_2",
                        miner_uid=5,
                        miner_hotkey="hk5",
                        prediction=0.75,
                        event_score=-0.40,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                ],
                [
                    {
                        "event_id": "expected_event_id_1",
                        "processed": 1,
                        "metagraph_score": 1.0,
                        "other_data": {
                            "sum_peer_score": 0.8,
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": 1,
                            "avg_peer_score": 0.008 * (99 / MOVING_AVERAGE_EVENTS),
                            "sqmax_avg_peer_score": 0.000064,
                        },
                    },
                    {
                        "event_id": "expected_event_id_2",
                        "processed": 1,
                        "metagraph_score": 0.9,
                        "other_data": {
                            "sum_peer_score": 1.2,
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": 2,
                            "avg_peer_score": 0.012 * (99 / MOVING_AVERAGE_EVENTS),
                            "sqmax_avg_peer_score": 0.000144,
                        },
                    },
                    {
                        "event_id": "expected_event_id_3",
                        "processed": 1,
                        "metagraph_score": 0.835,
                        "other_data": {
                            "sum_peer_score": 1.8,
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": 3,
                            "avg_peer_score": 0.018 * (99 / MOVING_AVERAGE_EVENTS),
                            "sqmax_avg_peer_score": 0.000164,
                        },
                    },
                    {
                        "event_id": "expected_event_id_2",
                        "processed": 1,
                        "metagraph_score": 0.1,
                        "other_data": {
                            "sum_peer_score": 0.40,
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": 1,
                            "avg_peer_score": 0.004 * (99 / MOVING_AVERAGE_EVENTS),
                            "sqmax_avg_peer_score": 0.000016,
                        },
                    },
                    {
                        "event_id": "expected_event_id_3",
                        "processed": 1,
                        "metagraph_score": 0.165,
                        "other_data": {
                            "sum_peer_score": 0.80,
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": 2,
                            "avg_peer_score": 0.008 * (99 / MOVING_AVERAGE_EVENTS),
                            "sqmax_avg_peer_score": 0.000064,
                        },
                    },
                    {
                        "event_id": "expected_event_id_2",
                        "processed": 1,
                        "metagraph_score": 0.0,
                        "other_data": {
                            "sum_peer_score": -0.40,
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": 1,
                            "avg_peer_score": -0.004 * (99 / MOVING_AVERAGE_EVENTS),
                            "sqmax_avg_peer_score": 0.0,
                        },
                    },
                ],
                {
                    "debug": [
                        (
                            "Found events to calculate metagraph scores.",
                            {"n_events": 3},
                        ),
                        (
                            "Processing event for metagraph scoring.",
                            {"event_id": "expected_event_id_1"},
                        ),
                        (
                            "Metagraph scores calculated successfully.",
                            {"event_id": "expected_event_id_1"},
                        ),
                        (
                            "Processing event for metagraph scoring.",
                            {"event_id": "expected_event_id_2"},
                        ),
                        (
                            "Metagraph scores calculated successfully.",
                            {"event_id": "expected_event_id_2"},
                        ),
                        (
                            "Processing event for metagraph scoring.",
                            {"event_id": "expected_event_id_3"},
                        ),
                        (
                            "Metagraph scores calculated successfully.",
                            {"event_id": "expected_event_id_3"},
                        ),
                        ("Metagraph scoring task completed.", {"errors_count": 0}),
                    ]
                },
            ),
        ],
    )
    async def test_run(
        self,
        metagraph_scoring_task: MetagraphScoring,
        scores_list,
        expected_result,
        log_calls,
        db_client,
        db_operations,
    ):
        # insert scores
        sql = f"""
            INSERT INTO scores ({', '.join(SCORE_FIELDS)})
            VALUES ({', '.join(['?'] * len(SCORE_FIELDS))})
        """
        if scores_list:
            score_tuples = [
                tuple(getattr(score, field) for field in SCORE_FIELDS) for score in scores_list
            ]
            await db_client.insert_many(sql, score_tuples)

            # confirm setup
            inserted_scores = await db_client.many("SELECT * FROM scores")
            assert len(inserted_scores) == len(scores_list)

        # run the task
        if "exception" in log_calls:
            # Mock the set_metagraph_peer_scores method to return non empty list
            db_operations.set_metagraph_peer_scores = AsyncMock(return_value=[100])

        await metagraph_scoring_task.run()
        updated_scores = await db_client.many("SELECT * FROM scores", use_row_factory=True)
        debug_calls = metagraph_scoring_task.logger.debug.call_args_list
        exception_calls = metagraph_scoring_task.logger.exception.call_args_list

        # confirm logs
        for log_type, calls in log_calls.items():
            for i, (args, kwargs) in enumerate(calls):
                if log_type == "debug":
                    assert debug_calls[i][0][0] == args
                    if kwargs:
                        assert debug_calls[i][1]["extra"] == kwargs
                elif log_type == "exception":
                    assert exception_calls[i][0][0] == args
                    assert exception_calls[i][1]["extra"] == kwargs

        # confirm results
        if expected_result:
            assert len(updated_scores) == len(expected_result)
            for i, expected in enumerate(expected_result):
                updated = updated_scores[i]
                assert updated["event_id"] == expected["event_id"]
                assert updated["processed"] == expected["processed"]
                assert updated["metagraph_score"] == pytest.approx(
                    expected["metagraph_score"], abs=1e-3
                )

                other_data = json.loads(updated["other_data"])
                assert other_data["sum_peer_score"] == pytest.approx(
                    expected["other_data"]["sum_peer_score"], abs=1e-3
                )
                assert other_data["count_peer_score"] == expected["other_data"]["count_peer_score"]
                assert (
                    other_data["true_count_peer_score"]
                    == expected["other_data"]["true_count_peer_score"]
                )
                assert other_data["avg_peer_score"] == pytest.approx(
                    expected["other_data"]["avg_peer_score"], abs=1e-3
                )
                assert other_data["sqmax_avg_peer_score"] == pytest.approx(
                    expected["other_data"]["sqmax_avg_peer_score"], abs=1e-3
                )
        else:
            assert len(updated_scores) == len(scores_list)
            for i, updated in enumerate(updated_scores):
                assert updated["metagraph_score"] is None
                assert updated["other_data"] is None
