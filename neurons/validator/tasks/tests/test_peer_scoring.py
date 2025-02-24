import copy
import math
import tempfile
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from bittensor.core.metagraph import MetagraphMixin
from freezegun import freeze_time
from pandas.testing import assert_frame_equal

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.event import EventsModel, EventStatus
from neurons.validator.models.miner import MinersModel
from neurons.validator.models.prediction import PredictionsModel
from neurons.validator.tasks.peer_scoring import CLIP_EPS, PeerScoring, PSNames
from neurons.validator.utils.common.interval import (
    AGGREGATION_INTERVAL_LENGTH_MINUTES,
    align_to_interval,
    minutes_since_epoch,
    to_utc,
)
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class TestPeerScoring:
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
    def peer_scoring_task(
        self,
        db_operations: DatabaseOperations,
    ):
        metagraph = MagicMock(spec=MetagraphMixin)
        metagraph.sync = MagicMock()

        # Mock metagraph attributes
        metagraph.uids = torch.tensor([1, 2, 3], dtype=torch.int32).to("cpu")
        metagraph.hotkeys = ["hotkey1", "hotkey2", "hotkey3"]

        logger = MagicMock(spec=InfiniteGamesLogger)

        with freeze_time("2024-12-27 07:00:00"):
            return PeerScoring(
                interval_seconds=60.0,
                db_operations=db_operations,
                metagraph=metagraph,
                logger=logger,
            )

    def test_init(self, peer_scoring_task: PeerScoring):
        unit = peer_scoring_task

        assert isinstance(unit, PeerScoring)
        assert torch.equal(unit.current_uids, torch.tensor([1, 2, 3], dtype=torch.int32))
        assert unit.current_hotkeys == ["hotkey1", "hotkey2", "hotkey3"]
        assert unit.n_hotkeys == 3
        assert unit.interval_seconds == 60.0
        assert unit.current_miners_df.index.size == 3
        assert unit.current_miners_df.miner_uid.tolist() == [1, 2, 3]

    def test_metagraph_lite_sync(self, peer_scoring_task: PeerScoring):
        unit = peer_scoring_task

        assert unit.current_miners_df.miner_uid.tolist() == [1, 2, 3]
        assert unit.current_hotkeys == ["hotkey1", "hotkey2", "hotkey3"]

        unit.metagraph.uids = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        unit.metagraph.hotkeys = ["hotkey1", "hotkey2", "hotkey3", "hotkey4"]

        unit.metagraph_lite_sync()
        assert unit.current_miners_df.miner_uid.tolist() == [1, 2, 3, 4]
        assert unit.current_miners_df.miner_hotkey.tolist() == [
            "hotkey1",
            "hotkey2",
            "hotkey3",
            "hotkey4",
        ]
        assert unit.current_miners_df.index.size == 4
        assert unit.current_hotkeys == ["hotkey1", "hotkey2", "hotkey3", "hotkey4"]
        assert unit.current_uids.tolist() == [1, 2, 3, 4]
        assert unit.n_hotkeys == 4

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "db_rows, current_miners_df, expected_result, expected_log",
        [
            # Test case 1: No DB rows
            # – expect False with a "no miners" error.
            (
                [],  # Empty list for DB rows.
                pd.DataFrame({PSNames.miner_uid: [1], PSNames.miner_hotkey: ["hot1"]}),
                False,
                "No miners found in the DB, skipping scoring!",
            ),
            # Test case 2: DB rows exist but no overlap with current_miners_df
            # – expect False with an overlap error.
            (
                [
                    {
                        PSNames.miner_uid: "1",  # Note: stored as a string in the DB.
                        PSNames.miner_hotkey: "hot1",
                        PSNames.registered_date: pd.Timestamp("2025-01-01"),
                        "is_validating": False,
                        "validator_permit": False,
                    }
                ],
                # current_miners_df has different values so the merge will be empty.
                pd.DataFrame({PSNames.miner_uid: [2], PSNames.miner_hotkey: ["hot2"]}),
                False,
                "No overlap in miners between DB and metagraph, skipping scoring!",
            ),
            # Test case 3: DB rows exist and there is overlap
            # – expect True and computed registered minutes.
            (
                [
                    {
                        PSNames.miner_uid: "1",
                        PSNames.miner_hotkey: "hot1",
                        PSNames.registered_date: pd.Timestamp("2025-01-01"),
                        "is_validating": False,
                        "validator_permit": False,
                    }
                ],
                pd.DataFrame(
                    {
                        PSNames.miner_uid: [1],
                        PSNames.miner_hotkey: ["hot1"],
                        "other": ["data"],
                    }
                ),
                True,
                None,  # No error is expected.
            ),
        ],
    )
    async def test_miners_last_reg_sync(
        self,
        peer_scoring_task: PeerScoring,
        db_rows,
        current_miners_df,
        expected_result,
        expected_log,
    ):
        unit = peer_scoring_task
        unit.current_miners_df = current_miners_df

        miner_db_rows = [MinersModel(**dict(row)) for row in db_rows]
        with patch.object(
            unit.db_operations,
            "get_miners_last_registration",
            new=AsyncMock(return_value=miner_db_rows),
        ):
            result = await unit.miners_last_reg_sync()

        assert result == expected_result

        if expected_log is not None:
            assert unit.logger.error.call_args_list[0].args[0] == expected_log
            assert unit.logger.error.call_count == 1
        else:
            # When the merge is successful, verify that the computed column exists.
            assert unit.miners_last_reg is not None
            assert PSNames.miner_registered_minutes in unit.miners_last_reg.columns

            expected_minutes = minutes_since_epoch(to_utc(pd.Timestamp("2025-01-01")))
            computed_minutes = unit.miners_last_reg[PSNames.miner_registered_minutes].iloc[0]
            assert computed_minutes == expected_minutes

    def test_set_right_cutoff(self):
        input_event = EventsModel(
            unique_event_id="1",
            event_id="e1",
            market_type="dummy_market",
            event_type="dummy_type",
            description="dummy description",
            status=2,
            metadata="{}",
            registered_date=datetime(2025, 1, 1, 9, 0, 0),
            cutoff=datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            resolved_at=datetime(2025, 1, 1, 11, 0, 0, tzinfo=timezone.utc),
        )

        original_event = copy.deepcopy(input_event)

        with freeze_time("2025-01-01 12:00:00"):
            # Compute the effective cutoff.
            effective_cutoff = min(
                to_utc(input_event.cutoff),
                to_utc(input_event.resolved_at),
                datetime.now(timezone.utc),
            )
            result_event = PeerScoring.set_right_cutoff(input_event)

        assert result_event.cutoff == effective_cutoff
        assert result_event.resolved_at == effective_cutoff
        # Assert that registered_date has been converted using to_utc.
        assert result_event.registered_date == to_utc(original_event.registered_date)

        # Verify that the input event was not modified.
        assert input_event.cutoff == original_event.cutoff
        assert input_event.resolved_at == original_event.resolved_at
        assert input_event.registered_date == original_event.registered_date

    def test_get_intervals_df_no_intervals(self, peer_scoring_task: PeerScoring):
        unit = peer_scoring_task
        event_registered_start_minutes = 100
        event_cutoff_start_minutes = 100

        # Call the method.
        intervals_df = unit.get_intervals_df(
            event_registered_start_minutes, event_cutoff_start_minutes
        )

        # Verify that an error was logged.
        assert unit.logger.error.call_count == 1
        logged_args = unit.logger.error.call_args_list[0].args
        assert "n_intervals computed to be <= 0" in logged_args[0]

        # Verify that the returned DataFrame has the expected columns and is empty.
        expected_columns = [
            PSNames.interval_idx,
            PSNames.interval_start,
            PSNames.interval_end,
            PSNames.weight,
        ]
        assert list(intervals_df.columns) == expected_columns
        assert intervals_df.empty

    def test_get_intervals_df_success(self, peer_scoring_task: PeerScoring):
        unit = peer_scoring_task
        event_registered_start_minutes = 0
        event_cutoff_start_minutes = 580

        unit.errors_count = 0
        unit.logger.error.reset_mock()

        intervals_df = unit.get_intervals_df(
            event_registered_start_minutes, event_cutoff_start_minutes
        )

        assert unit.errors_count == 0
        assert unit.logger.error.call_count == 0

        n_intervals = (
            event_cutoff_start_minutes - event_registered_start_minutes
        ) // AGGREGATION_INTERVAL_LENGTH_MINUTES
        assert n_intervals == 2
        assert intervals_df.shape[0] == n_intervals

        expected_columns = [
            PSNames.interval_idx,
            PSNames.interval_start,
            PSNames.interval_end,
            PSNames.weight,
        ]
        assert list(intervals_df.columns) == expected_columns

        for idx, row in intervals_df.iterrows():
            assert row[PSNames.interval_idx] == idx

            expected_interval_start = (
                event_registered_start_minutes + idx * AGGREGATION_INTERVAL_LENGTH_MINUTES
            )
            expected_interval_end = expected_interval_start + AGGREGATION_INTERVAL_LENGTH_MINUTES
            assert row[PSNames.interval_start] == expected_interval_start
            assert row[PSNames.interval_end] == expected_interval_end

            expected_weight = math.exp(-(n_intervals / (n_intervals - idx)) + 1)
            np.testing.assert_almost_equal(row[PSNames.weight], expected_weight)

    def test_prepare_predictions_df_no_valid_miner(self, peer_scoring_task: PeerScoring):
        prediction = PredictionsModel(
            unique_event_id="ev1",
            minerHotkey="hotkey3",
            minerUid="3",
            predictedOutcome="1",
            canOverwrite=None,
            outcome="1",
            interval_start_minutes=100,
            interval_agg_prediction=0.5,
            interval_count=1,
            submitted=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            blocktime=12345,
            exported=False,
        )
        predictions = [prediction]

        miners = pd.DataFrame(
            {
                PSNames.miner_uid: [1, 2],
                PSNames.miner_hotkey: ["hotkey1", "hotkey2"],
            }
        )

        result_df = peer_scoring_task.prepare_predictions_df(predictions, miners)
        assert result_df.shape[0] == miners.shape[0]
        assert PSNames.interval_agg_prediction in result_df.columns
        assert result_df[PSNames.interval_agg_prediction].isna().all()

    def test_prepare_predictions_df_with_valid_miners(self, peer_scoring_task: PeerScoring):
        prediction = PredictionsModel(
            unique_event_id="ev1",
            minerHotkey="hotkey1",
            minerUid="1",
            predictedOutcome="1",
            canOverwrite=None,
            outcome="1",
            interval_start_minutes=100,
            # Use a value that may require clipping;
            interval_agg_prediction=0.005,
            interval_count=1,
            submitted=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            blocktime=12345,
            exported=False,
        )
        predictions = [prediction]

        miners = pd.DataFrame(
            {
                PSNames.miner_uid: [1, 2],
                PSNames.miner_hotkey: ["hotkey1", "hotkey2"],
            }
        )
        result_df = peer_scoring_task.prepare_predictions_df(predictions, miners)
        assert result_df.shape[0] == 2

        row1 = result_df[result_df[PSNames.miner_uid] == 1].iloc[0]
        assert row1[PSNames.interval_start] == 100
        assert row1[PSNames.miner_hotkey] == "hotkey1"

        expected_clipped = max(CLIP_EPS, min(0.005, 1 - CLIP_EPS))
        np.testing.assert_almost_equal(row1[PSNames.interval_agg_prediction], expected_clipped)

        row2 = result_df[result_df[PSNames.miner_uid] == 2].iloc[0]
        assert pd.isna(row2[PSNames.interval_agg_prediction])

    @pytest.mark.parametrize(
        "predictions_data, expected_value",
        [
            # Case 1: No predictions for any miner.
            (
                [
                    {
                        PSNames.miner_uid: 1,
                        PSNames.miner_hotkey: "hotkey1",
                        PSNames.interval_start: 100240,
                        PSNames.interval_agg_prediction: pd.NA,
                    }
                ],
                None,
            ),
            # Case 2: Predictions list with a valid prediction for miner 1 at interval_start 100.
            (
                [
                    {
                        PSNames.miner_uid: 1,
                        PSNames.miner_hotkey: "hotkey1",
                        PSNames.interval_start: 100480,
                        PSNames.interval_agg_prediction: 0.8,
                    }
                ],
                0.8,
            ),
        ],
    )
    def test_get_interval_scores_base_parametrized(
        self, peer_scoring_task: PeerScoring, predictions_data, expected_value
    ):
        miners = pd.DataFrame(
            {
                PSNames.miner_uid: [1, 2],
                PSNames.miner_hotkey: ["hotkey1", "hotkey2"],
                PSNames.miner_registered_minutes: [100, 200],
            }
        )

        intervals = pd.DataFrame(
            {
                PSNames.interval_idx: [0, 1],
                PSNames.interval_start: [100240, 100480],
                PSNames.interval_end: [100480, 100720],
                PSNames.weight: [0.8, 0.5],
            }
        )
        predictions_df = pd.DataFrame(predictions_data)

        result_df = peer_scoring_task.get_interval_scores_base(predictions_df, miners, intervals)
        assert result_df.shape[0] == 4
        assert result_df.columns.to_list() == [
            PSNames.miner_uid,
            PSNames.miner_hotkey,
            PSNames.miner_registered_minutes,
            PSNames.interval_idx,
            PSNames.interval_start,
            PSNames.interval_end,
            PSNames.weight,
            PSNames.interval_agg_prediction,
        ]

        if expected_value is None:
            assert result_df[PSNames.interval_agg_prediction].isna().all()
        else:
            row = result_df[
                (result_df[PSNames.miner_uid] == 1)
                & (result_df[PSNames.miner_hotkey] == "hotkey1")
                & (result_df[PSNames.interval_start] == 100480)
            ]
            assert not row.empty
            np.testing.assert_almost_equal(
                row.iloc[0][PSNames.interval_agg_prediction], expected_value
            )

            row_other = result_df[
                (result_df[PSNames.miner_uid] == 2)
                & (result_df[PSNames.miner_hotkey] == "hotkey2")
                & (result_df[PSNames.interval_start] == 100240)
            ]
            assert not row_other.empty
            assert pd.isna(row_other.iloc[0][PSNames.interval_agg_prediction])

    def test_return_empty_scores_df(self, peer_scoring_task: PeerScoring):
        unit = peer_scoring_task
        df = unit.return_empty_scores_df("test", "event_id")

        assert unit.logger.error.call_count == 1
        assert unit.logger.error.call_args_list[0].args[0] == "test"
        assert df.shape[0] == 0
        assert PSNames.miner_uid in df.columns
        assert PSNames.miner_hotkey in df.columns
        assert PSNames.rema_prediction in df.columns
        assert PSNames.rema_peer_score in df.columns

    @pytest.mark.parametrize(
        "outcome, prediction, expected_ls",
        [
            (1, 0.01, np.log(0.01)),
            (1, 0.5, np.log(0.5)),
            (1, 0.99, np.log(0.99)),
            (0, 0.01, np.log(1 - 0.01)),
            (0, 0.5, np.log(1 - 0.5)),
            (0, 0.99, np.log(1 - 0.99)),
        ],
    )
    def test_log_score_and_inverse_log_score_with_expected_ls(
        self, outcome, prediction, expected_ls
    ):
        ls = PeerScoring.log_score(prediction, outcome)
        np.testing.assert_allclose(
            ls,
            expected_ls,
            rtol=1e-5,
            atol=1e-8,
            err_msg=f"log_score incorrect for outcome={outcome} prediction={prediction}",
        )

        recovered_prediction = PeerScoring.inverse_log_score(ls, outcome)
        np.testing.assert_allclose(
            recovered_prediction,
            prediction,
            rtol=1e-5,
            atol=1e-8,
            err_msg=f"Failed for outcome {outcome} with prediction {prediction}",
        )

    # TODO: add more test cases
    @pytest.mark.parametrize(
        "outcome_round, input_df, expected",
        [
            # Test case for outcome_round == 1:
            (
                1,
                pd.DataFrame(
                    {
                        # Two miners in the same interval (interval_idx 0)
                        PSNames.miner_uid: [1, 2],
                        PSNames.miner_registered_minutes: [50, 150],
                        PSNames.interval_start: [100, 100],
                        # Row 1 is missing a prediction (will be imputed to wrong_outcome)
                        PSNames.interval_agg_prediction: [pd.NA, 0.8],
                        PSNames.weight: [1, 1],
                        PSNames.interval_idx: [0, 0],
                    }
                ),
                {
                    # For outcome_round 1, wrong_outcome = 1 - abs(1 - CLIP_EPS) = 1 - 0.99 = 0.01.
                    # Row 1:
                    1: {
                        PSNames.interval_agg_prediction: 0.01,  # imputed
                        PSNames.log_score: np.log(0.01),  # ≈ -4.605170186
                        # Group: sum = np.log(0.01)+np.log(0.8) = -4.605170186 + (-0.223143551)
                        # Others’ mean for row1 = (-4.828313737 - (-4.605170186)) = -0.223143551
                        PSNames.peer_score: np.log(0.01) - (-0.223143551),  # ≈ -4.382026635
                    },
                    # Row 2:
                    2: {
                        PSNames.interval_agg_prediction: 0.8,
                        PSNames.log_score: np.log(0.8),  # ≈ -0.223143551
                        # Others’ mean for row2 = (-4.828313737 - (-0.223143551)) = -4.605170186
                        PSNames.peer_score: np.log(0.8) - (-4.605170186),  # ≈ 4.382026635
                    },
                },
            ),
            # Test case for outcome_round == 0:
            (
                0,
                pd.DataFrame(
                    {
                        PSNames.miner_uid: [1, 2],
                        PSNames.miner_registered_minutes: [50, 150],
                        PSNames.interval_start: [100, 100],
                        # For outcome_round 0, row 1 missing prediction will be imputed to 0.99.
                        PSNames.interval_agg_prediction: [pd.NA, 0.2],
                        PSNames.weight: [1, 1],
                        PSNames.interval_idx: [0, 0],
                    }
                ),
                {
                    # For outcome_round 0, wrong_outcome = 1 - abs(0 - CLIP_EPS) = 1 - 0.01 = 0.99.
                    # Row 1:
                    1: {
                        PSNames.interval_agg_prediction: 0.99,  # imputed
                        PSNames.log_score: np.log(1 - 0.99),  # np.log(0.01) ≈ -4.605170186
                        # Others’ mean for row1 = (-4.828313737 - (-4.605170186)) = -0.223143551
                        PSNames.peer_score: np.log(1 - 0.99) - (-0.223143551),  # ≈ -4.382026635
                    },
                    # Row 2:
                    2: {
                        PSNames.interval_agg_prediction: 0.2,
                        PSNames.log_score: np.log(1 - 0.2),  # np.log(0.8) ≈ -0.223143551
                        # Others’ mean for row2 = (-4.828313737 - (-0.223143551)) = -4.605170186
                        PSNames.peer_score: np.log(1 - 0.2) - (-4.605170186),  # ≈ 4.382026635
                    },
                },
            ),
        ],
    )
    def test_peer_score_intervals(
        self, peer_scoring_task: PeerScoring, outcome_round, input_df, expected
    ):
        result_df = peer_scoring_task.peer_score_intervals(input_df, outcome_round)

        # For each expected row (keyed by miner_uid), compare the key columns.
        for uid, exp in expected.items():
            row = result_df[result_df[PSNames.miner_uid] == uid].iloc[0]
            np.testing.assert_allclose(
                row[PSNames.interval_agg_prediction],
                exp[PSNames.interval_agg_prediction],
                rtol=1e-5,
                err_msg=f"Incorrect interval_agg_prediction for miner_uid {uid} and outcome_round {outcome_round}",
            )
            np.testing.assert_allclose(
                row[PSNames.log_score],
                exp[PSNames.log_score],
                rtol=1e-5,
                err_msg=f"Incorrect log_score for miner_uid {uid} and outcome_round {outcome_round}",
            )
            np.testing.assert_allclose(
                row[PSNames.peer_score],
                exp[PSNames.peer_score],
                rtol=1e-5,
                err_msg=f"Incorrect peer_score for miner_uid {uid} and outcome_round {outcome_round}",
            )

    @pytest.mark.parametrize(
        "input_data, expected_data",
        [
            # Test case 1: Single-group (one miner with two rows)
            (
                {
                    PSNames.miner_uid: [1, 1],
                    PSNames.miner_hotkey: ["hotkey1", "hotkey1"],
                    PSNames.weighted_prediction: [0.2, 0.3],
                    PSNames.weighted_peer_score: [0.1, 0.2],
                    PSNames.weight: [2, 3],
                },
                {
                    PSNames.miner_uid: [1],
                    PSNames.miner_hotkey: ["hotkey1"],
                    PSNames.rema_prediction: [(0.2 + 0.3) / (2 + 3)],  # 0.5/5 = 0.1
                    PSNames.rema_peer_score: [(0.1 + 0.2) / (2 + 3)],  # 0.3/5 = 0.06
                },
            ),
            # Test case 2: Multi-group (two miners, each with two rows)
            (
                {
                    PSNames.miner_uid: [1, 1, 2, 2],
                    PSNames.miner_hotkey: ["hotkey1", "hotkey1", "hotkey2", "hotkey2"],
                    PSNames.weighted_prediction: [0.2, 0.3, 0.5, 0.5],
                    PSNames.weighted_peer_score: [0.1, 0.2, 0.4, 0.6],
                    PSNames.weight: [2, 3, 4, 6],
                },
                {
                    PSNames.miner_uid: [1, 2],
                    PSNames.miner_hotkey: ["hotkey1", "hotkey2"],
                    PSNames.rema_prediction: [
                        (0.2 + 0.3) / (2 + 3),  # 0.5/5 = 0.1 for miner 1
                        (0.5 + 0.5) / (4 + 6),  # 1/10 = 0.1 for miner 2
                    ],
                    PSNames.rema_peer_score: [
                        (0.1 + 0.2) / (2 + 3),  # 0.3/5 = 0.06 for miner 1
                        (0.4 + 0.6) / (4 + 6),  # 1/10 = 0.1 for miner 2
                    ],
                },
            ),
            # Test case 3: Empty DataFrames
            (
                {
                    PSNames.miner_uid: [],
                    PSNames.miner_hotkey: [],
                    PSNames.weighted_prediction: [],
                    PSNames.weighted_peer_score: [],
                    PSNames.weight: [],
                },
                {
                    PSNames.miner_uid: [],
                    PSNames.miner_hotkey: [],
                    PSNames.rema_prediction: [],
                    PSNames.rema_peer_score: [],
                },
            ),
        ],
    )
    def test_reduce_scored_intervals_df_parametrized(
        self, peer_scoring_task: PeerScoring, input_data, expected_data
    ):
        input_df = pd.DataFrame(input_data)

        result_df = peer_scoring_task.reduce_scored_intervals_df(input_df)

        expected_df = pd.DataFrame(expected_data)

        result_df_sorted = result_df.sort_values(
            by=[PSNames.miner_uid, PSNames.miner_hotkey]
        ).reset_index(drop=True)
        expected_df_sorted = expected_df.sort_values(
            by=[PSNames.miner_uid, PSNames.miner_hotkey]
        ).reset_index(drop=True)

        pd.testing.assert_frame_equal(result_df_sorted, expected_df_sorted)

    async def test_peer_score_event_no_intervals(
        self, peer_scoring_task: PeerScoring, db_operations, db_client
    ):
        event = EventsModel(
            unique_event_id="evt_no_intervals",
            event_id="e1",
            market_type="dummy",
            event_type="dummy",
            description="dummy event",
            metadata="{}",
            status=EventStatus.SETTLED,
            outcome="1",
            cutoff=datetime(2025, 1, 1, 8, 0, tzinfo=timezone.utc),  # before registered_date
            registered_date=datetime(2025, 1, 1, 9, 0, tzinfo=timezone.utc),
        )
        await db_operations.upsert_pydantic_events([event])
        predictions = []

        unit = peer_scoring_task
        result = await unit.peer_score_event(event, predictions)

        assert result.empty
        assert PSNames.rema_prediction in result.columns
        assert unit.errors_count == 1
        assert unit.logger.error.call_count == 2
        assert "n_intervals computed to be <= 0" == unit.logger.error.call_args_list[0].args[0]
        assert (
            "No intervals to score - event discarded."
            == unit.logger.error.call_args_list[1].args[0]
        )

        updated_events = await db_client.many("""SELECT * FROM events""", use_row_factory=True)
        assert len(updated_events) == 1
        assert updated_events[0]["status"] == str(EventStatus.DISCARDED.value)

    async def test_peer_score_event_no_miners(self, peer_scoring_task: PeerScoring):
        event = EventsModel(
            unique_event_id="evt_no_miners",
            event_id="e2",
            market_type="dummy",
            event_type="dummy",
            description="dummy event",
            metadata="{}",
            status=1,
            outcome="1",
            cutoff=datetime(2025, 1, 1, 16, 0, tzinfo=timezone.utc),
            registered_date=datetime(2025, 1, 1, 8, 0, tzinfo=timezone.utc),
        )
        predictions = []

        unit = peer_scoring_task

        # miner after cutoff
        event_cutoff_minutes = minutes_since_epoch(event.cutoff)
        event_cutoff_start_minutes = align_to_interval(event_cutoff_minutes)
        unit.miners_last_reg = pd.DataFrame(
            {
                PSNames.miner_uid: [1],
                PSNames.miner_hotkey: ["hotkey1"],
                PSNames.miner_registered_minutes: [event_cutoff_start_minutes + 1],
            }
        )
        result = await unit.peer_score_event(event, predictions)

        assert result.empty
        assert PSNames.rema_prediction in result.columns
        assert unit.logger.error.call_count >= 1
        assert unit.logger.error.call_args_list[-1].args[0] == "No miners to score."

    async def test_peer_score_event_no_predictions(self, peer_scoring_task: PeerScoring):
        event = EventsModel(
            unique_event_id="evt_no_predictions",
            event_id="e3",
            market_type="dummy",
            event_type="dummy",
            description="dummy event",
            metadata="{}",
            status=1,
            outcome="1",
            cutoff=datetime(2025, 1, 1, 16, 0, tzinfo=timezone.utc),
            registered_date=datetime(2025, 1, 1, 8, 0, tzinfo=timezone.utc),
        )
        predictions = []  # No predictions provided.

        unit = peer_scoring_task

        event_registered_minutes = minutes_since_epoch(event.registered_date)
        event_registered_start_minutes = align_to_interval(event_registered_minutes)
        unit.miners_last_reg = pd.DataFrame(
            {
                PSNames.miner_uid: [1],
                PSNames.miner_hotkey: ["hotkey1"],
                PSNames.miner_registered_minutes: [event_registered_start_minutes],
            }
        )

        # Patch prepare_predictions_df to return an empty DataFrame.
        unit.prepare_predictions_df = lambda predictions, miners: pd.DataFrame()

        result = await unit.peer_score_event(event, predictions)
        assert result.empty
        assert PSNames.rema_prediction in result.columns
        # Expect an error message indicating no predictions.
        assert unit.logger.error.call_count >= 1
        assert unit.logger.error.call_args_list[-1].args[0] == "No predictions to score."

    # Test the normal scenario where all required data is present.
    async def test_peer_score_event_normal(self, peer_scoring_task: PeerScoring):
        event = EventsModel(
            unique_event_id="evt_normal",
            event_id="e4",
            market_type="dummy",
            event_type="dummy",
            description="dummy event",
            metadata="{}",
            status=1,
            outcome="1",  # outcome text "1" -> float 1.0 -> int 1
            cutoff=datetime(2025, 1, 1, 16, 0, tzinfo=timezone.utc),
            registered_date=datetime(2025, 1, 1, 8, 0, tzinfo=timezone.utc),
        )
        # Create a dummy prediction.
        prediction = PredictionsModel(
            unique_event_id="evt_normal",
            minerHotkey="hotkey1",
            minerUid="1",
            predictedOutcome="1",
            canOverwrite=None,
            outcome="1",
            interval_start_minutes=align_to_interval(minutes_since_epoch(event.cutoff)),
            interval_agg_prediction=0.8,
            interval_count=1,
            submitted=datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc),
            blocktime=12345,
            exported=False,
        )
        predictions = [prediction]

        unit = peer_scoring_task

        event_cutoff_minutes = minutes_since_epoch(event.cutoff)
        event_cutoff_start_minutes = align_to_interval(event_cutoff_minutes)

        event_registered_minutes = minutes_since_epoch(event.registered_date)
        event_registered_start_minutes = align_to_interval(event_registered_minutes)
        unit.miners_last_reg = pd.DataFrame(
            {
                PSNames.miner_uid: [1],
                PSNames.miner_hotkey: ["hotkey1"],
                PSNames.miner_registered_minutes: [event_registered_start_minutes],
            }
        )

        unit.prepare_predictions_df = lambda predictions, miners: pd.DataFrame(
            {
                PSNames.miner_uid: [1],
                PSNames.miner_hotkey: ["hotkey1"],
                PSNames.interval_start: [event_cutoff_start_minutes],
                PSNames.interval_agg_prediction: [0.8],
            }
        )

        # Patch get_interval_scores_base to return a dummy interval scores DataFrame.
        unit.get_interval_scores_base = lambda predictions_df, miners, intervals: pd.DataFrame(
            {
                PSNames.miner_uid: [1],
                PSNames.miner_hotkey: ["hotkey1"],
                PSNames.interval_start: [event_cutoff_start_minutes],
                PSNames.interval_agg_prediction: [0.8],
            }
        )

        # Patch peer_score_intervals to return a dummy scored intervals DataFrame.
        unit.peer_score_intervals = lambda interval_scores, outcome_round: pd.DataFrame(
            {
                PSNames.miner_uid: [1],
                PSNames.miner_hotkey: ["hotkey1"],
                PSNames.peer_score: [0.1],
                PSNames.weighted_peer_score: [0.2],
                PSNames.weighted_prediction: [0.8],
            }
        )

        # Patch reduce_scored_intervals_df to return the final scores DataFrame.
        unit.reduce_scored_intervals_df = lambda scored_df: pd.DataFrame(
            {
                PSNames.miner_uid: [1],
                PSNames.miner_hotkey: ["hotkey1"],
                PSNames.rema_prediction: [0.8],
                PSNames.rema_peer_score: [0.1],
            }
        )

        result = await unit.peer_score_event(event, predictions)

        assert not result.empty
        for col in [
            PSNames.miner_uid,
            PSNames.miner_hotkey,
            PSNames.rema_prediction,
            PSNames.rema_peer_score,
        ]:
            assert col in result.columns

        assert result.iloc[0][PSNames.rema_prediction] == 0.8
        assert result.iloc[0][PSNames.rema_peer_score] == 0.1

        assert unit.errors_count == 0
        assert unit.logger.error.call_count == 0

    @pytest.mark.parametrize(
        "input_data, event_id, expected_valid_count, expected_error_messages",
        [
            # Case 1: All valid records.
            (
                {
                    PSNames.miner_uid: [1, 2],
                    PSNames.miner_hotkey: ["hk1", "hk2"],
                    PSNames.rema_prediction: [0.5, 0.6],
                    PSNames.rema_peer_score: [0.1, 0.2],
                },
                "event1",
                2,
                [],  # No errors expected.
            ),
            # Case 2: One record invalid (second record's rema_peer_score is non-numeric).
            (
                {
                    PSNames.miner_uid: [1, 2],
                    PSNames.miner_hotkey: ["hk1", "hk2"],
                    PSNames.rema_prediction: [0.5, 0.6],
                    PSNames.rema_peer_score: [0.1, "bad"],
                },
                "event2",
                1,  # Only the first record is valid.
                ["Error while creating a score record."],  # An error should be logged.
            ),
            # Case 3: All records invalid.
            (
                {
                    PSNames.miner_uid: [1],
                    PSNames.miner_hotkey: ["hk1"],
                    PSNames.rema_prediction: ["bad"],
                    PSNames.rema_peer_score: ["bad"],
                },
                "event3",
                0,  # No valid records.
                [
                    "Error while creating a score record.",
                    "No scores to export.",
                ],  # Two errors expected.
            ),
        ],
    )
    async def test_export_peer_scores_to_db(
        self,
        db_client,
        peer_scoring_task: PeerScoring,
        input_data,
        event_id,
        expected_valid_count,
        expected_error_messages,
    ):
        unit = peer_scoring_task
        df = pd.DataFrame(input_data)
        unit.spec_version = 1037

        # Reset errors_count.
        unit.errors_count = 0

        await unit.export_peer_scores_to_db(df, event_id)

        inserted_scores = await db_client.many("SELECT * FROM scores")
        if expected_valid_count > 0:
            assert len(inserted_scores) == expected_valid_count
            assert inserted_scores[0][0] == event_id
        else:
            assert len(inserted_scores) == 0

        error_calls = unit.logger.error.call_args_list
        if expected_error_messages:
            assert len(error_calls) == len(expected_error_messages)
            for i, msg in enumerate(expected_error_messages):
                assert msg == error_calls[i].args[0]
            assert unit.errors_count == len(expected_error_messages)
        else:
            assert unit.errors_count == 0
            assert len(error_calls) == 0

    async def test_e2e_run(
        self,
        peer_scoring_task: PeerScoring,
        db_client: DatabaseClient,
    ):
        # Mock dependencies
        unit = peer_scoring_task
        unit.export_scores = AsyncMock()

        # real db client
        db_ops = unit.db_operations

        await unit.run()
        # expect no miners found in the DB
        assert unit.errors_count == 1
        assert unit.logger.error.call_count == 1
        assert (
            unit.logger.error.call_args_list[0].args[0]
            == "No miners found in the DB, skipping scoring!"
        )

        # reset unit
        unit.errors_count = 0
        unit.logger.error.reset_mock()

        # insert miners
        # second miner registered after event 2 cutoff
        miners = [
            (
                "1",
                "hotkey1",
                "0.0.0.0",
                "2024-12-01",
                "100",
                False,
                False,
                "0.0.0.0",
                "100",
            ),
            (
                "2",
                "hotkey2",
                "0.0.0.0",
                "2024-12-26T12:00:00+00:00",
                "100",
                False,
                False,
                "0.0.0.0",
                "100",
            ),
        ]
        await db_ops.upsert_miners(miners)

        # no events
        await unit.run()
        assert unit.errors_count == 0
        assert unit.logger.debug.call_count == 2
        assert unit.logger.debug.call_args_list[0].args[0] == "No events to calculate peer scores."

        assert unit.miners_last_reg.index.size == 2
        assert unit.miners_last_reg.miner_uid.tolist() == [1, 2]
        assert unit.miners_last_reg[PSNames.miner_registered_minutes].tolist() == [482400, 519120]

        # reset unit
        unit.errors_count = 0
        unit.logger.debug.reset_mock()

        # insert events
        expected_event_id = "event1"
        events = [
            EventsModel(
                unique_event_id=expected_event_id,
                event_id=expected_event_id,
                market_type="truncated_market1",
                event_type="market1",
                description="desc1",
                outcome="1",
                status=3,
                metadata='{"key": "value"}',
                created_at="2024-12-02T14:30:00+00:00",
                cutoff="2024-12-27T14:30:00+00:00",
                resolved_at="2024-12-30T14:30:00+00:00",
            ),
            # keep cutoff before miner 2 reg date
            EventsModel(
                unique_event_id="event2",
                event_id="event2",
                market_type="truncated_market2",
                event_type="market2",
                description="desc2",
                outcome="0",
                status=3,
                metadata='{"key": "value"}',
                created_at="2024-12-02T14:30:00+00:00",
                cutoff="2024-12-26T11:30:00+00:00",
                resolved_at="2024-12-30T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="event3",
                event_id="event3",
                market_type="truncated_market3",
                event_type="market3",
                description="desc3",
                outcome=None,
                status=2,
                metadata='{"key": "value"}',
                created_at="2024-12-02T14:30:00+00:00",
                cutoff="2024-12-27T14:30:00+00:00",
                resolved_at="2024-12-30T14:30:00+00:00",
            ),
        ]
        await db_ops.upsert_pydantic_events(events)
        # registered_date is set by insertion - update it
        fixed_timestamp = datetime(2024, 12, 25, 12, 0, 0, tzinfo=timezone.utc).isoformat()
        await db_client.update(f"UPDATE events SET registered_date = '{fixed_timestamp}'")

        # check correct events are inserted
        events_for_scoring = await db_ops.get_events_for_scoring()
        assert len(events_for_scoring) == 2
        assert events_for_scoring[0].event_id == expected_event_id

        # no predictions, 2 events
        await unit.run()
        assert unit.logger.debug.call_count == 4
        assert (
            unit.logger.debug.call_args_list[0].args[0] == "Found events to calculate peer scores."
        )
        assert (
            unit.logger.debug.call_args_list[1].args[0] == "Calculating peer scores for an event."
        )
        assert (
            unit.logger.debug.call_args_list[3].args[0]
            == "Peer Scoring run finished. Resetting errors count."
        )
        assert unit.logger.debug.call_args_list[3].kwargs["extra"]["errors_count_in_logs"] == 2

        assert unit.logger.warning.call_count == 0
        assert unit.logger.error.call_count == 2
        assert (
            unit.logger.error.call_args_list[0].args[0]
            == "There are no predictions for a settled event - discarding."
        )
        updated_events = await db_client.many("""SELECT * FROM events""", use_row_factory=True)
        assert len(updated_events) == 3
        assert updated_events[0]["status"] == str(EventStatus.DISCARDED.value)
        assert updated_events[1]["status"] == str(EventStatus.DISCARDED.value)
        assert updated_events[2]["status"] == str(EventStatus.PENDING.value)

        # reset unit
        unit.errors_count = 0
        unit.logger.debug.reset_mock()
        unit.logger.warning.reset_mock()
        await db_client.update(
            "UPDATE events SET status = ? WHERE status = ? ",
            parameters=[EventStatus.SETTLED, EventStatus.DISCARDED],
        )

        # insert predictions
        predictions = [
            (
                "event2",
                "hotkey1",
                "1",
                None,
                519840,
                "1",
                519840,
                "1",
            ),
            (
                "event2",
                "hotkey2",
                "2",
                None,
                519840,
                "1",
                519840,
                "1",
            ),
            (
                expected_event_id,
                "hotkey2",
                "2",
                None,
                519840,
                "1",
                519840,
                "1",
            ),
            (
                expected_event_id,
                "hotkey3",
                "3",
                None,
                519840,
                "1",
                519840,
                "1",
            ),
        ]
        await db_ops.upsert_predictions(predictions)
        exp_predictions = await db_ops.get_predictions_for_scoring(
            unique_event_id=expected_event_id
        )
        exp_predictions += await db_ops.get_predictions_for_scoring(unique_event_id="event2")
        assert len(exp_predictions) == 4

        # test return empty scores df
        with patch.object(peer_scoring_task, "prepare_predictions_df", return_value=pd.DataFrame()):
            await unit.run()

        assert unit.logger.debug.call_args_list[3].kwargs["extra"]["errors_count_in_logs"] == 2
        assert unit.logger.error.call_count == 6
        assert (
            unit.logger.error.call_args_list[0].args[0]
            == "There are no predictions for a settled event - discarding."
        )
        assert unit.logger.error.call_args_list[2].args[0] == "No predictions to score."
        assert unit.logger.error.call_args_list[0].kwargs["extra"]["event_id"] == expected_event_id
        assert (
            unit.logger.error.call_args_list[3].args[0]
            == "Peer scores could not be calculated for an event."
        )
        assert unit.logger.error.call_args_list[3].kwargs["extra"]["event_id"] == expected_event_id

        # reset unit
        unit.errors_count = 0
        unit.logger.debug.reset_mock()
        unit.logger.error.reset_mock()

        # normal run
        await unit.run()

        assert unit.logger.debug.call_count == 6
        assert unit.logger.debug.call_args_list[5].kwargs["extra"]["errors_count_in_logs"] == 0
        assert unit.logger.debug.call_args_list[2].kwargs["extra"]["event_id"] == expected_event_id

        expected_scores_log_ev_1 = {
            0: {
                "miner_uid": 1,
                "miner_hotkey": "hotkey1",
                "rema_prediction": 0.01000000000000001,
                "rema_peer_score": -0.04275585662554291,
            },
            1: {
                "miner_uid": 2,
                "miner_hotkey": "hotkey2",
                "rema_prediction": 0.01911853027985871,
                "rema_peer_score": 0.04275585662554291,
            },
        }
        expected_scores_log_ev_2 = {
            0: {
                "miner_uid": 1,
                "miner_hotkey": "hotkey1",
                "rema_prediction": 0.99,
                "rema_peer_score": 0.0,
            }
        }
        df_expected_ev_1 = (
            pd.DataFrame(expected_scores_log_ev_1).sort_index().reset_index(drop=True)
        )
        df_expected_ev_2 = (
            pd.DataFrame(expected_scores_log_ev_2).sort_index().reset_index(drop=True)
        )
        actual_scores_ev_1 = unit.logger.debug.call_args_list[2].kwargs["extra"]["scores"]
        actual_scores_ev_2 = unit.logger.debug.call_args_list[4].kwargs["extra"]["scores"]

        df_actual_ev_1 = pd.DataFrame(actual_scores_ev_1).sort_index().reset_index(drop=True)
        df_actual_ev_2 = pd.DataFrame(actual_scores_ev_2).sort_index().reset_index(drop=True)

        assert_frame_equal(df_expected_ev_1, df_actual_ev_1, rtol=1e-5, atol=1e-8)
        assert_frame_equal(df_expected_ev_2, df_actual_ev_2, rtol=1e-5, atol=1e-8)

        # check events are marked as processed
        events_for_scoring = await db_ops.get_events_for_scoring()
        assert len(events_for_scoring) == 0
