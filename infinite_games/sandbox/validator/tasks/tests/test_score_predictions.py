import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import bittensor as bt
import pandas as pd
import pytest
import torch
from bittensor.core.metagraph import MetagraphMixin
from bittensor_wallet.wallet import Wallet
from freezegun import freeze_time

from infinite_games.sandbox.validator.db.client import Client
from infinite_games.sandbox.validator.db.operations import DatabaseOperations
from infinite_games.sandbox.validator.if_games.client import IfGamesClient
from infinite_games.sandbox.validator.models.event import EventsModel
from infinite_games.sandbox.validator.models.prediction import PredictionsModel
from infinite_games.sandbox.validator.tasks.score_predictions import ScorePredictions
from infinite_games.sandbox.validator.utils.logger.logger import AbstractLogger

CURRENT_DIR = Path(__file__).parent


@freeze_time("2024-12-27 07:00:00")
class TestScorePredictions:
    @pytest.fixture(scope="function", autouse=True)
    async def setup_test_dir(self):
        test_dir = CURRENT_DIR / "test_dir"
        test_dir.mkdir(exist_ok=True)
        yield test_dir
        for file in test_dir.iterdir():
            file.unlink()
        test_dir.rmdir()

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
    def db_operations(self, db_client: Client):
        return DatabaseOperations(db_client=db_client)

    @pytest.fixture
    def score_predictions_task(self, db_operations: DatabaseOperations, setup_test_dir):
        api_client = IfGamesClient(env="test", logger=MagicMock(spec=AbstractLogger))
        metagraph = MagicMock(spec=MetagraphMixin)
        config = MagicMock(spec=bt.Config)
        config.neuron = MagicMock()
        wallet = MagicMock(spec=Wallet)
        wallet.hotkey = MagicMock()
        subtensor = MagicMock(spec=bt.Subtensor)

        metagraph.uids = [1, 2, 3]
        metagraph.hotkeys = ["hotkey1", "hotkey2", "hotkey3"]
        wallet.hotkey.ss58_address = "hotkey2"
        subtensor.network = "mock"

        config.neuron.full_path = setup_test_dir
        return ScorePredictions(
            interval_seconds=60.0,
            db_operations=db_operations,
            api_client=api_client,
            metagraph=metagraph,
            config=config,
            wallet=wallet,
            subtensor=subtensor,
        )

    def test_init(self, score_predictions_task: ScorePredictions):
        unit = score_predictions_task

        assert isinstance(unit, ScorePredictions)
        assert unit.current_uids == [1, 2, 3]
        assert unit.current_hotkeys == ["hotkey1", "hotkey2", "hotkey3"]
        assert unit.n_hotkeys == 3
        assert unit.wallet.hotkey.ss58_address == "hotkey2"
        assert unit.vali_uid == 1
        assert unit.is_test is True
        assert unit.state_file == Path(CURRENT_DIR, "test_dir/state.pt")

    def test_load_save_state(self, score_predictions_task: ScorePredictions):
        unit = score_predictions_task

        # init from missing file
        assert unit.state_file.exists() is False

        assert unit.state is not None
        assert unit.state["step"] == 0
        assert unit.state["hotkeys"] == unit.current_hotkeys
        assert isinstance(unit.state["scores"], torch.Tensor)
        assert unit.state["scores"].tolist() == [0.0, 0.0, 0.0]
        assert unit.state["average_scores"].tolist() == [0.0, 0.0, 0.0]
        assert unit.state["previous_average_scores"].tolist() == [0.0, 0.0, 0.0]
        assert unit.state["scoring_iterations"] == 0
        assert unit.state["latest_reset_date"] == datetime(
            2024, 12, 26, 0, 0, 0, 0, tzinfo=timezone.utc
        )

        # adjust and save state
        unit.state["scoring_iterations"] = 10
        unit.state["scores"] = torch.Tensor([1.0, 2.0, 3.0])
        unit.state["latest_reset_date"] = datetime(2024, 12, 27, 0, 0, 0, 0, tzinfo=timezone.utc)

        unit.save_state()
        unit.state = None
        unit.load_state()
        assert unit.state["scoring_iterations"] == 10
        assert isinstance(unit.state["scores"], torch.Tensor)
        assert unit.state["scores"].tolist() == [1.0, 2.0, 3.0]
        assert unit.state["latest_reset_date"] == datetime(
            2024, 12, 27, 0, 0, 0, 0, tzinfo=timezone.utc
        )

    def test_minutes_since_epoch(self, score_predictions_task: ScorePredictions):
        unit = score_predictions_task

        assert unit.minutes_since_epoch(datetime(2024, 12, 27, 0, 1, 0, 0, timezone.utc)) == 519841

    def test_align_to_interval(self, score_predictions_task: ScorePredictions):
        unit = score_predictions_task

        assert unit.align_to_interval(519841) == 519840
        assert unit.align_to_interval(520079) == 519840
        assert unit.align_to_interval(520080) == 520080

    @pytest.mark.parametrize(
        "predictions, context, miner_reg_date, expected_rema_brier_score, expected_rema_prediction",
        [
            # Case 0: Miner registered before event and not scored -> 0 Brier score
            (
                pd.DataFrame(
                    {
                        "interval_start_minutes": [],
                        "interval_agg_prediction": [],
                    }
                ),
                {
                    "miner_uid": "miner_1",
                    "n_intervals": 3,
                    "registered_date_start_minutes": 0,
                },
                datetime(2024, 12, 26, 0, 0, 0, 0, timezone.utc),
                0.0,
                -999,
            ),
            # Case 1: Miner registered after event, neutral score
            (
                pd.DataFrame(
                    {
                        "interval_start_minutes": [],
                        "interval_agg_prediction": [],
                    }
                ),
                {
                    "miner_uid": "miner_1",
                    "n_intervals": 3,
                    "registered_date_start_minutes": 0,
                },
                datetime(2024, 12, 27, 1, 0, 0, 0, timezone.utc),
                0.75,
                0.5,
            ),
            # Case 2: Predictions with valid intervals
            (
                pd.DataFrame(
                    {
                        "interval_start_minutes": [519840, 520080, 520320],
                        "interval_agg_prediction": [0.8, 0.6, 0.4],
                    }
                ),
                {
                    "miner_uid": "miner_1",
                    "n_intervals": 3,
                    "registered_date_start_minutes": 519840,
                },
                datetime(2024, 12, 26, 0, 0, 0, 0, timezone.utc),
                0.8933525,  # Expected Brier score (example calculation)
                0.6992802,  # Weighted average prediction
            ),
            # Case 3: Missing predictions for some intervals
            (
                pd.DataFrame(
                    {
                        "interval_start_minutes": [519840, 520320],
                        "interval_agg_prediction": [1.0, 1.0],
                    }
                ),
                {
                    "miner_uid": "miner_1",
                    "n_intervals": 3,
                    "registered_date_start_minutes": 519840,
                },
                datetime(2024, 12, 26, 0, 0, 0, 0, timezone.utc),
                0.6517926,  # Example with missing intervals handled
                0.6517926,
            ),
            # Case 4: Predictions with exaggerated values
            (
                pd.DataFrame(
                    {
                        "interval_start_minutes": [519840, 520080, 520320],
                        "interval_agg_prediction": [100.0, 299.0, 1000.0],
                    }
                ),
                {
                    "miner_uid": "miner_1",
                    "n_intervals": 3,
                    "registered_date_start_minutes": 519840,
                },
                datetime(2024, 12, 26, 0, 0, 0, 0, timezone.utc),
                1.0,
                1.0,
            ),
        ],
    )
    def test_process_miner_event_score(
        self,
        score_predictions_task,
        predictions,
        context,
        miner_reg_date,
        expected_rema_brier_score,
        expected_rema_prediction,
    ):
        unit = score_predictions_task

        # Mock event
        event = EventsModel(
            unique_event_id="unique_event_id",
            event_id="event_id",
            market_type="market_type",
            event_type="event_type",
            registered_date=datetime(2024, 12, 27, 0, 0, 0, 0, timezone.utc),
            description="description",
            outcome="1",
            status=3,
            metadata="metadata",
            created_at=datetime(2024, 12, 27, 0, 0, 0, 0, timezone.utc),
        )

        # Add miner's registration mock
        unit.miners_last_reg = pd.DataFrame(
            {
                "uid": ["miner_1"],
                "registered_date": [miner_reg_date],
            }
        )

        # Call the function
        result = unit.process_miner_event_score(event, predictions, context)

        assert result["rema_brier_score"] == pytest.approx(expected_rema_brier_score, rel=1e-6)
        assert result["rema_prediction"] == pytest.approx(expected_rema_prediction, rel=1e-6)

    @pytest.mark.parametrize(
        "predictions, miners_last_reg, expected_scores",
        [
            # Case 0: No miners and no predictions
            (
                [],  # Empty predictions
                pd.DataFrame(columns=["uid", "hotkey", "registered_date"]),
                pd.DataFrame(
                    columns=["miner_uid", "hotkey", "rema_brier_score", "rema_prediction"]
                ),
            ),
            # Case 1: One miner with valid predictions
            (
                [
                    PredictionsModel(
                        unique_event_id="unique_event_id",
                        minerUid="1",
                        interval_start_minutes=519840,
                        interval_agg_prediction=0.8,
                    ),
                    PredictionsModel(
                        unique_event_id="unique_event_id",
                        minerUid="1",
                        interval_start_minutes=520080,
                        interval_agg_prediction=0.6,
                    ),
                    PredictionsModel(
                        unique_event_id="unique_event_id",
                        minerUid="1",
                        interval_start_minutes=520320,
                        interval_agg_prediction=0.4,
                    ),
                ],
                pd.DataFrame(
                    {
                        "uid": [1],
                        "hotkey": ["hotkey_1"],
                        "registered_date": [datetime(2024, 12, 26, 0, 0, 0, 0, timezone.utc)],
                    }
                ),
                pd.DataFrame(
                    {
                        "miner_uid": [1],
                        "hotkey": ["hotkey_1"],
                        "rema_brier_score": [0.8933525],
                        "rema_prediction": [0.6992802],
                    }
                ),
            ),
            # Case 2: Multiple miners with mixed predictions
            (
                [
                    PredictionsModel(
                        unique_event_id="unique_event_id",
                        minerUid="1",
                        interval_start_minutes=519840,
                        interval_agg_prediction=1.0,
                    ),
                    PredictionsModel(
                        unique_event_id="unique_event_id",
                        minerUid="1",
                        interval_start_minutes=520080,
                        interval_agg_prediction=1.0,
                    ),
                    PredictionsModel(
                        unique_event_id="unique_event_id",
                        minerUid="1",
                        interval_start_minutes=520320,
                        interval_agg_prediction=1.0,
                    ),
                    PredictionsModel(
                        unique_event_id="unique_event_id",
                        minerUid="2",
                        interval_start_minutes=519840,
                        interval_agg_prediction=0.4,
                    ),
                ],
                pd.DataFrame(
                    {
                        "uid": [1, 2],
                        "hotkey": ["hotkey_1", "hotkey_2"],
                        "registered_date": [
                            datetime(2024, 12, 26, 0, 0, 0, 0, timezone.utc),
                            datetime(2024, 12, 26, 0, 0, 0, 0, timezone.utc),
                        ],
                    }
                ),
                pd.DataFrame(
                    {
                        "miner_uid": [1, 2],
                        "hotkey": ["hotkey_1", "hotkey_2"],
                        "rema_brier_score": [
                            1.0,
                            0.36742,
                        ],
                        "rema_prediction": [
                            1.0,
                            0.229638,
                        ],
                    }
                ),
            ),
        ],
    )
    def test_score_predictions_method(
        self, score_predictions_task, predictions, miners_last_reg, expected_scores
    ):
        unit = score_predictions_task

        event = EventsModel(
            unique_event_id="unique_event_id",
            event_id="event_id",
            market_type="market_type",
            event_type="event_type",
            registered_date=datetime(2024, 12, 27, 0, 0, 0, 0, timezone.utc),
            cutoff=datetime(2024, 12, 27, 12, 0, 0, 0, timezone.utc),
            description="description",
            outcome="1",
            status=3,
            metadata="metadata",
            created_at=datetime(2024, 12, 27, 0, 0, 0, 0, timezone.utc),
        )

        # Mock miners_last_reg in the task
        unit.miners_last_reg = miners_last_reg

        # Call the method
        result = unit.score_predictions(event, predictions)

        # Compare the resulting DataFrame to the expected DataFrame
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True),
            expected_scores.reset_index(drop=True),
            check_exact=False,
        )
