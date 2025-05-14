import base64
import pickle
import tempfile
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from neurons.validator.api.routes.utils import get_lr_predictions_events
from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.score import ScoresModel
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


@pytest.fixture(scope="function")
async def db_client():
    temp_db = tempfile.NamedTemporaryFile(delete=False)
    db_path = temp_db.name
    temp_db.close()

    logger = MagicMock(spec=InfiniteGamesLogger)

    db_client = DatabaseClient(db_path, logger)

    await db_client.migrate()

    return db_client


@pytest.fixture
def db_operations(db_client: DatabaseClient):
    logger = MagicMock(spec=InfiniteGamesLogger)

    return DatabaseOperations(db_client=db_client, logger=logger)


async def test_get_lr_predictions_events(
    db_operations: DatabaseOperations, db_client: DatabaseClient
):
    unique_event_id_1 = "unique_event_id_1"
    unique_event_id_2 = "unique_event_id_2"
    unique_event_ids = [unique_event_id_1, unique_event_id_2, "fake_unique_event_id"]

    # patch api_logger
    logger = MagicMock(spec=InfiniteGamesLogger)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("neurons.validator.api.routes.utils.api_logger", logger)

        predictions = await get_lr_predictions_events(
            unique_event_ids=unique_event_ids,
            interval_start_minutes=1,
            top_n_ranks=10,
            db_operations=db_operations,
        )
        logger.error.assert_called_with("No model found for community predictions.")
        logger.error.reset_mock()
        assert predictions == {}

        # train a fake model
        X_train_fake = pd.DataFrame(
            [[i * 0.01 for i in range(100)], [1 - i * 0.01 for i in range(100)]],
            columns=[f"prediction_rank_{i}" for i in range(1, 101)],
        )
        y_train_fake = pd.Series([0, 1])

        model = LogisticRegression(solver="saga").fit(X_train_fake, y_train_fake)
        model_blob = pickle.dumps(model)
        model_blob_encoded = base64.b64encode(model_blob)
        await db_operations.save_community_predictions_model(
            name="community_prediction_lr", model_blob=model_blob_encoded
        )

        assert len(model.coef_.ravel()) == 100

        predictions = await get_lr_predictions_events(
            unique_event_ids=unique_event_ids,
            interval_start_minutes=1,
            top_n_ranks=100,
            db_operations=db_operations,
        )
        logger.error.assert_called_with("No data to make predictions.")
        logger.error.reset_mock()
        assert predictions == {}

        # insert data
        scores = [
            ScoresModel(
                event_id="event_id_1",
                miner_uid=i,
                miner_hotkey=f"hk{i}",
                prediction=0.5,
                event_score=0.5,
                spec_version=1,
            )
            for i in range(100)
        ] + [
            ScoresModel(
                event_id="event_id_1",
                miner_uid=i,
                miner_hotkey=f"hk{i}",
                prediction=0.5,
                event_score=0.5,
                spec_version=1,
            )
            for i in range(100)
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
                "1",
                10,
                i * 0.01,
                1,
                i * 0.01,
            )
            for i in range(100)
        ] + [
            (
                unique_event_id_2,
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

        predictions = await get_lr_predictions_events(
            unique_event_ids=unique_event_ids,
            interval_start_minutes=10,
            top_n_ranks=100,
            db_operations=db_operations,
        )
        logger.error.assert_not_called()
        assert len(predictions) == 2
        assert predictions[unique_event_id_1] == pytest.approx(0.88, abs=1e-1)
        assert predictions[unique_event_id_2] == pytest.approx(0.88, abs=1e-1)
