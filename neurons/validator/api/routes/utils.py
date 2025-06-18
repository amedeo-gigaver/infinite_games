import base64
import pickle

import pandas as pd

from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.tasks.train_cp_model import MODEL_NAME, TrainCommunityPredictionModel
from neurons.validator.utils.logger.logger import api_logger


async def get_lr_predictions_events(
    unique_event_ids: list[str],
    interval_start_minutes: int,
    top_n_ranks: int,
    db_operations: DatabaseOperations,
) -> dict[str, float]:
    """
    Helper function to get the predictions for events using the LR model.
    """

    lr_model_blob = await db_operations.get_community_predictions_model(name=MODEL_NAME)
    if not lr_model_blob:
        api_logger.error("No model found for community predictions.")
        return {}

    lr_model_decoded = base64.b64decode(lr_model_blob)
    lr_model = pickle.loads(lr_model_decoded)

    raw_predictions = await db_operations.get_community_inference_dataset(
        unique_event_ids=unique_event_ids,
        interval_start_minutes=interval_start_minutes,
        top_n_ranks=top_n_ranks,
    )

    if not raw_predictions:
        api_logger.warning("No data to make predictions.")
        return {}

    df = pd.DataFrame.from_records(raw_predictions, columns=raw_predictions[0].keys())
    df = df.astype(
        {
            "prev_batch_miner_rank": "int32",
            "agg_prediction": "float32",
        }
    )

    # for compatibility with the training dataset
    df["batch_rank"] = 0
    df = df.rename(
        columns={
            "unique_event_id": "event_id",
        }
    )

    try:
        wide_df = TrainCommunityPredictionModel.pivot_top_n(
            df=df,
            n=top_n_ranks,
            train=False,
        )
    except ValueError:
        api_logger.warning("Error pivoting data for community predictions", exc_info=True)

        return {}

    X = wide_df.drop(columns=["event_id", "batch_rank"])

    predictions = lr_model.predict_proba(X)[:, 1]
    predictions_dict = dict(zip(wide_df["event_id"], predictions.tolist()))

    return predictions_dict
