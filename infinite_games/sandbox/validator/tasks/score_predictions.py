import copy
import math
from datetime import datetime, now, timezone

import bittensor as bt
import numpy as np
import pandas as pd
import torch
from bittensor.core.metagraph import MetagraphMixin

from infinite_games.sandbox.validator.db.operations import DatabaseOperations
from infinite_games.sandbox.validator.if_games.client import IfGamesClient
from infinite_games.sandbox.validator.models.event import EventsModel
from infinite_games.sandbox.validator.models.prediction import PredictionsModel
from infinite_games.sandbox.validator.scheduler.task import AbstractTask
from infinite_games.sandbox.validator.utils.common.converters import pydantic_models_to_dataframe
from infinite_games.sandbox.validator.utils.logger.logger import logger

# The base time epoch for clustering intervals.
SCORING_REFERENCE_DATE = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

# Intervals are grouped in 4-hour blocks (240 minutes).
AGGREGATION_INTERVAL_LENGTH_MINUTES = 60 * 4  # 240

RESET_INTERVAL_SECONDS = 60 * 60 * 24  # 24 hours

EXP_FACTOR_K = 30
NEURON_MOVING_AVERAGE_ALPHA = 0.8


class ScorePredictions(AbstractTask):
    interval: float
    page_size: int
    api_client: IfGamesClient
    db_operations: DatabaseOperations
    config: bt.Config

    def __init__(
        self,
        interval_seconds: float,
        db_operations: DatabaseOperations,
        metagraph: MetagraphMixin,
        config: bt.Config,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        # Validate db_operations
        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        # get current hotkeys and uids; regularly update these after each event scoring
        self.metagraph = metagraph
        self.metagraph.sync(lite=True)
        self.current_hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        self.n_hotkeys = len(self.current_hotkeys)
        self.current_uids = copy.deepcopy(self.metagraph.uids)

        self.config = config

        self.interval = interval_seconds
        self.db_operations = db_operations
        self.miners_last_reg = None

        # /home/vscode/.bittensor/miners/validator/default/netuid155/validator/state.pt
        self.state_file = self.config.neuron.full_path + "/state.pt"

        self.state = None

    @property
    def name(self):
        return "score-predictions"

    @property
    def interval_seconds(self):
        return self.interval

    def save_state(self):
        try:
            torch.save(self.state, self.state_file)
        except Exception:
            logger.exception("Failed to save state.")
        logger.debug("State saved.")

    def load_state(self) -> pd.DataFrame:
        if not self.state_file.exists():
            # reconstruct the state file
            state = {}
            state["step"] = 0
            state["hotkeys"] = self.current_hotkeys
            state["scores"] = torch.zeros((self.n_hotkeys), dtype=torch.float32)
            state["average_scores"] = torch.zeros((self.n_hotkeys), dtype=torch.float32)
            state["previous_average_scores"] = torch.zeros((self.n_hotkeys), dtype=torch.float32)
            state["scoring_iterations"] = 0
            state["latest_reset_date"] = now(timezone.utc).isoformat()
        else:
            state = torch.load(self.state_file)

        if "miner_uids" not in state:
            state["miner_uids"] = self.current_uids
            return state

    def minutes_since_epoch(self, dt: datetime) -> int:
        """Convert a given datetime to the 'minutes since the reference date'."""
        return int((dt - SCORING_REFERENCE_DATE).total_seconds()) // 60

    def align_to_interval(self, minutes_since: int) -> int:
        """
        Align a given number of minutes_since_epoch down to
        the nearest AGGREGATION_INTERVAL_LENGTH_MINUTES boundary.
        """
        return minutes_since - (minutes_since % AGGREGATION_INTERVAL_LENGTH_MINUTES)

    async def run(self):
        self.load_state()

        miners_last_reg_rows = await self.db_operations.get_miners_last_registration()
        self.miners_last_reg = pydantic_models_to_dataframe(miners_last_reg_rows)
        events_to_score = await self.db_operations.get_events_for_scoring()

        if not events_to_score:
            logger.debug("No events to score.")
            return

        for event in events_to_score:
            await self.score_event(event)

        pass

    async def score_event(self, event: EventsModel, predictions: list[PredictionsModel]):
        if not event.cutoff:
            logger.error("Event has no cutoff date.", extra={"event_id": event.unique_event_id})
            return

        event_id = event.unique_event_id

        # TODO: cleanup this after we have resolve date for all events in DB
        # https://linear.app/infinite-games/issue/INF-203/events-solved-before-cutoff
        if event.resolve_date:
            effective_cutoff = min(event.cutoff, event.resolve_date)
        else:
            effective_cutoff = min(event.cutoff, now(timezone.utc))
        # dirty: mutate the event object
        event.cutoff = effective_cutoff

        predictions = await self.db_operations.get_predictions_for_scoring(event_id)

        if not predictions:
            logger.warning(
                "There are no predictions for a settled event.",
                extra_info={"event_id": event_id, "event_cutoff": event.cutoff},
            )
            return

        scores = self.score_predictions(event, predictions)
        logger.debug("Scores calculated, sample below.", extra={"scores": scores.head(n=5)})

        norm_scores = self.normalize_scores(scores)

        self.update_daily_scores(norm_scores)

        self.check_reset_daily_scores()

        # set weights

        # export scores

    def score_predictions(self, event, predictions):
        # Convert cutoff and now to minutes since epoch, then align them to the interval start
        effective_cutoff_minutes = self.minutes_since_epoch(event.cutoff)
        effective_cutoff_start_minutes = self.align_to_interval(effective_cutoff_minutes)

        # Determine when we started, based on registered_date
        registered_date_minutes = self.minutes_since_epoch(event.cutoff)
        registered_date_start_minutes = self.align_to_interval(registered_date_minutes)

        n_intervals = (
            effective_cutoff_start_minutes - registered_date_start_minutes
        ) // AGGREGATION_INTERVAL_LENGTH_MINUTES

        scores = pd.DataFrame(
            columns=[
                "miner_uid",
                "hotkey",
                "rema_brier_score",
            ]
        )

        for miner in self.miners_last_reg.itertuples():
            miner_uid = int(miner.uid)
            miner_hotkey = miner.hotkey
            miner_predictions = pydantic_models_to_dataframe(
                [p for p in predictions if p.miner_uid == miner_uid]
            )
            context = {
                "miner_uid": miner_uid,
                "registered_date_start_minutes": registered_date_start_minutes,
                "n_intervals": n_intervals,
            }

            miner_rema_brier_score = self.process_miner_event_score(
                event, miner_predictions, context
            )

            scores = scores.append(
                {
                    "miner_uid": miner_uid,
                    "miner_hotkey": miner_hotkey,
                    "rema_brier_score": miner_rema_brier_score,
                },
                ignore_index=True,
            )

        # if any score is outside the range [0, 1] or None, log an error
        if scores["rema_brier_score"].isnull().any():
            logger.error(
                "Some Brier scores are None for an event.",
                extra={"event_id": event.unique_event_id},
            )
        if (scores["rema_brier_score"] < 0).any() or (scores["rema_brier_score"] > 1).any():
            logger.error(
                "Scores outside the range [0, 1] for an event.",
                extra={"event_id": event.unique_event_id},
            )

        return scores

    def process_miner_event_score(self, event, miner_predictions, context) -> float:
        # Removed the special treatment for azuro events, made no sense

        # clamp predictions between 0 and 1
        miner_predictions["interval_agg_prediction"] = miner_predictions[
            "interval_agg_prediction"
        ].clip(0, 1)

        # if miners could have predicted but didn't, gets a score of 0
        miner_reg_date = self.miners_last_reg[self.miners_last_reg["uid"] == context["miner_uid"]][
            "registered_date"
        ]
        if miner_reg_date < event.registered_date and not miner_predictions:
            return 0.0

        weights_brier_sum = 0
        weights_sum = 0

        for interval_idx in range(context["n_intervals"]):
            interval_start = (
                context["registered_date_start_minutes"]
                + interval_idx * AGGREGATION_INTERVAL_LENGTH_MINUTES
            )
            interval_end = interval_start + AGGREGATION_INTERVAL_LENGTH_MINUTES

            if miner_reg_date > interval_end:
                # if miner registered after the interval, gets a neutral score
                m1_brier_score = 1 - ((0.5 - event.outcome) ** 2)
            else:
                agg_predictions = miner_predictions[
                    miner_predictions["interval_start_minutes"] == interval_start
                ]["interval_agg_prediction"]

                if agg_predictions.empty:
                    # if miner should have answered but didn't, gets m1_brier_score of 0
                    m1_brier_score = 0
                else:
                    # DB primary key for predictions ensures maximum one prediction
                    ans = agg_predictions.iloc[0]
                    m1_brier_score = 1 - ((ans - event.outcome) ** 2)

            # reverse exponential MA (rema): oldest interval gets the highest weight = 1
            wk = math.exp(-(context["n_intervals"] / (context["n_intervals"] - interval_idx)) + 1)
            weights_sum += wk
            weights_brier_sum += wk * m1_brier_score

        if weights_sum < 0.01:
            logger.error(
                "Weights sum is too low for an event-miner.",
                extra={
                    "miner_uid": context["miner_uid"],
                    "event_id": event.unique_event_id,
                    "weights_sum": weights_sum,
                    "n_intervals": context["n_intervals"],
                    "miner_reg_date": miner_reg_date,
                    "event_reg_date": event.registered_date,
                },
            )
            rema_brier_score = 0.0
        else:
            rema_brier_score = weights_brier_sum / weights_sum

        return rema_brier_score

    def normalize_scores(self, scores):
        processed_scores = scores.copy()
        # TODO: check what happens if all scores are 0

        processed_scores["normalized_score"] = np.exp(EXP_FACTOR_K * processed_scores["scores"])
        logger.debug(
            "Scores exponentiation sample.", extra={"processed_scores": processed_scores.head(n=5)}
        )

        # Normalize the scores - sum cannot be 0, all elements are >= 1
        processed_scores["normalized_score"] /= processed_scores["normalized_score"].sum()
        logger.debug(
            "Scores normalization sample.", extra={"processed_scores": processed_scores.head(n=5)}
        )

        return processed_scores

    def update_state(self, norm_scores_extended: pd.DataFrame):
        # update and save the state
        self.state["scoring_iterations"] += 1
        self.state["scores"] = torch.tensor(
            norm_scores_extended["eff_scores"].values, dtype=torch.float32
        )
        self.state["average_scores"] = torch.tensor(
            norm_scores_extended["average_scores"].values, dtype=torch.float32
        )
        self.state["previous_average_scores"] = torch.tensor(
            norm_scores_extended["previous_average_scores"].values, dtype=torch.float32
        )
        self.state["miner_uids"] = torch.tensor(
            norm_scores_extended["miner_uid"].values, dtype=torch.int32
        )
        self.state["hotkeys"] = torch.tensor(norm_scores_extended["hotkey"].values)

    def update_daily_scores(self, norm_scores: pd.DataFrame):
        # if the miners are not anymore in the current uid - hotkeys, remove them
        miners_df = pd.DataFrame(
            {
                "miner_uid": self.current_uids.tolist(),
                "hotkey": self.current_hotkeys.tolist(),
            }
        )
        norm_scores = pd.merge(norm_scores, miners_df, on=["miner_uid", "hotkey"], how="inner")

        state_df = pd.DataFrame(
            {
                "miner_uid": self.state["miner_uids"].to_list(),
                "hotkey": self.state["hotkeys"].to_list(),
                "eff_scores": self.state["scores"].to_list(),
                "average_scores": self.state["average_scores"].to_list(),
                "previous_average_scores": self.state["previous_average_scores"].to_list(),
            }
        )

        norm_scores = pd.merge(norm_scores, state_df, on=["miner_uid", "hotkey"], how="left")
        # if the miner is not in the state, it gets NAs -> fill them with 0
        norm_scores.fillna(0.0, inplace=True)

        # update the daily average
        norm_scores["average_scores"] = (
            norm_scores["average_scores"] * self.state["scoring_iterations"]
            + norm_scores["normalized_score"]
        ).div(self.state["scoring_iterations"] + 1)

        # update the moving average - we could be missing the state file
        if (norm_scores["previous_average_scores"] == 0).all():
            norm_scores["eff_scores"] = (
                NEURON_MOVING_AVERAGE_ALPHA * norm_scores["average_scores"]
                + (1 - NEURON_MOVING_AVERAGE_ALPHA) * norm_scores["previous_average_scores"]
            )
        else:
            logger.warning("Missing the first iteration for moving average.")
            norm_scores["eff_scores"] = (
                NEURON_MOVING_AVERAGE_ALPHA * norm_scores["average_scores"]
                + (1 - NEURON_MOVING_AVERAGE_ALPHA) * norm_scores["eff_scores"]
            )

        self.update_state(norm_scores)
        self.save_state()

    def check_reset_daily_scores(self):
        pass
