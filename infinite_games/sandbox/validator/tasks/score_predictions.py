import math
from datetime import datetime, now, timezone

import pandas as pd

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


class ScorePredictions(AbstractTask):
    interval: float
    page_size: int
    api_client: IfGamesClient
    db_operations: DatabaseOperations

    def __init__(
        self,
        interval_seconds: float,
        db_operations: DatabaseOperations,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        # Validate db_operations
        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        self.interval = interval_seconds
        self.db_operations = db_operations
        self.miners_last_reg = None

    @property
    def name(self):
        return "score-predictions"

    @property
    def interval_seconds(self):
        return self.interval

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
            logger.debug(f"No predictions for event {event_id}.")
            return

        scores = self.score_predictions(event, predictions)
        effective_scores = self.calculate_effective_score(scores)  # noqa

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
                "rema_brier_score",
            ]
        )
        miner_uids = [str(miner.uid) for miner in self.miners_last_reg]

        for miner_uid in miner_uids:
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
                    "rema_brier_score": miner_rema_brier_score,
                },
                ignore_index=True,
            )

            return scores

    def process_miner_event_score(self, event, miner_predictions, context) -> float:
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

        weights_brier_sum = []
        weights_sum = 0

        for interval_idx in range(context["n_intervals"]):
            interval_start = (
                context["registered_date_start_minutes"]
                + interval_idx * AGGREGATION_INTERVAL_LENGTH_MINUTES
            )
            interval_end = interval_start + AGGREGATION_INTERVAL_LENGTH_MINUTES

            if miner_reg_date > interval_end:
                ans = 0.5
            else:
                agg_predictions = miner_predictions[
                    miner_predictions["interval_start_minutes"] == interval_start
                ]["interval_agg_prediction"]

                if agg_predictions.empty:
                    weights_brier_sum.append(0)
                    continue
                else:
                    # DB primary key for predictions ensures maximum one prediction
                    ans = agg_predictions.iloc[0]

            # reverse exponential MA (rema): oldest interval gets the highest weight = 1
            wk = math.exp(-(context["n_intervals"] / (context["n_intervals"] - interval_idx)) + 1)
            weights_sum += wk
            m1_brier_score = 1 - ((ans - event.outcome) ** 2)
            weights_brier_sum.append(wk * m1_brier_score)

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
            rema_brier_score = sum(weights_brier_sum) / weights_sum

        return rema_brier_score

    def calculate_effective_score(scores):
        # TODO: implement the normalization of scores
        return scores
