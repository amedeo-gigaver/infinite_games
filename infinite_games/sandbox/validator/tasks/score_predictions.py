import base64
import copy
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import backoff
import bittensor as bt
import numpy as np
import pandas as pd
import requests
import torch
from bittensor.core.metagraph import MetagraphMixin
from bittensor_wallet.wallet import Wallet

from infinite_games import __spec_version__ as spec_version
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
EXPORT_SCORES_ENDPOINT = "/api/v1/validators/results"


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
        api_client: IfGamesClient,
        metagraph: MetagraphMixin,
        config: bt.Config,
        subtensor: bt.Subtensor,
        wallet: Wallet,  # type: ignore
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
        self.subtensor = subtensor
        self.wallet = wallet
        self.vali_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        self.spec_version = spec_version
        self.is_test = self.subtensor.network in ["test", "mock", "local"]
        self.base_api_url = "https://stage.ifgames.win" if self.is_test else "https://ifgames.win"
        logger.info(
            "Init info.",
            extra={
                "vali_uid": self.vali_uid,
                "spec_version": self.spec_version,
                "is_test": self.is_test,
            },
        )

        self.interval = interval_seconds
        self.db_operations = db_operations
        self.api_client = api_client
        self.miners_last_reg = None

        # /home/vscode/.bittensor/miners/validator/default/netuid155/validator/state.pt
        self.state_file = Path(self.config.neuron.full_path, "state.pt")

        # load last object state
        self.state = None
        self.load_state()

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

    def load_state(self) -> dict:
        if not self.state_file.exists():
            # reconstruct the state file
            state = {}
            state["step"] = 0  # unused but kept for compatibility
            state["hotkeys"] = self.current_hotkeys
            state["scores"] = torch.zeros((self.n_hotkeys), dtype=torch.float32)
            state["average_scores"] = torch.zeros((self.n_hotkeys), dtype=torch.float32)
            state["previous_average_scores"] = torch.zeros((self.n_hotkeys), dtype=torch.float32)
            state["scoring_iterations"] = 0
            # reset the state to the previous day midnight
            # it will be updated to the current day midnight after the first event
            state["latest_reset_date"] = (
                datetime.now(timezone.utc) - timedelta(seconds=RESET_INTERVAL_SECONDS)
            ).replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            state = torch.load(self.state_file)

        if "miner_uids" not in state:
            state["miner_uids"] = self.current_uids

        self.state = state

    def minutes_since_epoch(self, dt: datetime) -> int:
        """Convert a given datetime to the 'minutes since the reference date'."""
        return int((dt - SCORING_REFERENCE_DATE).total_seconds()) // 60

    def align_to_interval(self, minutes_since: int) -> int:
        """
        Align a given number of minutes_since_epoch down to
        the nearest AGGREGATION_INTERVAL_LENGTH_MINUTES boundary.
        """
        return minutes_since - (minutes_since % AGGREGATION_INTERVAL_LENGTH_MINUTES)

    def process_miner_event_score(
        self, event: EventsModel, miner_predictions: pd.DataFrame, context: dict
    ) -> float:
        # Removed the special treatment for azuro events, made no sense

        # outcome is text in DB :|
        outcome = float(event.outcome)

        # clamp predictions between 0 and 1
        miner_predictions["interval_agg_prediction"] = miner_predictions[
            "interval_agg_prediction"
        ].clip(0, 1)

        # if miners could have predicted but didn't, gets a score of 0
        miner_reg_date = self.miners_last_reg[self.miners_last_reg["uid"] == context["miner_uid"]][
            "registered_date"
        ].iloc[
            0
        ]  # DB primary key ensures only one row
        if miner_reg_date < event.registered_date and miner_predictions.empty:
            logger.debug(
                "Miner did not predict for an event.",
                extra={
                    "miner_uid": context["miner_uid"],
                    "event_id": event.unique_event_id,
                    "miner_reg_date": miner_reg_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "event_reg_date": event.registered_date.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            return {
                "rema_brier_score": 0.0,
                "rema_prediction": -999,
            }

        weights_brier_sum = 0
        weights_pred_sum = 0
        weights_sum = 0
        miner_reg_date_since_epoch = self.minutes_since_epoch(miner_reg_date)

        for interval_idx in range(context["n_intervals"]):
            interval_start = (
                context["registered_date_start_minutes"]
                + interval_idx * AGGREGATION_INTERVAL_LENGTH_MINUTES
            )
            interval_end = interval_start + AGGREGATION_INTERVAL_LENGTH_MINUTES

            if miner_reg_date_since_epoch > interval_end:
                # if miner registered after the interval, gets a neutral score
                ans = 0.5
            else:
                agg_predictions = miner_predictions[
                    miner_predictions["interval_start_minutes"] == interval_start
                ]["interval_agg_prediction"]

                if agg_predictions.empty:
                    # if miner should have answered but didn't, assume wrong prediction
                    # and will get a m1_brier_score of 0
                    ans = 1 - outcome
                else:
                    # DB primary key for predictions ensures maximum one prediction
                    ans = agg_predictions.iloc[0]

            m1_brier_score = 1 - ((ans - outcome) ** 2)

            # reverse exponential MA (rema): oldest interval gets the highest weight = 1
            wk = math.exp(-(context["n_intervals"] / (context["n_intervals"] - interval_idx)) + 1)
            weights_sum += wk
            weights_pred_sum += wk * ans
            weights_brier_sum += wk * m1_brier_score

        if weights_sum < 0.01:
            logger.error(
                "Weights sum is too low for an event-miner.",
                extra={
                    "miner_uid": context["miner_uid"],
                    "event_id": event.unique_event_id,
                    "weights_sum": weights_sum,
                    "n_intervals": context["n_intervals"],
                    "miner_reg_date": miner_reg_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "event_reg_date": event.registered_date.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            rema_brier_score = 0.0
            rema_prediction = -99
        else:
            rema_brier_score = weights_brier_sum / weights_sum
            rema_prediction = weights_pred_sum / weights_sum

        remas = {
            "rema_brier_score": rema_brier_score,
            "rema_prediction": rema_prediction,
        }
        return remas

    def score_predictions(
        self, event: EventsModel, predictions: list[PredictionsModel]
    ) -> pd.DataFrame:
        # Convert cutoff and now to minutes since epoch, then align them to the interval start
        effective_cutoff_minutes = self.minutes_since_epoch(event.cutoff)
        effective_cutoff_start_minutes = self.align_to_interval(effective_cutoff_minutes)

        # Determine when we started, based on registered_date
        registered_date_minutes = self.minutes_since_epoch(event.registered_date)
        registered_date_start_minutes = self.align_to_interval(registered_date_minutes)

        n_intervals = (
            effective_cutoff_start_minutes - registered_date_start_minutes
        ) // AGGREGATION_INTERVAL_LENGTH_MINUTES

        scores_ls = []

        for miner in self.miners_last_reg.itertuples():
            miner_uid = int(miner.uid)
            miner_hotkey = miner.hotkey
            miner_predictions = pydantic_models_to_dataframe(
                [p for p in predictions if int(p.minerUid) == miner_uid]
            )
            context = {
                "miner_uid": miner_uid,
                "registered_date_start_minutes": registered_date_start_minutes,
                "n_intervals": n_intervals,
            }

            miner_remas = self.process_miner_event_score(event, miner_predictions, context)

            scores_ls.append(
                {
                    "miner_uid": miner_uid,
                    "hotkey": miner_hotkey,
                    "rema_brier_score": miner_remas["rema_brier_score"],
                    "rema_prediction": miner_remas["rema_prediction"],
                },
            )

        scores_df = pd.DataFrame(scores_ls)

        if scores_df.empty:
            logger.error(
                "No scores calculated for an event.",
                extra={"event_id": event.unique_event_id},
            )
            return pd.DataFrame(
                columns=["miner_uid", "hotkey", "rema_brier_score", "rema_prediction"]
            )

        # if any score is outside the range [0, 1] or None, log an error
        if scores_df["rema_brier_score"].isnull().any():
            logger.error(
                "Some Brier scores are None for an event.",
                extra={"event_id": event.unique_event_id},
            )
        if (scores_df["rema_brier_score"] < 0).any() or (scores_df["rema_brier_score"] > 1).any():
            logger.error(
                "Scores outside the range [0, 1] for an event.",
                extra={"event_id": event.unique_event_id},
            )

        return scores_df

    def normalize_scores(self, scores: pd.DataFrame) -> pd.DataFrame:
        processed_scores = scores.copy()
        # TODO: check what happens if all scores are 0

        processed_scores["normalized_score"] = np.exp(
            EXP_FACTOR_K * processed_scores["rema_brier_score"]
        )
        logger.debug(
            "Scores exponentiation sample.", extra={"processed_scores": processed_scores.head(n=5)}
        )

        # Normalize the scores - sum cannot be 0, all elements are >= 1
        processed_scores["normalized_score"] /= processed_scores["normalized_score"].sum()
        logger.debug(
            "Scores normalization sample.", extra={"processed_scores": processed_scores.head(n=5)}
        )

        return processed_scores

    def update_daily_scores(self, norm_scores: pd.DataFrame) -> pd.DataFrame:
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
        # miner can be new -> not in the state file which can be hours old
        # it gets NAs -> fill them with 0
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

    def update_state(self, norm_scores_extended: pd.DataFrame):
        # update the state to the last scores and current hotkeys & uids
        self.metagraph.sync(lite=True)
        self.current_hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        self.n_hotkeys = len(self.current_hotkeys)
        self.current_uids = copy.deepcopy(self.metagraph.uids)
        self.state["scoring_iterations"] += 1

        # realign the scores to the current hotkeys - right join:
        # - new hotkeys get 0 scores
        # - remove the hotkeys that are not in the current hotkeys
        norm_scores_aligned = norm_scores_extended.merge(
            pd.DataFrame(
                {
                    "miner_uid": self.current_uids.tolist(),
                    "hotkey": self.current_hotkeys.tolist(),
                }
            ),
            on=["miner_uid", "hotkey"],
            how="right",
        )
        # new miners get NAs -> fill them with 0
        norm_scores_aligned.fillna(0.0, inplace=True)

        # if there are duplicates, log error
        if norm_scores_aligned.duplicated(subset=["miner_uid", "miner_hotkey"]).any():
            logger.error(
                "Duplicated miner_uid-hotkey pairs in the normalized scores.",
                extra={"norm_scores_extended": norm_scores_aligned[["miner_uid", "miner_hotkey"]]},
            )
            norm_scores_aligned.drop_duplicates(subset=["miner_uid", "miner_hotkey"], inplace=True)

        self.state["scores"] = torch.tensor(
            norm_scores_aligned["eff_scores"].values, dtype=torch.float32
        )
        self.state["average_scores"] = torch.tensor(
            norm_scores_aligned["average_scores"].values, dtype=torch.float32
        )
        self.state["previous_average_scores"] = torch.tensor(
            norm_scores_aligned["previous_average_scores"].values, dtype=torch.float32
        )
        self.state["miner_uids"] = torch.tensor(
            norm_scores_aligned["miner_uid"].values, dtype=torch.long
        )
        self.state["hotkeys"] = torch.tensor(norm_scores_aligned["hotkey"].values)

        # this is what we will export to the Clickhouse DB
        # reflects what we use for the weights
        return norm_scores_aligned

    def set_weights(self):
        # re-normalize the scores for the weights
        raw_weights = torch.nn.functional.normalize(self.state["scores"], p=1, dim=0)

        # this is only for logging purposes
        sorted_indices = torch.argsort(raw_weights, descending=True)
        logger.debug(
            "Top 10 and bottom 10 weights.",
            extra={
                "top_10_weights": raw_weights[sorted_indices][:10].tolist(),
                "top_10_uids": self.state["miner_uids"][sorted_indices][:10].tolist(),
                "bottom_10_weights": raw_weights[sorted_indices][-10:].tolist(),
                "bottom_10_uids": self.state["miner_uids"][sorted_indices][-10:].tolist(),
            },
        )

        # set the weights in the metagraph
        processed_uids, processed_weights = bt.utils.weights_utils.process_weights(
            uids=self.state["miner_uids"].to("cpu"),
            weights=raw_weights.to("cpu"),
            metagraph=self.metagraph,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
        )

        if processed_uids is None or processed_weights is None:
            logger.error(
                "Failed to process the weights - received None.",
                extra={"processed_uids": processed_uids, "processed_weights": processed_weights},
            )
            return

        if processed_uids != self.state["miner_uids"]:
            logger.error(
                "Processed UIDs do not match the original UIDs.",
                extra={"processed_uids": processed_uids, "original_uids": self.state["miner_uids"]},
            )
            return

        if processed_weights != raw_weights:
            logger.warning(
                "Processed weights do not match the original weights.",
                extra={"processed_weights": processed_weights, "original_weights": raw_weights},
            )

        successful, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=processed_uids,
            weights=processed_weights,
            version_key=self.spec_version,
            wait_for_inclusion=False,
            wait_for_finalization=False,
            max_retries=5,
        )

        if not successful:
            logger.error(
                "Failed to set the weights.",
                extra={
                    "msg": msg,
                    "processed_uids": processed_uids,
                    "processed_weights": processed_weights,
                },
            )
        else:
            logger.debug("Weights set successfully.")

    @backoff.on_exception(
        backoff.expo,
        requests.exceptions.RequestException,
        max_tries=3,
        factor=2,
        on_backoff=lambda details: logger.warning(
            "Retrying export scores.",
            extra={"details": details},
        ),
        on_giveup=lambda details: logger.error(
            "Failed to export scores.",
            extra={"details": details},
        ),
    )
    def export_scores(self, event: EventsModel, final_scores: pd.DataFrame):
        scores_df = final_scores[
            ["miner_uid", "hotkey", "rema_brier_score", "rema_prediction", "eff_scores"]
        ].copy()
        scores_df.rename(
            columns={
                "hotkey": "miner_hotkey",
                "rema_brier_score": "miner_score",
                "rema_prediction": "prediction",
                "eff_scores": "miner_effective_score",
            },
            inplace=True,
        )
        scores_df["event_id"] = event.unique_event_id
        scores_df["provider_type"] = event.market_type
        scores_df["title"] = event.description[:50]  # as in the original
        scores_df["description"] = event.description
        scores_df["category"] = "event"  # as in the original
        scores_df["start_date"] = event.starts.isoformat() if event.starts else None
        scores_df["end_date"] = (
            event.resolve_date.isoformat() if event.resolve_date else None
        )  # as in the original
        scores_df["resolve_date"] = event.resolve_date.isoformat() if event.resolve_date else None
        scores_df["settle_date"] = event.cutoff.isoformat()  # as in the original
        scores_df["answer"] = float(event.outcome)
        scores_df["validator_hotkey"] = self.wallet.get_hotkey().ss58_address
        scores_df["validator_uid"] = int(self.vali_uid)
        scores_df["metadata"] = event.metadata
        scores_df["spec_version"] = str(self.spec_version)

        body = {
            "results": scores_df.to_dict(orient="records"),
        }

        hk = self.wallet.get_hotkey()
        signed = base64.b64encode(hk.sign(json.dumps(body))).decode("utf-8")
        signing_headers = {
            "Authorization": f"Bearer {signed}",
            "Validator": hk.ss58_address,
        }
        _ = self.api_client.post_scores(scores=body, signing_headers=signing_headers)

    def check_reset_daily_scores(self):
        now_dt = datetime.now(timezone.utc)
        seconds_since_reset = (now_dt - self.state["latest_reset_date"]).total_seconds()

        # change previous logic - adjust to reset to every midnight
        if seconds_since_reset <= RESET_INTERVAL_SECONDS:
            return

        today_midnight = now_dt.replace(hour=0, minute=0, second=0, microsecond=0)

        # if all average scores are 0
        if (self.state["average_scores"] == 0).all():
            logger.error("Reset daily scores: average scores are 0, not resetting.")
        else:
            logger.debug(
                "Resetting daily scores.", extra={"seconds_since_reset": seconds_since_reset}
            )
            self.state["previous_average_scores"] = self.state["average_scores"]
            self.state["average_scores"] = torch.zeros((self.n_hotkeys), dtype=torch.float32)
            self.state["latest_reset_date"] = today_midnight
            self.state["scoring_iterations"] = 0
            self.save_state()

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
            effective_cutoff = min(event.cutoff, datetime.now(timezone.utc))
        # dirty: mutate the event object
        event.cutoff = effective_cutoff

        predictions = await self.db_operations.get_predictions_for_scoring(event_id)

        if not predictions:
            logger.warning(
                "There are no predictions for a settled event.",
                extra_info={"event_id": event_id, "event_cutoff": event.cutoff},
            )
            return

        scores = self.score_predictions(event=event, predictions=predictions)
        logger.debug("Scores calculated, sample below.", extra={"scores": scores.head(n=5)})

        norm_scores = self.normalize_scores(scores=scores)

        norm_scores_extended = self.update_daily_scores(norm_scores=norm_scores)

        norm_scores_aligned = self.update_state(norm_scores_extended=norm_scores_extended)

        self.save_state()

        self.set_weights()

        self.db_operations.mark_event_as_processed(unique_event_id=event.unique_event_id)

        self.export_scores(event=event, final_scores=norm_scores_aligned)

        self.db_operations.mark_event_as_exported(unique_event_id=event.unique_event_id)

        self.check_reset_daily_scores()

    async def run(self):
        miners_last_reg_rows = await self.db_operations.get_miners_last_registration()
        self.miners_last_reg = pydantic_models_to_dataframe(miners_last_reg_rows)
        events_to_score = await self.db_operations.get_events_for_scoring()

        if not events_to_score:
            logger.debug("No events to score.")
            return
        else:
            logger.debug("Found events to score.", extra={"n_events": len(events_to_score)})

        for event in events_to_score:
            await self.score_event(event)

        logger.debug("All events scored, weights set, scores exported.")
