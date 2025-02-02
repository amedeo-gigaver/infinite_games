import random

import bittensor as bt
from forecasting_tools import BinaryQuestion, ForecastBot

from neurons.miner.main import Miner
from neurons.miner.utils.miner_cache import MinerCacheObject

DEV_MINER_UID = 93


class ForecastingToolsMiner(Miner):
    def __init__(self, config=None, forecaster: ForecastBot | None = None):
        super(Miner, self).__init__(config=config)
        self.forecaster = forecaster

    async def _generate_prediction(self, market: MinerCacheObject) -> None:
        if self.is_testnet and self.uid != DEV_MINER_UID:
            # in testnet, we just assign a random probability; do not make real API calls
            # but keep real calls for the dev miner for testing purposes
            market.event.probability = random.random()
            return
        try:
            llm_prediction = await self._forecast_with_forecast_bot(market)

            if llm_prediction is not None:
                market.event.probability = llm_prediction
            else:
                market.event.probability = 0

            bt.logging.info(
                "({}) Calculate {} prob to {} event {}, retries left: {}".format(
                    "No LLM" if llm_prediction is None else "LLM",
                    market.event.probability,
                    market.event.market_type.name,
                    market.event.event_id,
                    market.event.retries,
                )
            )
        except Exception as e:
            bt.logging.error("Failed to assign, probability, {}".format(repr(e)), exc_info=True)

    async def _forecast_with_forecast_bot(self, market: MinerCacheObject) -> float | None:
        assert self.forecaster is not None
        try:
            title = market.event.title
            resolution_criteria = market.event.description
        except Exception:
            title = market.event.description
            resolution_criteria = None

        question = BinaryQuestion(
            id_of_post=0,
            question_text=title,
            resolution_criteria=resolution_criteria,
            background_info=None,
            fine_print=None,
        )
        forecast_report = await self.forecaster.forecast_question(question)
        return forecast_report.prediction
