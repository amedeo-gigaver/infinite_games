from forecasting_tools import BinaryQuestion, QuestionState, TemplateBot

from neurons.miner.forecasters.base import BaseForecaster
from neurons.miner.models.event import MinerEvent
from neurons.miner.models.sn13 import SN13Response
from neurons.miner.sn13.client import Subnet13Client
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class LLMForecaster(BaseForecaster):
    def __init__(self, event: MinerEvent, logger: InfiniteGamesLogger, extremize: bool = False):
        super().__init__(event, logger, extremize)
        self.bot = TemplateBot(
            research_reports_per_question=1,
            predictions_per_research_report=5,
        )

    async def _get_question(self) -> BinaryQuestion:
        question = BinaryQuestion(
            question_text=self.event.get_description(),
            background_info=None,
            resolution_criteria=None,
            fine_print=None,
            id_of_post=0,
            state=QuestionState.OPEN,
        )
        return question

    async def _run(self) -> tuple[float | int, str | None]:
        try:
            question = await self._get_question()
            reports = await self.bot.forecast_questions([question])
            probability = reports[0].prediction
        except Exception as e:
            self.logger.error(f"Error forecasting question with llm: {e}")
            probability = 0.5

        return probability, None


class LLMForecasterWithSN13(LLMForecaster):
    def __init__(
        self,
        event: MinerEvent,
        logger: InfiniteGamesLogger,
        sn13_client: Subnet13Client,
        extremize: bool = False,
    ):
        super().__init__(event, logger, extremize)
        self.sn13_client = sn13_client

    async def _get_question(self) -> BinaryQuestion:
        response: SN13Response = await self.sn13_client.get_on_demand_data_with_gpt(self.event)
        content = ". ".join([data.content for data in response.data])
        self.logger.debug(f"Content for question {self.event.get_event_id()}: {content}")
        question = BinaryQuestion(
            question_text=self.event.get_description(),
            background_info=content,
            resolution_criteria=None,
            fine_print=None,
            id_of_post=0,
            state=QuestionState.OPEN,
        )
        return question
