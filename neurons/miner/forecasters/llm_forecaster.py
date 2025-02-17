from forecasting_tools import BinaryQuestion, QuestionState, TemplateBot

from neurons.miner.forecasters.base import BaseForecaster
from neurons.miner.models.event import MinerEvent
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class LLMForecaster(BaseForecaster):
    def __init__(self, event: MinerEvent, logger: InfiniteGamesLogger, extremize: bool = False):
        super().__init__(event, logger, extremize)
        self.bot = TemplateBot(
            research_reports_per_question=1,
            predictions_per_research_report=5,
        )

    async def _run(self) -> float | int:
        question = BinaryQuestion(
            question_text=self.event.get_description(),
            background_info=None,
            resolution_criteria=None,
            fine_print=None,
            id_of_post=0,
            state=QuestionState.OPEN,
        )
        try:
            reports = await self.bot.forecast_questions([question])
            probability = reports[0].prediction
        except Exception as e:
            self.logger.error(f"Error forecasting question with llm: {e}")
            probability = 0.5
        return probability
