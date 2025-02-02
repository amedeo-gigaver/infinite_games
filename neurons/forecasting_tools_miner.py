# -- DO NOT TOUCH BELOW - ENV SET --
# flake8: noqa: E402
import os
import sys

# Force torch - must be set before importing bittensor
os.environ["USE_TORCH"] = "1"

# Add the parent directory of the script to PYTHONPATH
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
# -- DO NOT TOUCH ABOVE --

import time
from datetime import datetime

from bittensor import logging
from forecasting_tools import (
    BinaryQuestion,
    Gpt4o,
    MetaculusQuestion,
    ReasonedPrediction,
    SmartSearcher,
    TemplateBot,
)
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents

from neurons.miner.forecasting_tools_main import ForecastingToolsMiner


class MyCustomBot(TemplateBot):
    async def run_research(self, question: MetaculusQuestion) -> str:
        searcher = SmartSearcher(num_searches_to_run=3, num_sites_per_search=10)

        prompt = clean_indents(
            f"""
            Analyze this forecasting question:
            1. Filter for recent events in the past 6 months
            2. Look for current trends and data
            3. Find historical analogies and base rates

            Question: {question.question_text}

            Background Info: {question.background_info}
            Resolution Criteria: {question.resolution_criteria}
            """
        )

        report = await searcher.invoke(prompt)
        return report

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        # TODO: Only add to the example the fields that translate from events
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            background:
            {question.background_info}
            {question.resolution_criteria}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) What the outcome would be if nothing changed.
            (c) What you would forecast if there was only a quarter of the time left.
            (d) What you would forecast if there was 4x the time left.

            You write your rationale and then the last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        reasoning = await Gpt4o(temperature=0.1).invoke(prompt)
        prediction = self._extract_forecast_from_binary_rationale(
            reasoning, max_prediction=0.99, min_prediction=0.01
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)


if __name__ == "__main__":
    start_time = time.time()
    forecast_bot = MyCustomBot(
        research_reports_per_question=1,
        predictions_per_research_report=5,
    )
    with ForecastingToolsMiner(forecaster=forecast_bot) as miner:
        while True:
            logging.debug(f"Miner running for {time.time() - start_time:.1f} seconds")
            time.sleep(5)
