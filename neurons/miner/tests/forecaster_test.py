from unittest.mock import AsyncMock, Mock

import pytest

from neurons.miner.forecasters.base import DummyForecaster
from neurons.miner.forecasters.llm_forecaster import LLMForecaster
from neurons.miner.models.event import MinerEvent


class TestDummyForecaster:
    @pytest.fixture
    def mock_event(self):
        event = Mock(spec=MinerEvent)
        event.set_probability = AsyncMock()
        return event

    @pytest.mark.asyncio
    async def test_run_sets_probability(self, mock_event):
        resolver = DummyForecaster(mock_event, logger=Mock())
        prob = await resolver._run()

        assert prob in [0, 1]


class TestLLMForecaster:
    @pytest.fixture
    def llm_resolver(self):
        mock_event = Mock(spec=MinerEvent)
        mock_event.get_description = Mock(return_value="test_event")
        return LLMForecaster(event=mock_event, logger=Mock(), extremize=False)

    @pytest.mark.asyncio
    async def test_llm_resolver_successful_prediction(self, llm_resolver):
        # Mock the forecast_questions method to return a prediction
        mock_report = Mock()
        mock_report.prediction = 0.75
        llm_resolver.bot.forecast_questions = AsyncMock(return_value=[mock_report])

        result = await llm_resolver._run()

        assert isinstance(result, float)
        assert result == 0.75
        assert llm_resolver.bot.forecast_questions.called

    @pytest.mark.asyncio
    async def test_llm_resolver_handles_exception(self, llm_resolver):
        # Mock the forecast_questions method to raise an exception
        llm_resolver.bot.forecast_questions = AsyncMock(side_effect=Exception("API Error"))

        result = await llm_resolver._run()

        assert isinstance(result, float)
        assert result == 0.5  # Default value when exception occurs
