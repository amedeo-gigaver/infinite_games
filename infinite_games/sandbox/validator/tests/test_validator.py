import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from bittensor.core.config import Config
from bittensor.core.metagraph import MetagraphMixin

from infinite_games.sandbox.validator.validator import main


class TestValidator:
    def test_main(self):
        # Patch key dependencies inside the method
        with (
            patch("infinite_games.sandbox.validator.validator.get_config") as mock_get_config,
            patch("infinite_games.sandbox.validator.validator.Dendrite", spec=True),
            patch(
                "infinite_games.sandbox.validator.validator.Subtensor", spec=True
            ) as MockSubtensor,
            patch(
                "infinite_games.sandbox.validator.validator.DatabaseClient", spec=True
            ) as MockClient,
            patch(
                "infinite_games.sandbox.validator.validator.TasksScheduler"
            ) as MockTasksScheduler,
            patch("infinite_games.sandbox.validator.validator.logger", spec=True) as mock_logger,
            patch("infinite_games.sandbox.validator.validator.ScorePredictions", spec=True),
        ):
            # Mock Config
            mock_config = MagicMock(spec=Config)
            mock_config.get.side_effect = lambda key: {
                "netuid": "mock_netuid",
                "subtensor": MagicMock(get=lambda k: {"network": "mock_network"}[k]),
            }[key]

            mock_get_config.return_value = MagicMock(spec=Config)

            # Mock Subtensor
            mock_subtensor = MockSubtensor.return_value
            mock_subtensor.metagraph.return_value = MagicMock(spec=MetagraphMixin)

            # Mock Database Client
            mock_client = MockClient.return_value
            mock_client.migrate = AsyncMock()

            # Mock TasksScheduler
            mock_scheduler = MockTasksScheduler.return_value
            mock_scheduler.start = AsyncMock(return_value=None)

            # Mock Logger
            mock_logger.start_session = MagicMock()

            # Run the event loop to test the async function
            asyncio.run(main())

            # Verify start session called
            mock_logger.start_session.assert_called_once()

            # Verify migrate() was called
            mock_client.migrate.assert_awaited_once()

            # Verify start() was called
            mock_scheduler.start.assert_awaited_once()

            # Verify tasks
            assert mock_scheduler.add.call_count == 4
