from unittest.mock import AsyncMock, MagicMock, call

import pytest

from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.tasks.db_cleaner import DbCleaner
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class TestDbCleanerTask:
    @pytest.fixture
    def db_operations_mock(self):
        db_operations = AsyncMock(spec=DatabaseOperations)

        return db_operations

    async def test_db_cleaner_run(self, db_operations_mock: AsyncMock):
        # Prepare mocks
        db_operations_mock.delete_predictions = AsyncMock(return_value=[1, 2])
        db_operations_mock.delete_scores = AsyncMock(return_value=[3, 4])
        logger_mock = MagicMock(spec=InfiniteGamesLogger)
        batch_size = 100

        db_cleaner_task = DbCleaner(
            interval_seconds=1.0,
            db_operations=db_operations_mock,
            batch_size=batch_size,
            logger=logger_mock,
        )

        # Act
        await db_cleaner_task.run()

        # Assert
        db_operations_mock.delete_predictions.assert_awaited_once_with(batch_size)
        db_operations_mock.delete_scores.assert_awaited_once_with(batch_size)
        logger_mock.debug.assert_has_calls(
            [
                call("Predictions deleted", extra={"deleted_count": 2}),
                call("Scores deleted", extra={"deleted_count": 2}),
            ]
        )

    async def test_db_cleaner_run_no_deletions(self, db_operations_mock: AsyncMock):
        # Prepare mocks
        db_operations_mock.delete_predictions = AsyncMock(return_value=[])
        db_operations_mock.delete_scores = AsyncMock(return_value=[])
        logger_mock = MagicMock(spec=InfiniteGamesLogger)
        batch_size = 100

        db_cleaner_task = DbCleaner(
            interval_seconds=1.0,
            db_operations=db_operations_mock,
            batch_size=batch_size,
            logger=logger_mock,
        )

        # Act
        await db_cleaner_task.run()

        # Assert
        db_operations_mock.delete_predictions.assert_awaited_once_with(batch_size)
        db_operations_mock.delete_scores.assert_awaited_once_with(batch_size)
        logger_mock.debug.assert_not_called()
