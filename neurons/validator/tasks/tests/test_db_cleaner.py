from unittest.mock import AsyncMock, MagicMock

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
        logger_mock.debug.assert_called_once_with("Predictions deleted", extra={"deleted_count": 2})
