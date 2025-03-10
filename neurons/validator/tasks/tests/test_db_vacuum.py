from unittest.mock import AsyncMock, MagicMock

import pytest

from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.tasks.db_vacuum import DbVacuum
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class TestDbVacuumTask:
    @pytest.fixture
    def db_operations_mock(self):
        db_operations = AsyncMock(spec=DatabaseOperations)

        return db_operations

    def test_vacuum_task_properties(self, db_operations_mock: AsyncMock):
        logger_mock = MagicMock(spec=InfiniteGamesLogger)
        interval = 300.0
        pages = 500

        vacuum_task = DbVacuum(
            interval_seconds=interval,
            db_operations=db_operations_mock,
            logger=logger_mock,
            pages=pages,
        )

        assert vacuum_task.name == "vacuum-task"
        assert vacuum_task.interval_seconds == interval
        assert vacuum_task.pages == pages
        assert vacuum_task._first_run is True

    async def test_db_vacuum_run(self, db_operations_mock: AsyncMock):
        db_operations_mock.vacuum_database = AsyncMock(return_value=None)
        logger_mock = MagicMock(spec=InfiniteGamesLogger)
        pages = 500

        vacuum_task = DbVacuum(
            interval_seconds=300.0,
            db_operations=db_operations_mock,
            logger=logger_mock,
            pages=pages,
        )

        await vacuum_task.run()

        # Assert - first run should be skipped
        db_operations_mock.vacuum_database.assert_not_awaited()

        await vacuum_task.run()

        # Assert - second run should execute vacuum
        db_operations_mock.vacuum_database.assert_awaited_once_with(pages)
