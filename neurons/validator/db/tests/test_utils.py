import tempfile
from unittest.mock import MagicMock

import pytest

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class TestDbOperationsBase:
    @pytest.fixture(scope="function")
    async def db_client(self):
        temp_db = tempfile.NamedTemporaryFile(delete=False)
        db_path = temp_db.name
        temp_db.close()

        logger = MagicMock(spec=InfiniteGamesLogger)

        db_client = DatabaseClient(db_path, logger)

        await db_client.migrate()

        return db_client

    @pytest.fixture
    async def db_operations(self, db_client: DatabaseClient):
        logger = MagicMock(spec=InfiniteGamesLogger)

        db_operations = DatabaseOperations(db_client=db_client, logger=logger)

        return db_operations
