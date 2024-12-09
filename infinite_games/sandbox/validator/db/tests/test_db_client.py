import tempfile
from unittest.mock import MagicMock

import aiosqlite
import pytest

from infinite_games.sandbox.validator.db.client import Client
from infinite_games.sandbox.validator.utils.logger.logger import AbstractLogger


class TestDbClient:
    @pytest.fixture(scope="class")
    def logger(self):
        # Mock logger

        return MagicMock(spec=AbstractLogger)

    @pytest.fixture(scope="function")
    async def client(self, logger):
        temp_db = tempfile.NamedTemporaryFile(delete=False)
        db_path = temp_db.name
        temp_db.close()

        client = Client(db_path, logger)

        # Prepare database schema
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute(
                """
                    CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, name TEXT);
                """
            )
            await conn.commit()

        return client

    async def test_insert(self, client, logger):
        sql = "INSERT INTO test_table (name) VALUES (?) returning name"
        params = ("test_insert",)
        result = await client.insert(sql, params)

        assert result == [("test_insert",)]
        logger.info.assert_called()  # Ensure logger was called

    async def test_delete(self, client, logger):
        # Insert a row to delete
        await client.insert("INSERT INTO test_table (name) VALUES ('test_delete')")

        sql = "DELETE FROM test_table WHERE name = ? returning name"
        params = ("test_delete",)
        result = await client.delete(sql, params)

        assert result == [("test_delete",)]
        logger.info.assert_called()  # Ensure logger was called

    async def test_update(self, client, logger):
        # Insert a row to update
        await client.insert("INSERT INTO test_table (name) VALUES ('test_update_1')")

        sql = "UPDATE test_table SET name = ? WHERE name = ? returning name"
        params = ("test_update_2", "test_update_1")
        result = await client.update(sql, params)

        assert result == [("test_update_2",)]
        logger.info.assert_called()  # Ensure logger was called

    async def test_one(self, client, logger):
        # Insert a row for querying
        await client.insert("INSERT INTO test_table (name) VALUES ('test_one')")

        sql = "SELECT * FROM test_table WHERE name = ?"
        params = ("test_one",)
        result = await client.one(sql, params)

        assert result == (1, "test_one")  # Verify returned row
        logger.info.assert_called()  # Ensure logger was called

    async def test_many(self, client, logger):
        # Insert multiple rows for querying
        await client.insert(
            "INSERT INTO test_table (name) VALUES ('test_many_1'), ('test_many_2')",
        )

        sql = "SELECT * FROM test_table"
        result = await client.many(sql)

        assert result == [(1, "test_many_1"), (2, "test_many_2")]  # Verify returned rows
        logger.info.assert_called()  # Ensure logger was called
        logger.warning.assert_not_called()  # No warning since rows < 100

    async def test_many_with_warning(self, client, logger):
        # Insert 101 rows for testing pagination warning
        sql = "INSERT INTO test_table (name) VALUES (?)"

        for i in range(101):
            params = (f"Name {i}",)
            await client.insert(sql, params)

        sql = "SELECT * FROM test_table"
        result = await client.many(sql)

        assert len(result) == 101  # Verify returned rows
        logger.warning.assert_called_with("Query returning 101 rows")  # Ensure warning logged

    async def test_migrate(self, client):
        await client.migrate()

        for table_name in ["events", "miners", "predictions"]:
            table = await client.one(
                "SELECT name FROM sqlite_master WHERE type='table' AND name = ?", (table_name,)
            )

            assert table is not None  # Check table was created
