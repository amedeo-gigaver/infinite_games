import tempfile
from unittest.mock import MagicMock

import pytest

from infinite_games.sandbox.validator.db.client import DatabaseClient
from infinite_games.sandbox.validator.utils.logger.logger import InfiniteGamesLogger


class TestDbClient:
    @pytest.fixture(scope="class")
    def logger(self):
        # Mock logger

        return MagicMock(spec=InfiniteGamesLogger)

    @pytest.fixture(scope="function")
    async def client(self, logger):
        temp_db = tempfile.NamedTemporaryFile(delete=False)
        db_path = temp_db.name
        temp_db.close()

        client = DatabaseClient(db_path, logger)

        # Prepare database schema
        await client.script(
            """
                CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, name TEXT);
            """
        )

        return client

    async def test_add_column_if_not_exists(self, client):
        await client.add_column_if_not_exists(
            table_name="test_table",
            column_name="created_at",
            column_type="DATETIME",
            default_value=None,
        )

        response = await client.add_column_if_not_exists(
            table_name="test_table",
            column_name="created_at",
            column_type="DATETIME",
            default_value=None,
        )

        # No error thrown
        assert response is None

    async def test_insert(self, client, logger):
        sql = "INSERT INTO test_table (name) VALUES (?) returning name"
        params = ("test_insert",)
        result = await client.insert(sql, params)

        assert result == [("test_insert",)]
        logger.debug.assert_called()  # Ensure logger was called

    async def test_insert_many(self, client, logger):
        sql = "INSERT INTO test_table (name) VALUES (?)"
        params = [("test_insert_1",), ("test_insert_2",)]

        result = await client.insert_many(sql, params)
        assert result is None

        sql = "SELECT * FROM test_table"
        result = await client.many(sql)

        assert result == [(1, "test_insert_1"), (2, "test_insert_2")]
        logger.debug.assert_called()  # Ensure logger was called

    async def test_insert_many_no_params(self, client, logger):
        sql = "INSERT INTO test_table (name) VALUES (?)"
        params = []

        result = await client.insert_many(sql, params)
        assert result is None

        sql = "SELECT * FROM test_table"
        result = await client.many(sql)

        assert result == []
        logger.debug.assert_called()  # Ensure logger was called

    async def test_delete(self, client, logger):
        # Insert a row to delete
        await client.insert("INSERT INTO test_table (name) VALUES ('test_delete')")

        sql = "DELETE FROM test_table WHERE name = ? returning name"
        params = ("test_delete",)
        result = await client.delete(sql, params)

        assert result == [("test_delete",)]
        logger.debug.assert_called()  # Ensure logger was called

    async def test_update(self, client, logger):
        # Insert a row to update
        await client.insert("INSERT INTO test_table (name) VALUES ('test_update_1')")

        sql = "UPDATE test_table SET name = ? WHERE name = ? returning name"
        params = ("test_update_2", "test_update_1")
        result = await client.update(sql, params)

        assert result == [("test_update_2",)]
        logger.debug.assert_called()  # Ensure logger was called

    async def test_one(self, client, logger):
        # Insert a row for querying
        await client.insert("INSERT INTO test_table (name) VALUES ('test_one')")

        sql = "SELECT * FROM test_table WHERE name = ?"
        params = ("test_one",)
        result = await client.one(sql, params)

        assert result == (1, "test_one")  # Verify returned row
        logger.debug.assert_called()  # Ensure logger was called

    async def test_many(self, client, logger):
        # Insert multiple rows for querying
        await client.insert(
            "INSERT INTO test_table (name) VALUES ('test_many_1'), ('test_many_2')",
        )

        sql = "SELECT * FROM test_table"
        result = await client.many(sql)

        assert result == [(1, "test_many_1"), (2, "test_many_2")]  # Verify returned rows
        logger.debug.assert_called()  # Ensure logger was called
        logger.warning.assert_not_called()  # No warning since rows are few

    async def test_many_with_warning(self, client, logger):
        # Insert many rows for testing pagination warning
        sql = "INSERT INTO test_table (name) VALUES (?)"

        rows_to_insert = 501

        for i in range(rows_to_insert):
            params = (f"Name {i}",)
            await client.insert(sql, params)

        sql = "SELECT * FROM test_table"
        result = await client.many(sql)

        assert len(result) == rows_to_insert  # Verify returned rows
        logger.warning.assert_called_with(
            "Query returning many rows", extra={"rows": rows_to_insert}
        )  # Ensure warning logged

    async def test_migrate(self, client):
        await client.migrate()

        for table_name in ["events", "miners", "predictions"]:
            table = await client.one(
                "SELECT name FROM sqlite_master WHERE type='table' AND name = ?", (table_name,)
            )

            assert table is not None  # Check table was created
