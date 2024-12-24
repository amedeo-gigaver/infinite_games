import time
from typing import Any, Awaitable, Callable, Iterable, Optional

import aiosqlite

from infinite_games.sandbox.validator.utils.logger.logger import AbstractLogger


class Client:
    __db_path: str
    __logger: AbstractLogger

    def __init__(self, db_path: str, logger: AbstractLogger) -> None:
        self.__db_path = db_path
        self.__logger = logger

    async def __wrap_execution(
        self, operation: Callable[[aiosqlite.Connection], Awaitable[any]]
    ) -> Awaitable[Any]:
        start_time = time.time()

        try:
            connection = await aiosqlite.connect(self.__db_path)
            return await operation(connection)
        finally:
            if connection is not None:
                await connection.close()

            elapsed_time_ms = round((time.time() - start_time) * 1000)

            log = "SQL executed"
            extra = {"elapsed_time_ms": elapsed_time_ms}

            if elapsed_time_ms > 500:
                self.__logger.warning(log, extra=extra)
            else:
                self.__logger.debug(log, extra=extra)

    async def insert(
        self, sql: str, parameters: Optional[Iterable[Any]] = None
    ) -> Iterable[aiosqlite.Row]:
        async def execute(connection: aiosqlite.Connection):
            cursor = await connection.execute(sql=sql, parameters=parameters)
            inserted = await cursor.fetchall()

            await cursor.close()
            await connection.commit()

            return inserted

        return await self.__wrap_execution(execute)

    async def script(
        self,
        sql_script: str,
    ) -> None:
        async def execute(connection: aiosqlite.Connection):
            cursor = await connection.executescript(sql_script=sql_script)

            await cursor.close()
            await connection.commit()

            return None

        return await self.__wrap_execution(execute)

    async def insert_many(self, sql: str, parameters: Iterable[Iterable[Any]]) -> None:
        async def execute(connection: aiosqlite.Connection):
            cursor = await connection.executemany(sql=sql, parameters=parameters)

            await cursor.close()
            await connection.commit()

            # Execute many does not support returning
            # https://github.com/python/cpython/issues/100021
            return None

        return await self.__wrap_execution(execute)

    async def delete(
        self, sql: str, parameters: Optional[Iterable[Any]] = None
    ) -> Iterable[aiosqlite.Row]:
        async def execute(connection: aiosqlite.Connection):
            cursor = await connection.execute(sql=sql, parameters=parameters)
            deleted = await cursor.fetchall()

            await cursor.close()
            await connection.commit()

            return deleted

        return await self.__wrap_execution(execute)

    async def update(
        self, sql: str, parameters: Optional[Iterable[Any]] = None
    ) -> Iterable[aiosqlite.Row]:
        async def execute(connection: aiosqlite.Connection):
            cursor = await connection.execute(sql=sql, parameters=parameters)
            updated = await cursor.fetchall()

            await cursor.close()
            await connection.commit()

            return updated

        return await self.__wrap_execution(execute)

    async def one(
        self, sql: str, parameters: Optional[Iterable[Any]] = None
    ) -> Optional[aiosqlite.Row]:
        async def execute(connection: aiosqlite.Connection):
            cursor = await connection.execute(sql=sql, parameters=parameters)
            row = await cursor.fetchone()

            await cursor.close()

            return row

        return await self.__wrap_execution(execute)

    async def many(
        self,
        sql: str,
        parameters: Optional[Iterable[Any]] = None,
        use_row_factory: bool = False,
    ) -> Iterable[aiosqlite.Row]:
        async def execute(connection: aiosqlite.Connection):
            if use_row_factory:
                connection.row_factory = aiosqlite.Row
            cursor = await connection.execute(sql=sql, parameters=parameters)

            # Note query should be paginated
            rows = await cursor.fetchall()

            await cursor.close()

            if len(rows) > 500:
                self.__logger.warning("Query returning many rows", extra={"rows": len(rows)})

            return rows

        return await self.__wrap_execution(execute)

    async def add_column_if_not_exists(
        self,
        table_name: str,
        column_name: str,
        column_type: str,
        default_value: str | int | float | None = None,
    ):
        response = await self.many(f"PRAGMA table_info({table_name})")

        columns = [row[1] for row in response]

        if column_name not in columns:
            alter_query = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"

            if default_value is not None:
                alter_query += f" DEFAULT {default_value}"

            await self.script(alter_query)

    async def migrate(self):
        await self.script(
            """
                CREATE TABLE IF NOT EXISTS events (
                    unique_event_id PRIMARY KEY,
                    event_id TEXT,
                    market_type TEXT,
                    registered_date DATETIME,
                    description TEXT,
                    starts DATETIME,
                    resolve_date DATETIME,
                    outcome TEXT,
                    local_updated_at DATETIME,
                    status TEXT,
                    metadata TEXT,
                    processed BOOLEAN DEFAULT false,
                    exported INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS miners (
                    miner_hotkey TEXT,
                    miner_uid TEXT,
                    node_ip TEXT,
                    registered_date DATETIME,
                    last_updated DATETIME,
                    blocktime INTEGER,
                    blocklisted boolean DEFAULT false,
                    PRIMARY KEY (miner_hotkey, miner_uid)
                );

                CREATE TABLE IF NOT EXISTS predictions (
                    unique_event_id TEXT,
                    minerHotkey TEXT,
                    minerUid TEXT,
                    predictedOutcome TEXT,
                    canOverwrite BOOLEAN,
                    outcome TEXT,
                    interval_start_minutes INTEGER,
                    interval_agg_prediction REAL,
                    interval_count INTEGER,
                    submitted DATETIME,
                    blocktime INTEGER,
                    exported INTEGER DEFAULT 0,
                    PRIMARY KEY (unique_event_id, interval_start_minutes, minerUid)
                );
            """
        )

        await self.add_column_if_not_exists(
            table_name="events",
            column_name="created_at",
            column_type="DATETIME",
            default_value=None,
        )

        await self.add_column_if_not_exists(
            table_name="events",
            column_name="cutoff",
            column_type="DATETIME",
            default_value=None,
        )

        await self.update(
            """
                UPDATE
                    events
                SET
                    cutoff = datetime(metadata->>'cutoff', 'unixepoch')
                WHERE
                    cutoff IS NULL
                    AND metadata->>'cutoff' IS NOT NULL
            """
        )

        await self.add_column_if_not_exists(
            table_name="events",
            column_name="end_date",
            column_type="DATETIME",
            default_value=None,
        )

        await self.update(
            """
                UPDATE
                    events
                SET
                    end_date = datetime(metadata->>'end_date', 'unixepoch')
                WHERE
                    end_date IS NULL
                    AND metadata->>'end_date' IS NOT NULL
            """
        )

        await self.add_column_if_not_exists(
            table_name="events",
            column_name="resolved_at",
            column_type="DATETIME",
            default_value=None,
        )
