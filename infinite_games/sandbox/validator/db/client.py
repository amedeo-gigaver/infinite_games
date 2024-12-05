import time
from typing import Any, Awaitable, Callable, Iterable, Optional

import aiosqlite

from infinite_games.sandbox.validator.utils.logger import AbstractLogger


class Client:
    __db_path: str
    __logger: AbstractLogger

    def __init__(self, db_path: str, logger: AbstractLogger) -> None:
        self.__db_path = db_path
        self.__logger = logger

    async def __wrap_execution(self, operation: Callable[[str], Awaitable[int]]) -> Awaitable[Any]:
        start_time = time.time()

        try:
            connection = await aiosqlite.connect(self.__db_path)
            return await operation(connection)
        finally:
            if connection is not None:
                await connection.close()

            elapsed_time = time.time() - start_time
            log = f"Query: {elapsed_time:.3f} seconds"

            if elapsed_time > 0.5:
                self.__logger.warning(log)
            else:
                self.__logger.info(log)

    async def insert(self, sql: str, parameters: Optional[Iterable[Any]] = None):
        async def execute(connection: aiosqlite.Connection):
            cursor = await connection.execute(sql=sql, parameters=parameters)
            inserted = await cursor.fetchall()

            await cursor.close()
            await connection.commit()

            return inserted

        return await self.__wrap_execution(execute)

    async def delete(self, sql: str, parameters: Optional[Iterable[Any]] = None):
        async def execute(connection: aiosqlite.Connection):
            cursor = await connection.execute(sql=sql, parameters=parameters)
            deleted = await cursor.fetchall()

            await cursor.close()
            await connection.commit()

            return deleted

        return await self.__wrap_execution(execute)

    async def update(self, sql: str, parameters: Optional[Iterable[Any]] = None):
        async def execute(connection: aiosqlite.Connection):
            cursor = await connection.execute(sql=sql, parameters=parameters)
            updated = await cursor.fetchall()

            await cursor.close()
            await connection.commit()

            return updated

        return await self.__wrap_execution(execute)

    async def one(self, sql: str, parameters: Optional[Iterable[Any]] = None):
        async def execute(connection: aiosqlite.Connection):
            cursor = await connection.execute(sql=sql, parameters=parameters)
            row = await cursor.fetchone()

            await cursor.close()

            return row

        return await self.__wrap_execution(execute)

    async def many(self, sql: str, parameters: Optional[Iterable[Any]] = None):
        async def execute(connection: aiosqlite.Connection):
            cursor = await connection.execute(sql=sql, parameters=parameters)
            # Note query should be paginated
            rows = await cursor.fetchall()

            await cursor.close()

            if len(rows) > 100:
                self.__logger.warning(f"Query returning {len(rows)} rows")

            return rows

        return await self.__wrap_execution(execute)

    async def migrate(self):
        await self.insert(
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
            """
        )
        await self.insert(
            """
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
            """
        )

        await self.insert(
            """
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
