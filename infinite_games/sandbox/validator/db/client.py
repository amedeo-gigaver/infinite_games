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
