from typing import Literal

import aiohttp

EnvType = Literal["test", "prod"]


class IfGamesClient:
    __base_url: str
    __timeout: aiohttp.ClientTimeout

    def __init__(self, env: EnvType) -> None:
        self.__base_url = "https://stage.ifgames.win" if env == "test" else "https://ifgames.win"
        self.__timeout = aiohttp.ClientTimeout(total=30)  # In seconds

    async def get_events(self, from_date: int, offset: int, limit: int):
        # Check that all parameters are provided
        if from_date is None or offset is None or limit is None:
            raise ValueError("Invalid parameters")

        async with aiohttp.ClientSession(
            base_url=self.__base_url, timeout=self.__timeout
        ) as session:
            path = f"/api/v2/events?from_date={from_date}&offset={offset}&limit={limit}"

            async with session.get(path) as response:
                response.raise_for_status()

                return await response.json()

    async def get_event(self, event_id: str):
        if not isinstance(event_id, str):
            raise ValueError("Invalid parameter")

        async with aiohttp.ClientSession(
            base_url=self.__base_url, timeout=self.__timeout
        ) as session:
            path = f"/api/v2/events/{event_id}"

            async with session.get(path) as response:
                response.raise_for_status()

                return await response.json()
