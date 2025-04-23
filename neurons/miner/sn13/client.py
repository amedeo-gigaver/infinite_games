import asyncio
import os
import time
import typing
from datetime import datetime, timedelta, timezone

import aiohttp
from openai import AsyncOpenAI

from neurons.miner.models.event import MinerEvent
from neurons.miner.models.sn13 import SN13Response
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class Subnet13Client:
    __base_url: str
    __timeout: aiohttp.ClientTimeout
    __headers: aiohttp.typedefs.LooseHeaders
    __logger: InfiniteGamesLogger

    def __init__(self, logger: InfiniteGamesLogger) -> None:
        self.__logger = logger
        self.__base_url = "https://sn13.api.macrocosmos.ai"
        self.__timeout = aiohttp.ClientTimeout(total=90)  # In seconds

        sn13_api_key = os.getenv("SN13_API_KEY", None)
        if sn13_api_key is None:
            raise ValueError("SN13_API_KEY is not set")

        self.__headers = {
            "X-API-Key": sn13_api_key,
        }
        self.__keywords_prompt = open("./neurons/miner/sn13/keywords_prompt.txt", "r").read()

    def create_session(self, other_headers: dict = None) -> aiohttp.ClientSession:
        headers = self.__headers.copy()
        if other_headers:
            headers.update(other_headers)

        trace_config = aiohttp.TraceConfig()
        trace_config.on_request_start.append(self.on_request_start)
        trace_config.on_request_end.append(self.on_request_end)
        trace_config.on_request_exception.append(self.on_request_exception)

        return aiohttp.ClientSession(
            base_url=self.__base_url,
            timeout=self.__timeout,
            headers=headers,
            trace_configs=[trace_config],
        )

    async def on_request_start(self, _, trace_config_ctx, __):
        trace_config_ctx.start_time = time.time()

    async def on_request_end(self, _, trace_config_ctx, params: aiohttp.TraceRequestEndParams):
        elapsed_time_ms = round((time.time() - trace_config_ctx.start_time) * 1000)

        response_status = params.response.status

        extra = {
            "response_status": response_status,
            "method": params.method,
            "url": str(params.url),
            "elapsed_time_ms": elapsed_time_ms,
        }

        if response_status >= 400:
            response_message = await params.response.text()
            extra["response_message"] = response_message

            self.__logger.error("Http request failed", extra=extra)
        else:
            self.__logger.debug("Http request finished", extra=extra)

    async def on_request_exception(
        self, _, trace_config_ctx, params: aiohttp.TraceRequestExceptionParams
    ):
        exception = params.exception

        if isinstance(exception, asyncio.exceptions.CancelledError):
            return

        elapsed_time_ms = round((time.time() - trace_config_ctx.start_time) * 1000)

        extra = {
            "method": params.method,
            "url": str(params.url),
            "elapsed_time_ms": elapsed_time_ms,
        }

        self.__logger.exception("Http request exception", extra=extra)

    async def get_on_demand_data(
        self,
        event: MinerEvent,
        source: typing.Literal["X", "reddit"] = "X",
        usernames: typing.List[str] | None = None,
        keywords: typing.List[str] | None = None,
        start_date: datetime = datetime.now(timezone.utc) - timedelta(days=1),
        end_date: datetime = datetime.now(timezone.utc),
        limit: int = 10,
    ) -> SN13Response:
        self.__logger.debug(f"Getting on-demand data for event: {event.event_id}")

        data = {
            "source": source,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "limit": limit,
        }

        if usernames is not None:
            data["usernames"] = usernames

        if keywords is not None:
            data["keywords"] = keywords

        async with self.create_session({"Content-Type": "application/json"}) as session:
            path = f"/api/v1/on_demand_data_request"
            async with session.post(path, json=data) as response:
                response.raise_for_status()

                out = await response.json()
                return SN13Response(**out)

    async def get_on_demand_data_with_gpt(
        self,
        event: MinerEvent,
        source: typing.Literal["X", "reddit"] = "X",
        usernames: typing.List[str] | None = None,
        start_date: datetime = datetime.now(timezone.utc) - timedelta(days=1),
        end_date: datetime = datetime.now(timezone.utc),
        limit: int = 10,
    ) -> SN13Response:
        try:
            keywords = []
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": self.__keywords_prompt.format(
                            description=event.description,
                        ),
                    },
                ],
            )
            keywords = response.choices[0].message.content.split("keywords: ")[1].split("\n")[0]
            keywords = keywords.strip("[]")
            keywords = [keyword.strip() for keyword in keywords.split(", ")]
        except Exception as e:
            self.__logger.exception(f"Error getting keywords for event: {event.event_id}")
            raise

        response = await self.get_on_demand_data(
            event, source, usernames, keywords, start_date, end_date, limit
        )

        return response
