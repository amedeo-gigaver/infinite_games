import time
from typing import Literal

import aiohttp
import aiohttp.typedefs

from infinite_games import __version__
from infinite_games.sandbox.validator.utils.git import commit_short_hash
from infinite_games.sandbox.validator.utils.logger.logger import AbstractLogger

EnvType = Literal["test", "prod"]


class IfGamesClient:
    __base_url: str
    __timeout: aiohttp.ClientTimeout
    __headers: aiohttp.typedefs.LooseHeaders
    __logger: AbstractLogger

    def __init__(self, env: EnvType, logger: AbstractLogger) -> None:
        self.__logger = logger
        self.__base_url = "https://stage.ifgames.win" if env == "test" else "https://ifgames.win"
        self.__timeout = aiohttp.ClientTimeout(total=90)  # In seconds
        self.__headers = {
            "Validator-Version": __version__,
            "Validator-Hash": commit_short_hash,
        }

    def create_session(self):
        trace_config = aiohttp.TraceConfig()
        trace_config.on_request_start.append(self.on_request_start)
        trace_config.on_request_end.append(self.on_request_end)
        trace_config.on_request_exception.append(self.on_request_exception)

        return aiohttp.ClientSession(
            base_url=self.__base_url,
            timeout=self.__timeout,
            headers=self.__headers,
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
            self.__logger.error("Http request failed", extra=extra)
        else:
            self.__logger.debug("Http request finished", extra=extra)

    async def on_request_exception(
        self, _, trace_config_ctx, params: aiohttp.TraceRequestExceptionParams
    ):
        elapsed_time_ms = round((time.time() - trace_config_ctx.start_time) * 1000)

        extra = {
            "method": params.method,
            "url": str(params.url),
            "elapsed_time_ms": elapsed_time_ms,
        }

        self.__logger.exception("Http request exception", extra=extra)

    async def get_events(self, from_date: int, offset: int, limit: int):
        # Check that all parameters are provided
        if from_date is None or offset is None or limit is None:
            raise ValueError("Invalid parameters")

        async with self.create_session() as session:
            path = f"/api/v2/events?from_date={from_date}&offset={offset}&limit={limit}"

            async with session.get(path) as response:
                response.raise_for_status()

                return await response.json()

    async def get_event(self, event_id: str):
        if not isinstance(event_id, str):
            raise ValueError("Invalid parameter")

        async with self.create_session() as session:
            path = f"/api/v2/events/{event_id}"

            async with session.get(path) as response:
                response.raise_for_status()

                return await response.json()

    async def post_predictions(self, predictions: list[dict]):
        if not isinstance(predictions, list):
            raise ValueError("Invalid parameter")

        assert len(predictions) > 0

        async with self.create_session() as session:
            path = "/api/v2/validators/data"

            async with session.post(path, json=predictions) as response:
                response.raise_for_status()

                return await response.json()
