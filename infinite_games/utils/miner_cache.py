import asyncio
import enum
import heapq
import json
from copy import deepcopy
from datetime import datetime
from typing import Optional

import bittensor as bt

from asyncio import Lock

from pydantic import BaseModel, Field, field_validator

FILE_NAME = '.miner-cache.json'
RETRY_TIME = 2


class MinerCacheStatus(enum.Enum):
    NOT_STARTED = 1
    PENDING = 2
    COMPLETED = 3


class MarketType(enum.Enum):
    POLYMARKET = 1
    AZURO = 2


class Event(BaseModel):
    event_id: str
    market_type: MarketType
    probability: Optional[float] = Field(..., min=0, max=1)
    description: str
    starts: Optional[int]
    resolve_date: Optional[int]
    cutoff: Optional[int]

    @field_validator('cutoff', mode='after')
    def calculate_cutoff(cls, v, values):
        if v is None:
            try:
                if values.data['market_type'] == MarketType.POLYMARKET:
                    return values.data["resolve_date"] - 86400 - 3600  # 1 day + 1 hour
                elif values.data['market_type'] == MarketType.AZURO:
                    return values.data["starts"] - 1800  # 30 minutes
                return None
            except KeyError:
                raise ValueError(f"Invalid market type: {v}")
        return v

    @classmethod
    def init_from_market(cls, market: dict):
        return cls(
            event_id=market["event_id"],
            market_type=MarketType[market["market_type"].upper()],
            probability=market["probability"],
            description=market["description"],
            starts=market["starts"],
            resolve_date=market["resolve_date"],
            cutoff=None
        )


class MinerCacheObject(BaseModel):
    event: Event
    status: MinerCacheStatus

    def to_dict(self) -> dict:
        output = self.dict()
        output["event"]["market_type"] = self.event.market_type.name
        output["status"] = self.status.name
        return output

    @classmethod
    def init_from_market(cls, market: dict, status=MinerCacheStatus.NOT_STARTED):
        return cls(
            event=Event.init_from_market(market),
            status=status
        )


class MinerCache:

    def __init__(self):
        self.cache: dict[str, MinerCacheObject] = {}
        self.queue = []
        self._background_task = None
        self._lock = Lock()

    def __del__(self):
        self._background_task.cancel()

    def initialize_cache(self):
        try:
            with open(FILE_NAME, 'r') as file:
                loaded: dict[str, dict] = json.load(file)

            for key, value in loaded.items():
                self.cache[key] = MinerCacheObject.init_from_market(
                    value["event"], status=MinerCacheStatus[value["status"]]
                )
        except Exception as e:
            bt.logging.warning("Fail to load cache file {}".format(e))

    async def add(self, key, coro, market: MinerCacheObject):
        async with self._lock:
            self.cache[key] = market

            heapq.heappush(
                self.queue,
                (market.event.cutoff, key, coro, market)
            )

    async def _clean_cache(self) -> None:
        async with self._lock:
            keys_for_deletion: list[str] = []
            for key, market in self.cache.items():
                if market.status == MinerCacheStatus.COMPLETED:
                    expired = (market.event.starts or market.event.resolve_date) + 2*86400
                    if expired < int(datetime.utcnow().timestamp()):
                        keys_for_deletion.append(key)

            for key in keys_for_deletion:
                del self.cache[key]

    async def _store(self) -> None:
        async with self._lock:
            data_for_storing: dict[str, dict] = {}
            for key, market in self.cache.items():
                if market.status == MinerCacheStatus.COMPLETED:
                    data_for_storing[key] = market.to_dict()

        # Writing JSON data
        with open(FILE_NAME, 'w') as file:
            json.dump(data_for_storing, file, indent=4)

    async def _run_tasks(self) -> None:
        while True:
            try:
                async with self._lock:
                    flag = len(self.queue) > 0

                if flag:
                    async with self._lock:
                        _, key, coro, market = heapq.heappop(self.queue)
                        self.cache[key].status = MinerCacheStatus.PENDING
                        bt.logging.info("Task {} started".format(key))

                    await coro(market)

                    async with self._lock:
                        self.cache[key].status = MinerCacheStatus.COMPLETED
                        bt.logging.info("Pending tasks {}".format(len(self.queue)))

                    await self._store()
                    #await self._clean_cache()

            except Exception as e:
                bt.logging.error("Failed to create task {}".format(e))
            finally:
                await asyncio.sleep(RETRY_TIME)

    async def get(self, key: str) -> Optional[MinerCacheObject]:
        async with self._lock:
            if not self._background_task:
                bt.logging.info("Cache background task started")
                self._background_task = asyncio.create_task(self._run_tasks())

            if key in self.cache:
                return self.cache[key]

        return None
