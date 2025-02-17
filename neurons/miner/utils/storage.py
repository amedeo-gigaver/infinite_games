import asyncio
import json
import typing
from asyncio import Lock

from neurons.miner.models.event import MinerEvent
from neurons.validator.utils.logger.logger import InfiniteGamesLogger

STORAGE_FILE = "miner-events.json"
SAVE_INTERVAL_SECONDS = 30  # 5 minutes


class MinerStorage:
    def __init__(self, logger: InfiniteGamesLogger):
        self.cache: dict[str, MinerEvent] = {}
        self.lock = Lock()
        self.logger = logger

    async def load(self, condition=None):
        """
        Load events from file that meet the specified condition.
        condition: Optional callable that takes an Event and returns bool
        """
        try:
            with open(STORAGE_FILE, "r") as file:
                loaded: dict[str, dict] = json.load(file)

            async with self.lock:
                for key, value in loaded.items():
                    event: MinerEvent = MinerEvent.model_validate(value)
                    if condition is None or condition(event):
                        self.cache[key] = event
        except Exception:
            self.logger.warning("Fail to load cache file", exc_info=True)

    async def get(self, event_id: str) -> typing.Optional[MinerEvent]:
        async with self.lock:
            return self.cache.get(event_id, None)

    async def set(self, event_id: str, event: MinerEvent):
        async with self.lock:
            self.cache[event_id] = event

    async def _store(self):
        async with self.lock:
            serializable_cache = {
                event_id: await event.to_dict() for event_id, event in self.cache.items()
            }
        with open(STORAGE_FILE, "w") as file:
            json.dump(serializable_cache, file, indent=2)
        self.logger.info("Saved storage")

    async def save(self):
        self.logger.info("Saving storage task started")

        while True:
            try:
                await self._store()
            except Exception:
                self.logger.error("Failed to save storage", exc_info=True)
            finally:
                await asyncio.sleep(SAVE_INTERVAL_SECONDS)
