import random
from abc import ABC, abstractmethod

from neurons.miner.models.event import MinerEvent, MinerEventStatus
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class BaseForecaster(ABC):
    def __init__(
        self,
        event: MinerEvent,
        logger: InfiniteGamesLogger,
        extremize: bool = False,
    ):
        self.event = event
        self.extremize = extremize
        self.logger = logger

    @abstractmethod
    async def _run(self) -> float | int:
        raise NotImplementedError("Subclass must implement _run method")

    async def run(self) -> None:
        try:
            event_id = self.event.get_event_id()
            probability = await self._run()
            if self.extremize:
                probability = round(probability)
            self.event.set_probability(probability)
            self.event.set_status(MinerEventStatus.RESOLVED)
            self.logger.info(f"Event {event_id} forecasted with probability {probability}")
        except Exception as e:
            self.logger.error(f"Error forecasting event {event_id}: {e}")
            self.event.set_status(MinerEventStatus.UNRESOLVED)
            self.logger.error(f"Failed to forecast event {event_id}")
        return

    def __lt__(self, other: "BaseForecaster") -> bool:
        return self.event.cutoff < other.event.cutoff


class DummyForecaster(BaseForecaster):
    async def _run(self) -> float | int:
        return random.randint(0, 1)
