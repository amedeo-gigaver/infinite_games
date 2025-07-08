import enum
from datetime import datetime, timezone

from pydantic import Field, field_validator

from neurons.protocol import EventPrediction


class MinerEventStatus(enum.Enum):
    UNRESOLVED = "UNRESOLVED"
    PENDING = "PENDING"
    RESOLVED = "RESOLVED"

    def serialize(self):
        if self.name == "PENDING":
            return "UNRESOLVED"
        return self.name


class MinerEvent(EventPrediction):
    # Override cutoff type as datetime
    cutoff: datetime = Field(..., description="Last datetime the event can be predicted")

    # Miner-specific fields
    status: MinerEventStatus = Field(
        MinerEventStatus.UNRESOLVED, description="Status of the event prediction"
    )

    @field_validator("market_type")
    @classmethod
    def ensure_capital_market_type(cls, v: str) -> str:
        return v.upper()

    @field_validator("cutoff")
    @classmethod
    def ensure_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    async def to_dict(self) -> dict:
        out = self.model_dump()
        out["cutoff"] = out["cutoff"].isoformat()
        out["status"] = out["status"].serialize()
        return out

    def set_probability(self, probability: float | int):
        self.probability = max(0, min(1, probability))

    def get_probability(self) -> float | None:
        return self.probability

    def get_event_id(self) -> str:
        return self.event_id

    def get_status(self) -> MinerEventStatus:
        return self.status

    def set_status(self, status: MinerEventStatus):
        self.status = status

    def get_description(self) -> str:
        return self.description

    def set_reasoning(self, reasoning: str | None):
        self.reasoning = reasoning

    def get_reasoning(self) -> str | None:
        return self.reasoning
