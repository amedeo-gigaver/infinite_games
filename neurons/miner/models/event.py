import enum
import typing
from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator


class MinerEventStatus(enum.Enum):
    UNRESOLVED = "UNRESOLVED"
    PENDING = "PENDING"
    RESOLVED = "RESOLVED"

    def serialize(self):
        if self.name == "PENDING":
            return "UNRESOLVED"
        return self.name


class MinerEvent(BaseModel):
    event_id: str = Field(..., description="The ID of the event")
    market_type: str = Field(..., description="The market the event belongs to")
    probability: typing.Optional[float] = Field(
        None, ge=0, le=1, description="The prediction of the event"
    )
    description: str = Field(..., description="The title and the description of the event")
    cutoff: datetime = Field(..., description="The last date the event can be predicted")
    status: MinerEventStatus = Field(
        MinerEventStatus.UNRESOLVED, description="The status of the event"
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
