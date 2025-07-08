from bittensor import Synapse
from pydantic import BaseModel, Field


class EventPrediction(BaseModel):
    event_id: str = Field(..., description="ID of the event")
    market_type: str = Field(..., description="Market the event belongs to")
    description: str = Field(..., description="Description of the event")
    cutoff: int | None = Field(..., description="Last timestamp the event can be predicted")
    metadata: dict = Field(..., description="Metadata of the event")
    # Fields for miners to respond
    probability: float | None = Field(
        ..., ge=0, le=1, description="Miner's prediction for the event"
    )
    reasoning: str | None = Field(..., description="Reasoning for the miner's event prediction")
    miner_answered: bool = Field(
        ..., description="Flag indicating if the miner has answered the event"
    )


EventKey = str  # f'{market_type}-{event_id}'


class EventPredictionSynapse(Synapse):
    events: dict[EventKey, EventPrediction]
