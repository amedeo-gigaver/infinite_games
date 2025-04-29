from typing import Literal

from pydantic import BaseModel

from neurons.validator.models.event import EventsModel
from neurons.validator.models.prediction import PredictionsModel


class HealthCheckResponse(BaseModel):
    status: Literal["OK"]


class GetEventResponse(EventsModel):
    pass


class GetEventCommunityPrediction(BaseModel):
    event_id: str
    community_prediction: None | float


class GetEventsCommunityPredictions(BaseModel):
    count: int
    community_predictions: list[GetEventCommunityPrediction]


class GetEventPredictions(BaseModel):
    count: int
    predictions: list[PredictionsModel]
