# THIS SHOULD BE KEPT IN SYNC WITH BACKEND MODELS
# https://github.com/infinite-mech/infinite_games_events/blob/develop/event_api/api_models/validator.py

import datetime
import typing

from pydantic import BaseModel, ConfigDict, Field


class MinerScore(BaseModel):
    event_id: str
    prediction: float
    answer: float = Field(..., json_schema_extra={"ge": 0, "le": 1})
    miner_hotkey: str
    miner_uid: int
    miner_score: float
    miner_effective_score: float
    validator_hotkey: str
    validator_uid: int
    spec_version: typing.Optional[str] = "0.0.0"
    registered_date: typing.Optional[datetime.datetime]
    scored_at: typing.Optional[datetime.datetime]

    model_config = ConfigDict(from_attributes=True, extra="forbid")


class PostScores(BaseModel):
    results: typing.List[MinerScore] = Field(..., min_length=1)
