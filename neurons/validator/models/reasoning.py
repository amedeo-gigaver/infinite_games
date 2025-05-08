from datetime import datetime
from enum import IntEnum
from typing import Any, Optional

from pydantic import BaseModel, field_validator


class ReasoningExportedStatus(IntEnum):
    NOT_EXPORTED = 0
    EXPORTED = 1


class ReasoningModel(BaseModel):
    event_id: str
    miner_uid: int
    miner_hotkey: str
    reasoning: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    exported: Optional[bool] = False

    @property
    def primary_key(self):
        return [
            "event_id",
            "miner_uid",
            "miner_hotkey",
        ]

    @field_validator("exported", mode="before")
    def parse_exported_as_bool(cls, v: Any) -> bool:
        # If the DB returns an integer, convert it to boolean
        if isinstance(v, int):
            return bool(v)

        return v


REASONING_FIELDS = ReasoningModel.model_fields.keys()
