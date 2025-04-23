import typing
from datetime import datetime

from pydantic import BaseModel


class SN13Data(BaseModel):
    uri: str
    datetime: datetime
    source: str
    label: str | None
    content: str


class SN13Response(BaseModel):
    status: str
    data: typing.List[SN13Data]
    meta: dict
