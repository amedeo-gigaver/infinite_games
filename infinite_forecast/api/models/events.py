"""
Pydantic models for events in the API.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class EventType(str, Enum):
    """Types of events that can be predicted."""
    
    LLM = "llm"
    FRED = "fred"
    CRYPTO = "crypto"
    EARNINGS = "earnings"
    POLYMARKET = "polymarket"


class EventStatus(str, Enum):
    """Status of an event."""
    
    PENDING = "pending"
    ACTIVE = "active"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


class EventCreate(BaseModel):
    """Model for creating a new event."""
    
    event_type: EventType
    market_type: str = Field(..., description="Type of market for this event")
    description: str = Field(..., description="Human-readable description of the event")
    cutoff: datetime = Field(..., description="Time after which predictions cannot be modified")
    starts: datetime = Field(..., description="Time when the event starts")
    resolve_date: datetime = Field(..., description="Expected date of resolution")
    end_date: datetime = Field(..., description="End date for the event")
    metadata: Optional[Dict] = Field(default=None, description="Additional metadata for the event")
    
    @validator("resolve_date")
    def resolve_date_after_cutoff(cls, v, values):
        """Validate that resolve_date is after cutoff."""
        if "cutoff" in values and v <= values["cutoff"]:
            raise ValueError("resolve_date must be after cutoff")
        return v
    
    @validator("end_date")
    def end_date_after_resolve_date(cls, v, values):
        """Validate that end_date is after resolve_date."""
        if "resolve_date" in values and v <= values["resolve_date"]:
            raise ValueError("end_date must be after resolve_date")
        return v


class Event(EventCreate):
    """Model for an event."""
    
    id: str = Field(..., description="Unique identifier for the event")
    event_id: str = Field(..., description="Unique ID in the format 'market_type-event_id'")
    created_at: datetime = Field(..., description="Time when the event was created")
    updated_at: datetime = Field(..., description="Time when the event was last updated")
    status: EventStatus = Field(..., description="Current status of the event")
    outcome: Optional[float] = Field(default=None, description="Actual outcome (0 or 1)")
    
    class Config:
        """Pydantic config."""
        
        orm_mode = True


class EventResponse(Event):
    """Model for an event response."""
    
    predictions: Optional[List[Dict]] = Field(default=None, description="Predictions for this event")
    

class EventListResponse(BaseModel):
    """Model for a list of events."""
    
    items: List[Event]
    total: int
    page: int
    page_size: int
    
    class Config:
        """Pydantic config."""
        
        orm_mode = True


class EventUpdate(BaseModel):
    """Model for updating an event."""
    
    description: Optional[str] = None
    metadata: Optional[Dict] = None
    status: Optional[EventStatus] = None
    outcome: Optional[float] = None
    
    @validator("outcome")
    def validate_outcome(cls, v):
        """Validate that outcome is 0 or 1."""
        if v is not None and v not in (0, 1):
            raise ValueError("outcome must be 0 or 1")
        return v 