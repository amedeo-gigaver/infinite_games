 """
Pydantic models for API responses.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Generic, TypeVar

from pydantic import BaseModel, Field, validator
from pydantic.generics import GenericModel


T = TypeVar("T")


class Status(str, Enum):
    """Status of an API response."""
    
    SUCCESS = "success"
    ERROR = "error"


class ErrorResponse(BaseModel):
    """Model for an error response."""
    
    status: Status = Status.ERROR
    message: str
    code: Optional[str] = None
    details: Optional[Any] = None


class SuccessResponse(GenericModel, Generic[T]):
    """Generic model for a successful response."""
    
    status: Status = Status.SUCCESS
    data: T


class PaginatedResponse(GenericModel, Generic[T]):
    """Generic model for a paginated response."""
    
    status: Status = Status.SUCCESS
    data: List[T]
    meta: Dict = Field(
        ...,
        description="Metadata for the paginated response",
        example={
            "page": 1,
            "page_size": 10,
            "total": 100,
            "total_pages": 10,
        },
    )


class ResolverEvidenceSource(BaseModel):
    """Model for an evidence source used by the resolver."""
    
    name: str = Field(..., description="Name of the evidence source")
    url: Optional[str] = Field(default=None, description="URL of the evidence source")
    content: str = Field(..., description="Content of the evidence")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score of the evidence")


class ResolverResult(BaseModel):
    """Model for a resolver result."""
    
    event_id: str = Field(..., description="ID of the event being resolved")
    outcome: float = Field(..., ge=0.0, le=1.0, description="Resolved outcome (0 or 1)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the resolution")
    evidence: List[ResolverEvidenceSource] = Field(..., description="Evidence used for resolution")
    reasoning: str = Field(..., description="Reasoning behind the resolution")
    cost: float = Field(..., ge=0.0, description="Cost of the resolution in USD")
    resolution_time: float = Field(..., ge=0.0, description="Time taken to resolve the event in seconds")
    
    @validator("outcome")
    def validate_outcome(cls, v):
        """Validate that outcome is 0 or 1."""
        if v not in (0, 1):
            raise ValueError("outcome must be 0 or 1")
        return v


class GeneratorResult(BaseModel):
    """Model for a generator result."""
    
    event_type: str = Field(..., description="Type of event generated")
    description: str = Field(..., description="Description of the event")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the event")
    sources: List[Dict] = Field(..., description="Sources used to generate the event")
    metadata: Dict = Field(..., description="Additional metadata for the event")