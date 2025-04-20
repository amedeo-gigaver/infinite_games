"""
Pydantic models for predictions in the API.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator, root_validator


class PredictionSource(str, Enum):
    """Sources of predictions."""
    
    MINER = "miner"
    API = "api"
    UI = "ui"
    MANUAL = "manual"


class PredictionCreate(BaseModel):
    """Model for creating a new prediction."""
    
    event_id: str = Field(..., description="ID of the event being predicted")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of the event occurring")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, 
                                      description="Confidence in the prediction")
    metadata: Optional[Dict] = Field(default=None, description="Additional metadata for the prediction")
    source: PredictionSource = Field(default=PredictionSource.API, 
                                   description="Source of the prediction")
    
    @validator("probability")
    def validate_probability(cls, v):
        """Validate that probability is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("probability must be between 0 and 1")
        return v
    
    @validator("confidence")
    def validate_confidence(cls, v):
        """Validate that confidence is between 0 and 1."""
        if v is not None and not 0 <= v <= 1:
            raise ValueError("confidence must be between 0 and 1")
        return v


class Prediction(PredictionCreate):
    """Model for a prediction."""
    
    id: str = Field(..., description="Unique identifier for the prediction")
    created_at: datetime = Field(..., description="Time when the prediction was created")
    updated_at: datetime = Field(..., description="Time when the prediction was last updated")
    miner_id: Optional[str] = Field(default=None, description="ID of the miner that made the prediction")
    user_id: Optional[str] = Field(default=None, description="ID of the user that made the prediction")
    
    class Config:
        """Pydantic config."""
        
        orm_mode = True


class PredictionUpdate(BaseModel):
    """Model for updating a prediction."""
    
    probability: Optional[float] = Field(default=None, ge=0.0, le=1.0, 
                                       description="Probability of the event occurring")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, 
                                      description="Confidence in the prediction")
    metadata: Optional[Dict] = Field(default=None, description="Additional metadata for the prediction")
    
    @validator("probability")
    def validate_probability(cls, v):
        """Validate that probability is between 0 and 1."""
        if v is not None and not 0 <= v <= 1:
            raise ValueError("probability must be between 0 and 1")
        return v
    
    @validator("confidence")
    def validate_confidence(cls, v):
        """Validate that confidence is between 0 and 1."""
        if v is not None and not 0 <= v <= 1:
            raise ValueError("confidence must be between 0 and 1")
        return v


class PredictionResponse(Prediction):
    """Model for a prediction response."""
    
    event: Optional[Dict] = Field(default=None, description="Event details")


class PredictionListResponse(BaseModel):
    """Model for a list of predictions."""
    
    items: List[Prediction]
    total: int
    page: int
    page_size: int
    
    class Config:
        """Pydantic config."""
        
        orm_mode = True


class PredictionPerformance(BaseModel):
    """Model for prediction performance metrics."""
    
    id: str = Field(..., description="ID of the entity (miner or user)")
    total_predictions: int = Field(..., description="Total number of predictions made")
    correct_predictions: int = Field(..., description="Number of correct predictions")
    accuracy: float = Field(..., description="Accuracy of predictions")
    average_confidence: float = Field(..., description="Average confidence of predictions")
    early_prediction_score: float = Field(..., description="Score for early predictions")
    
    class Config:
        """Pydantic config."""
        
        orm_mode = True


class MinerPerformance(PredictionPerformance):
    """Model for miner performance metrics."""
    
    miner_id: str = Field(..., description="ID of the miner")
    rank: int = Field(..., description="Rank of the miner")
    stake: float = Field(..., description="Stake of the miner")
    events_responded: int = Field(..., description="Number of events the miner responded to")
    
    class Config:
        """Pydantic config."""
        
        orm_mode = True 