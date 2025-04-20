"""
FastAPI router for Infinite Games integration endpoints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from infinite_forecast.api.auth.dependencies import get_api_key
from infinite_forecast.api.core.logging import get_logger
from infinite_forecast.miner.forecasters.base import MinerEvent
from infinite_forecast.services.ifgames_service import get_ifgames_service

logger = get_logger(__name__)
router = APIRouter()


class EventResponse(BaseModel):
    """Response model for event data."""
    event_id: str = Field(..., description="Unique event identifier")
    title: str = Field(..., description="Event title")
    description: str = Field(..., description="Event description")
    market_type: str = Field(..., description="Market type (crypto, fred, llm, etc.)")
    status: str = Field(..., description="Event status")
    created_at: datetime = Field(..., description="Creation timestamp")
    cutoff: datetime = Field(..., description="Cutoff timestamp")
    resolves_at: Optional[datetime] = Field(None, description="Resolution timestamp")
    metadata: Dict[str, Any] = Field({}, description="Additional event metadata")


class PredictionRequest(BaseModel):
    """Request model for submitting a prediction."""
    probability: float = Field(..., ge=0.0, le=1.0, description="Prediction probability (0-1)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence (0-1)")
    explanation: Optional[str] = Field(None, description="Explanation for the prediction")
    metadata: Dict[str, Any] = Field({}, description="Additional prediction metadata")


class PredictionResponse(BaseModel):
    """Response model for prediction data."""
    event_id: str = Field(..., description="Event ID")
    probability: float = Field(..., description="Prediction probability")
    confidence: float = Field(..., description="Prediction confidence")
    status: str = Field(..., description="Prediction status")
    timestamp: datetime = Field(..., description="Prediction timestamp")


@router.get("/events", response_model=List[EventResponse], tags=["ifgames"])
async def get_events(
    status: Optional[str] = Query(None, description="Filter by status"),
    market_type: Optional[str] = Query(None, description="Filter by market type"),
    _: str = Depends(get_api_key),
):
    """
    Get all events from Infinite Games platform.
    
    Args:
        status: Optional status filter
        market_type: Optional market type filter
        
    Returns:
        List of events
    """
    try:
        service = get_ifgames_service()
        events = await service.get_eligible_events()
        
        # Apply filters if specified
        if status:
            events = [e for e in events if e.status == status]
        if market_type:
            events = [e for e in events if e.market_type == market_type]
        
        # Convert MinerEvent objects to EventResponse models
        return [
            EventResponse(
                event_id=event.event_id,
                title=event.title,
                description=event.description,
                market_type=event.market_type,
                status=event.status,
                created_at=event.created_at,
                cutoff=event.cutoff,
                resolves_at=event.resolves_at,
                metadata=event.metadata,
            )
            for event in events
        ]
    except Exception as e:
        logger.error(f"Error fetching events: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch events: {str(e)}",
        )


@router.get("/events/{event_id}", response_model=EventResponse, tags=["ifgames"])
async def get_event(
    event_id: str,
    _: str = Depends(get_api_key),
):
    """
    Get details for a specific event.
    
    Args:
        event_id: Event ID
        
    Returns:
        Event details
    """
    try:
        service = get_ifgames_service()
        client = service.client
        
        # Get event details from the API
        event_data = await client.get_event(event_id)
        
        # Convert to MinerEvent
        event = client.event_response_to_miner_event(event_data)
        
        # Convert to response model
        return EventResponse(
            event_id=event.event_id,
            title=event.title,
            description=event.description,
            market_type=event.market_type,
            status=event.status,
            created_at=event.created_at,
            cutoff=event.cutoff,
            resolves_at=event.resolves_at,
            metadata=event.metadata,
        )
    except Exception as e:
        logger.error(f"Error fetching event {event_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event not found or error: {str(e)}",
        )


@router.post("/events/{event_id}/predict", response_model=PredictionResponse, tags=["ifgames"])
async def submit_prediction(
    event_id: str,
    prediction: PredictionRequest,
    _: str = Depends(get_api_key),
):
    """
    Submit a prediction for an event.
    
    Args:
        event_id: Event ID
        prediction: Prediction data
        
    Returns:
        Prediction submission status
    """
    try:
        service = get_ifgames_service()
        
        # TODO: Implement the actual submission logic
        # This is a placeholder since the submission endpoint wasn't specified
        # in the provided API documentation
        
        # For now, we'll just return a mock response
        return PredictionResponse(
            event_id=event_id,
            probability=prediction.probability,
            confidence=prediction.confidence,
            status="submitted",
            timestamp=datetime.utcnow(),
        )
    except Exception as e:
        logger.error(f"Error submitting prediction for event {event_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit prediction: {str(e)}",
        )


@router.post("/process", tags=["ifgames"])
async def process_events(
    _: str = Depends(get_api_key),
):
    """
    Manually trigger processing of all eligible events.
    
    Returns:
        Process status
    """
    try:
        service = get_ifgames_service()
        
        # Process events asynchronously
        await service.process_events()
        
        return {"status": "success", "message": "Events processing triggered"}
    except Exception as e:
        logger.error(f"Error processing events: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process events: {str(e)}",
        ) 