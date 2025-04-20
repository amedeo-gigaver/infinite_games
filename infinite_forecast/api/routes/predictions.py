"""
API endpoints for predictions.
"""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Path, status

from infinite_forecast.api.core.security import get_api_key_dependency
from infinite_forecast.api.models.predictions import (
    Prediction, PredictionCreate, PredictionListResponse, 
    PredictionUpdate, PredictionResponse, MinerPerformance
)
from infinite_forecast.api.routes.events import events_db

router = APIRouter()

# In-memory storage for demo - replace with database in production
predictions_db = {}


@router.post(
    "/predictions", 
    response_model=PredictionResponse, 
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(get_api_key_dependency())]
)
async def create_prediction(prediction: PredictionCreate):
    """
    Create a new prediction.
    
    Args:
        prediction: Prediction data
        
    Returns:
        Created prediction
        
    Raises:
        HTTPException: If event not found
    """
    # Check if event exists
    if prediction.event_id not in events_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event with ID {prediction.event_id} not found",
        )
    
    # Check if event is active
    event = events_db[prediction.event_id]
    if event.status != "active":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot predict on {event.status} event",
        )
    
    # Check if event cutoff has passed
    now = datetime.utcnow()
    if event.cutoff < now:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Event cutoff has passed",
        )
    
    # Create prediction ID
    prediction_id = str(uuid4())
    
    # Create prediction object
    db_prediction = Prediction(
        id=prediction_id,
        created_at=now,
        updated_at=now,
        **prediction.dict()
    )
    
    # Store prediction
    predictions_db[prediction_id] = db_prediction
    
    return PredictionResponse(**db_prediction.dict(), event=event.dict())


@router.get(
    "/predictions", 
    response_model=PredictionListResponse,
    dependencies=[Depends(get_api_key_dependency())]
)
async def get_predictions(
    offset: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    event_id: Optional[str] = Query(None),
    source: Optional[str] = Query(None),
    miner_id: Optional[str] = Query(None),
):
    """
    Get predictions with pagination and filtering.
    
    Args:
        offset: Pagination offset
        limit: Pagination limit
        event_id: Filter by event ID
        source: Filter by source
        miner_id: Filter by miner ID
        
    Returns:
        List of predictions
    """
    # Apply filters
    filtered_predictions = list(predictions_db.values())
    
    if event_id:
        filtered_predictions = [p for p in filtered_predictions if p.event_id == event_id]
        
    if source:
        filtered_predictions = [p for p in filtered_predictions if p.source == source]
        
    if miner_id:
        filtered_predictions = [p for p in filtered_predictions if p.miner_id == miner_id]
    
    # Sort by created_at in descending order
    filtered_predictions.sort(key=lambda x: x.created_at, reverse=True)
    
    # Paginate
    total = len(filtered_predictions)
    paginated_predictions = filtered_predictions[offset : offset + limit]
    
    return PredictionListResponse(
        items=paginated_predictions,
        total=total,
        page=(offset // limit) + 1,
        page_size=limit,
    )


@router.get(
    "/predictions/{prediction_id}", 
    response_model=PredictionResponse,
    dependencies=[Depends(get_api_key_dependency())]
)
async def get_prediction(prediction_id: str = Path(...)):
    """
    Get a prediction by ID.
    
    Args:
        prediction_id: Prediction ID
        
    Returns:
        Prediction details
        
    Raises:
        HTTPException: If prediction not found
    """
    if prediction_id not in predictions_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction with ID {prediction_id} not found",
        )
    
    prediction = predictions_db[prediction_id]
    
    # Get associated event
    event = events_db.get(prediction.event_id)
    
    return PredictionResponse(**prediction.dict(), event=event.dict() if event else None)


@router.patch(
    "/predictions/{prediction_id}", 
    response_model=PredictionResponse,
    dependencies=[Depends(get_api_key_dependency())]
)
async def update_prediction(
    prediction_update: PredictionUpdate,
    prediction_id: str = Path(...),
):
    """
    Update a prediction.
    
    Args:
        prediction_id: Prediction ID
        prediction_update: Updated prediction data
        
    Returns:
        Updated prediction
        
    Raises:
        HTTPException: If prediction not found or event cutoff has passed
    """
    if prediction_id not in predictions_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction with ID {prediction_id} not found",
        )
    
    prediction = predictions_db[prediction_id]
    
    # Check if event cutoff has passed
    event = events_db.get(prediction.event_id)
    if event and event.cutoff < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Event cutoff has passed",
        )
    
    # Update fields
    for field, value in prediction_update.dict(exclude_unset=True).items():
        setattr(prediction, field, value)
    
    # Update updated_at timestamp
    prediction.updated_at = datetime.utcnow()
    
    # Save updated prediction
    predictions_db[prediction_id] = prediction
    
    return PredictionResponse(**prediction.dict(), event=event.dict() if event else None)


@router.get(
    "/miners/performance", 
    response_model=List[MinerPerformance],
    dependencies=[Depends(get_api_key_dependency())]
)
async def get_miner_performance():
    """
    Get performance metrics for all miners.
    
    Returns:
        List of miner performance metrics
    """
    # In a real implementation, calculate these metrics from the database
    # For demo, return mock data
    return [
        MinerPerformance(
            id="miner1",
            miner_id="miner1",
            total_predictions=100,
            correct_predictions=75,
            accuracy=0.75,
            average_confidence=0.8,
            early_prediction_score=0.9,
            rank=1,
            stake=1000,
            events_responded=90,
        ),
        MinerPerformance(
            id="miner2",
            miner_id="miner2",
            total_predictions=95,
            correct_predictions=65,
            accuracy=0.68,
            average_confidence=0.75,
            early_prediction_score=0.8,
            rank=2,
            stake=800,
            events_responded=85,
        ),
    ] 