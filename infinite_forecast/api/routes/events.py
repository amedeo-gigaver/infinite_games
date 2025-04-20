"""
API endpoints for events.
"""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Path, status

from infinite_forecast.api.core.security import get_api_key_dependency
from infinite_forecast.api.models.events import (
    Event, EventCreate, EventListResponse, EventUpdate, EventResponse
)


router = APIRouter()

# In-memory storage for demo - replace with database in production
events_db = {}


@router.post(
    "/events", 
    response_model=EventResponse, 
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(get_api_key_dependency())]
)
async def create_event(event: EventCreate):
    """
    Create a new event.
    
    Args:
        event: Event data
        
    Returns:
        Created event
    """
    event_id = f"{event.event_type.value}-{uuid4()}"
    now = datetime.utcnow()
    
    # Create event object
    db_event = Event(
        id=str(uuid4()),
        event_id=event_id,
        created_at=now,
        updated_at=now,
        status="active",
        **event.dict()
    )
    
    # Store event
    events_db[event_id] = db_event
    
    return EventResponse(**db_event.dict(), predictions=[])


@router.get(
    "/events", 
    response_model=EventListResponse,
    dependencies=[Depends(get_api_key_dependency())]
)
async def get_events(
    offset: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    event_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
):
    """
    Get events with pagination and filtering.
    
    Args:
        offset: Pagination offset
        limit: Pagination limit
        event_type: Filter by event type
        status: Filter by status
        
    Returns:
        List of events
    """
    # Apply filters
    filtered_events = list(events_db.values())
    
    if event_type:
        filtered_events = [e for e in filtered_events if e.event_type == event_type]
        
    if status:
        filtered_events = [e for e in filtered_events if e.status == status]
    
    # Sort by created_at in descending order
    filtered_events.sort(key=lambda x: x.created_at, reverse=True)
    
    # Paginate
    total = len(filtered_events)
    paginated_events = filtered_events[offset : offset + limit]
    
    return EventListResponse(
        items=paginated_events,
        total=total,
        page=(offset // limit) + 1,
        page_size=limit,
    )


@router.get(
    "/events/{event_id}", 
    response_model=EventResponse,
    dependencies=[Depends(get_api_key_dependency())]
)
async def get_event(event_id: str = Path(...)):
    """
    Get an event by ID.
    
    Args:
        event_id: Event ID
        
    Returns:
        Event details
        
    Raises:
        HTTPException: If event not found
    """
    if event_id not in events_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event with ID {event_id} not found",
        )
    
    event = events_db[event_id]
    
    # In a real implementation, fetch predictions for this event
    predictions = []
    
    return EventResponse(**event.dict(), predictions=predictions)


@router.patch(
    "/events/{event_id}", 
    response_model=EventResponse,
    dependencies=[Depends(get_api_key_dependency())]
)
async def update_event(
    event_update: EventUpdate,
    event_id: str = Path(...),
):
    """
    Update an event.
    
    Args:
        event_id: Event ID
        event_update: Updated event data
        
    Returns:
        Updated event
        
    Raises:
        HTTPException: If event not found
    """
    if event_id not in events_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event with ID {event_id} not found",
        )
    
    event = events_db[event_id]
    
    # Update fields
    for field, value in event_update.dict(exclude_unset=True).items():
        setattr(event, field, value)
    
    # Update updated_at timestamp
    event.updated_at = datetime.utcnow()
    
    # Save updated event
    events_db[event_id] = event
    
    return EventResponse(**event.dict(), predictions=[]) 