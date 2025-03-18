from fastapi import APIRouter, HTTPException

from neurons.validator.api.types import ApiRequest
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.api import (
    GetEventCommunityPrediction,
    GetEventPredictions,
    GetEventResponse,
)
from neurons.validator.utils.common.interval import get_interval_start_minutes

router = APIRouter()


@router.get(
    "/{event_id}",
)
async def get_event(event_id: str, request: ApiRequest) -> GetEventResponse:
    db_operations: DatabaseOperations = request.state.db_operations

    unique_event_id = f"ifgames-{event_id}"

    event = await db_operations.get_event(unique_event_id)

    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    return event


@router.get(
    "/{event_id}/community_prediction",
)
async def get_community_prediction(
    event_id: str, request: ApiRequest
) -> GetEventCommunityPrediction:
    db_operations: DatabaseOperations = request.state.db_operations

    unique_event_id = f"ifgames-{event_id}"

    event = await db_operations.get_event(unique_event_id=unique_event_id)

    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    current_interval = get_interval_start_minutes()

    community_prediction = await db_operations.get_wa_prediction_event(
        unique_event_id=unique_event_id, interval_start_minutes=current_interval
    )

    return {"event_id": event_id, "community_prediction": community_prediction}


@router.get(
    "/{event_id}/predictions",
)
async def get_predictions(event_id: str, request: ApiRequest) -> GetEventPredictions:
    db_operations: DatabaseOperations = request.state.db_operations

    unique_event_id = f"ifgames-{event_id}"
    current_interval = get_interval_start_minutes()

    predictions = await db_operations.get_predictions_for_event(
        unique_event_id=unique_event_id, interval_start_minutes=current_interval
    )

    return {"count": len(predictions), "predictions": predictions}
