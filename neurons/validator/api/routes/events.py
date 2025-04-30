from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from neurons.validator.api.types import ApiRequest
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.api import (
    GetEventCommunityPrediction,
    GetEventPredictions,
    GetEventResponse,
    GetEventsCommunityPredictions,
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
async def get_event_community_prediction(
    event_id: str, request: ApiRequest
) -> GetEventCommunityPrediction:
    db_operations: DatabaseOperations = request.state.db_operations

    unique_event_id = f"ifgames-{event_id}"

    event = await db_operations.get_event(unique_event_id=unique_event_id)

    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    current_interval = get_interval_start_minutes()

    community_predictions_by_id = await db_operations.get_wa_predictions_events(
        unique_event_ids=[unique_event_id], interval_start_minutes=current_interval
    )

    return {
        "event_id": unique_event_id.removeprefix("ifgames-"),
        "community_prediction": community_predictions_by_id.get(unique_event_id),
    }


@router.get(
    "/community_predictions/",
)
async def get_events_community_predictions(
    event_ids: Annotated[list[str], Query()], request: ApiRequest
) -> GetEventsCommunityPredictions:
    db_operations: DatabaseOperations = request.state.db_operations

    if len(event_ids) > 20:
        raise HTTPException(
            status_code=400,
            detail="Maximum 20 events ids allowed",
        )

    current_interval = get_interval_start_minutes()

    unique_event_ids = [f"ifgames-{event_id}" for event_id in event_ids]

    community_predictions_by_id = await db_operations.get_wa_predictions_events(
        unique_event_ids=unique_event_ids, interval_start_minutes=current_interval
    )

    community_predictions = [
        {
            "event_id": unique_event_id.removeprefix("ifgames-"),
            "community_prediction": community_predictions_by_id[unique_event_id],
        }
        for unique_event_id in community_predictions_by_id.keys()
    ]

    return {"count": len(community_predictions), "community_predictions": community_predictions}


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
