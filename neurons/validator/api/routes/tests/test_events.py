from unittest.mock import ANY

import pytest
from httpx import AsyncClient

from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.event import EventsModel, EventStatus


class TestEventsRoutes:
    @pytest.fixture
    async def mock_event(self, test_db_operations: DatabaseOperations):
        event_id = "event_id"
        unique_event_id = f"ifgames-{event_id}"

        events = [
            EventsModel(
                unique_event_id=unique_event_id,
                event_id=event_id,
                market_type="market_type",
                event_type="event_type",
                description="desc",
                status=EventStatus.SETTLED,
                metadata="{}",
                created_at="2012-12-02T14:30:00Z",
            ),
        ]

        await test_db_operations.upsert_pydantic_events(events=events)

        return {"events": events}

    async def test_get_event(
        self, test_api_client: AsyncClient, mock_event: dict[str, list[EventsModel]]
    ):
        event = mock_event["events"][0]

        response = await test_api_client.get(f"/api/events/{event.event_id}")

        assert response.status_code == 200
        assert response.json() == {
            "event_id": "event_id",
            "unique_event_id": "ifgames-event_id",
            "created_at": "2012-12-02T14:30:00Z",
            "cutoff": None,
            "deleted_at": None,
            "description": "desc",
            "event_type": "event_type",
            "exported": False,
            "local_updated_at": ANY,
            "market_type": "market_type",
            "metadata": "{}",
            "outcome": None,
            "processed": False,
            "registered_date": ANY,
            "resolved_at": None,
            "status": 3,
        }

        # Test not found event
        response = await test_api_client.get("/api/events/not_found_event_id")

        assert response.status_code == 404
        assert response.json() == {"detail": "Event not found"}

    async def test_get_event_auth(
        self,
        test_api_client_no_auth: AsyncClient,
    ):
        response = await test_api_client_no_auth.get("/api/events/event_id")

        assert response.status_code == 401
        assert response.json() == {"detail": "Unauthorized"}
