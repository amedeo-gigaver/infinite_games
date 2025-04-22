"""
Client for interacting with the Infinite Games API.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
from pydantic import BaseModel

from infinite_forecast.api.core.config import settings
from infinite_forecast.api.core.logging import get_logger
from infinite_forecast.miner.forecasters.base import MinerEvent

logger = get_logger(__name__)


class IFGamesClient:
    """Client for interacting with the Infinite Games API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the client with API key.
        
        Args:
            api_key: API key for authentication (defaults to config)
        """
        self.api_key = api_key or settings.IFGAMES_API_KEY
        self.base_url = settings.IFGAMES_API_URL
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    async def _request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make an HTTP request to the API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request data (for POST requests)
            
        Returns:
            Response data
            
        Raises:
            Exception: On request failure
        """
        url = f"{self.base_url}{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.request(
                    method=method,
                    url=url,
                    headers=self.headers,
                    json=data,
                    timeout=30  # 30 second timeout
                ) as response:
                    response_text = await response.text()
                    
                    if response.status >= 400:
                        logger.error(
                            f"API request failed: {response.status} - {response_text}"
                        )
                        raise Exception(
                            f"API request failed: {response.status} - {response_text}"
                        )
                    
                    return json.loads(response_text)
            except aiohttp.ClientError as e:
                logger.error(f"HTTP error: {str(e)}")
                raise Exception(f"HTTP error: {str(e)}")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)} - Response: {response_text}")
                raise Exception(f"JSON decode error: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                raise

    async def get_events(self) -> List[Dict[str, Any]]:
        """Get all events from the platform.
        
        Returns:
            List of events
        """
        return await self._request("GET", "/api/v2/events")

    async def get_event(self, event_id: str) -> Dict[str, Any]:
        """Get details for a specific event.
        
        Args:
            event_id: Event ID
            
        Returns:
            Event details
        """
        return await self._request("GET", f"/api/v2/validator/events/{event_id}")

    async def get_community_prediction(self, event_id: str) -> Dict[str, Any]:
        """Get community prediction for an event.
        
        Args:
            event_id: Event ID
            
        Returns:
            Community prediction data
        """
        return await self._request(
            "GET", 
            f"/api/v2/validator/events/{event_id}/community_prediction"
        )

    async def get_predictions(self, event_id: str) -> Dict[str, Any]:
        """Get all predictions for an event.
        
        Args:
            event_id: Event ID
            
        Returns:
            Predictions data
        """
        return await self._request(
            "GET", 
            f"/api/v2/validator/events/{event_id}/predictions"
        )

    async def submit_prediction(
        self, 
        event_id: str, 
        probability: float,
        confidence: float,
        explanation: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Submit a prediction for an event.
        
        Args:
            event_id: Event ID
            probability: Prediction probability (0-1)
            confidence: Prediction confidence (0-1)
            explanation: Optional explanation for the prediction
            metadata: Additional prediction metadata
            
        Returns:
            Submission response
        """
        data = {
            "probability": probability,
            "confidence": confidence
        }
        
        if explanation:
            data["explanation"] = explanation
            
        if metadata:
            data["metadata"] = metadata
        
        # Note: Endpoint needs to be confirmed from API documentation
        return await self._request(
            "POST", 
            f"/api/v2/validator/events/{event_id}/predictions", 
            data=data
        )

    def event_response_to_miner_event(self, event_data: Dict[str, Any]) -> MinerEvent:
        """Convert API event response to MinerEvent.
        
        Args:
            event_data: Event data from API
            
        Returns:
            MinerEvent object
        """
        # Extract and parse timestamps
        created_at = datetime.fromisoformat(event_data.get("created_at", ""))
        cutoff = datetime.fromisoformat(event_data.get("cutoff", ""))
        
        resolves_at = None
        if event_data.get("resolves_at"):
            resolves_at = datetime.fromisoformat(event_data.get("resolves_at", ""))
        
        # Extract event metadata and properties
        metadata = event_data.get("metadata", {})
        market_type = event_data.get("market_type", "unknown")
        
        return MinerEvent(
            event_id=event_data.get("id"),
            title=event_data.get("title", ""),
            description=event_data.get("description", ""),
            market_type=market_type,
            status=event_data.get("status", ""),
            created_at=created_at,
            cutoff=cutoff,
            resolves_at=resolves_at,
            metadata=metadata
        ) 