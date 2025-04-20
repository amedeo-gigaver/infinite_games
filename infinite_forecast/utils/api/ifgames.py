"""
Infinite Games API client for interacting with the platform.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import httpx

from infinite_forecast.api.core.config import get_api_settings
from infinite_forecast.api.core.logging import get_logger
from infinite_forecast.miner.forecasters.base import MinerEvent, Prediction

# Initialize logger and settings
logger = get_logger(__name__)
settings = get_api_settings()


class IfGamesClient:
    """Client for the Infinite Games API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://ifgames.win/api/v2",
        timeout: int = 30,
    ):
        """
        Initialize API client.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL for the API
            timeout: Timeout for API requests in seconds
        """
        self.api_key = api_key or settings.ifgames_api_key
        self.base_url = base_url
        self.timeout = timeout
        self.logger = logger
        
        # Validate API key
        if not self.api_key:
            self.logger.warning("No API key provided for Infinite Games API")
            
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make a request to the API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            data: Request body
            
        Returns:
            Response data
            
        Raises:
            Exception: If API request fails
        """
        # Build URL
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Build headers
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        # Add API key if available
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        
        # Make request
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    headers=headers,
                )
                
                # Check for errors
                response.raise_for_status()
                
                # Parse response
                return response.json()
                
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            # Try to parse error response
            try:
                error_data = e.response.json()
                error_message = error_data.get("detail", str(e))
            except Exception:
                error_message = str(e)
                
            raise Exception(f"API error: {error_message}")
            
        except httpx.RequestError as e:
            self.logger.error(f"Request error: {e}")
            raise Exception(f"Request error: {e}")
            
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            raise
    
    async def create_event(
        self,
        title: str,
        description: str,
        cutoff: datetime,
    ) -> Dict[str, Any]:
        """
        Create a new event.
        
        Args:
            title: Event title
            description: Event description
            cutoff: Cutoff time
            
        Returns:
            Created event data
        """
        data = {
            "title": title,
            "description": description,
            "cutoff": cutoff.isoformat(),
        }
        
        return await self._request("POST", "/events", data=data)
    
    async def get_event(self, event_id: str) -> Dict[str, Any]:
        """
        Get event details.
        
        Args:
            event_id: Event ID
            
        Returns:
            Event details
        """
        return await self._request("GET", f"/validator/events/{event_id}")
    
    async def get_community_prediction(self, event_id: str) -> Dict[str, Any]:
        """
        Get community prediction for an event.
        
        Args:
            event_id: Event ID
            
        Returns:
            Community prediction data
        """
        return await self._request("GET", f"/validator/events/{event_id}/community_prediction")
    
    async def get_event_predictions(self, event_id: str) -> Dict[str, Any]:
        """
        Get predictions for an event.
        
        Args:
            event_id: Event ID
            
        Returns:
            Event predictions
        """
        return await self._request("GET", f"/validator/events/{event_id}/predictions")
    
    async def get_all_events(
        self,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all events.
        
        Args:
            params: Query parameters
            
        Returns:
            List of events
        """
        # In a real implementation, this would call the appropriate endpoint
        # For now, simulate it with a dummy implementation
        self.logger.warning("get_all_events is not implemented in the API, using mock data")
        
        # Simulate a short delay
        await asyncio.sleep(0.2)
        
        # Return mock data
        return [
            {
                "unique_event_id": "crypto-123",
                "event_id": "crypto-123",
                "market_type": "crypto",
                "event_type": "crypto",
                "description": "Bitcoin will exceed $60,000 by the end of May 2023",
                "starts": datetime.utcnow().isoformat(),
                "resolve_date": datetime.utcnow().isoformat(),
                "cutoff": (datetime.utcnow() + datetime.timedelta(days=7)).isoformat(),
                "end_date": (datetime.utcnow() + datetime.timedelta(days=30)).isoformat(),
                "status": 1,
                "metadata": json.dumps({"symbol": "BTC"}),
            },
            {
                "unique_event_id": "fred-456",
                "event_id": "fred-456",
                "market_type": "fred",
                "event_type": "fred",
                "description": "US inflation rate will exceed 3% in June 2023",
                "starts": datetime.utcnow().isoformat(),
                "resolve_date": datetime.utcnow().isoformat(),
                "cutoff": (datetime.utcnow() + datetime.timedelta(days=14)).isoformat(),
                "end_date": (datetime.utcnow() + datetime.timedelta(days=45)).isoformat(),
                "status": 1,
                "metadata": json.dumps({"indicator": "CPI"}),
            },
        ]
    
    def event_response_to_miner_event(self, event_data: Dict[str, Any]) -> MinerEvent:
        """
        Convert API response to MinerEvent.
        
        Args:
            event_data: Event data from API
            
        Returns:
            MinerEvent object
        """
        # Parse dates
        starts = datetime.fromisoformat(event_data["starts"].replace("Z", "+00:00"))
        resolve_date = datetime.fromisoformat(event_data["resolve_date"].replace("Z", "+00:00"))
        cutoff = datetime.fromisoformat(event_data["cutoff"].replace("Z", "+00:00"))
        end_date = datetime.fromisoformat(event_data["end_date"].replace("Z", "+00:00"))
        
        # Parse metadata
        metadata = {}
        if "metadata" in event_data and event_data["metadata"]:
            try:
                if isinstance(event_data["metadata"], str):
                    metadata = json.loads(event_data["metadata"])
                else:
                    metadata = event_data["metadata"]
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse metadata: {event_data['metadata']}")
        
        # Map status
        status_map = {
            0: "pending",
            1: "active",
            2: "resolved",
            3: "cancelled",
        }
        status = status_map.get(event_data.get("status", 1), "active")
        
        # Create MinerEvent
        return MinerEvent(
            event_id=event_data["event_id"],
            market_type=event_data["market_type"],
            description=event_data["description"],
            cutoff=cutoff,
            starts=starts,
            resolve_date=resolve_date,
            end_date=end_date,
            status=status,
            outcome=event_data.get("outcome"),
            metadata=metadata,
        )


# Create singleton instance
ifgames_client = IfGamesClient()

# Convenience function to get client
def get_ifgames_client() -> IfGamesClient:
    """
    Get Infinite Games API client instance.
    
    Returns:
        IfGamesClient instance
    """
    return ifgames_client 