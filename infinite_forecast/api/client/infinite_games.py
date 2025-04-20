"""
API client for interacting with the Infinite Games prediction platform.
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urljoin

import aiohttp

from infinite_forecast.api.core.logging import get_logger

logger = get_logger(__name__)


class InfiniteGamesClient:
    """Client for interacting with the Infinite Games prediction platform API.
    
    This client handles authentication, API requests, and response parsing
    for the Infinite Games platform, focusing on cryptocurrency forecasting.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://ifgames.win/api/v2",
        timeout: int = 30
    ):
        """Initialize the Infinite Games API client.
        
        Args:
            api_key: API key for authentication. If not provided, will look for
                    INFINITE_GAMES_API_KEY environment variable.
            base_url: Base URL for the API.
            timeout: Timeout for API requests in seconds.
        """
        self.api_key = api_key or os.getenv("INFINITE_GAMES_API_KEY")
        if not self.api_key:
            logger.warning("No API key provided for Infinite Games client")
        
        self.base_url = base_url
        self.timeout = timeout
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make an HTTP request to the Infinite Games API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Query parameters
            data: Form data
            json_data: JSON data for POST requests
            
        Returns:
            API response as dictionary
            
        Raises:
            ValueError: If the API returns an error response
            aiohttp.ClientError: If there is a connection or HTTP error
        """
        url = urljoin(self.base_url, endpoint)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    data=data,
                    json=json_data,
                    timeout=self.timeout
                ) as response:
                    # Check if response is successful
                    if response.status >= 400:
                        error_text = await response.text()
                        logger.error(f"API error ({response.status}): {error_text}")
                        raise ValueError(f"API error ({response.status}): {error_text}")
                    
                    # Try to parse JSON response
                    try:
                        result = await response.json()
                        return result
                    except json.JSONDecodeError:
                        # If not JSON, return text
                        text = await response.text()
                        logger.error(f"Invalid JSON response: {text}")
                        return {"text": text}
        
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error during API request: {str(e)}")
            raise
    
    async def get_events(
        self,
        limit: int = 100,
        offset: int = 0,
        status: Optional[str] = None,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get a list of events from the Infinite Games platform.
        
        Args:
            limit: Maximum number of events to return
            offset: Offset for pagination
            status: Filter by event status (open, closed, resolved)
            category: Filter by event category
            
        Returns:
            List of event objects
        """
        params = {
            "limit": limit,
            "offset": offset
        }
        
        if status:
            params["status"] = status
        
        if category:
            params["category"] = category
        
        response = await self._request("GET", "/validator/events", params=params)
        return response.get("data", [])
    
    async def get_crypto_events(
        self,
        limit: int = 100,
        offset: int = 0,
        status: Optional[str] = "open"
    ) -> List[Dict[str, Any]]:
        """Get a list of cryptocurrency-related events.
        
        Args:
            limit: Maximum number of events to return
            offset: Offset for pagination
            status: Filter by event status (open, closed, resolved)
            
        Returns:
            List of cryptocurrency event objects
        """
        return await self.get_events(
            limit=limit,
            offset=offset,
            status=status,
            category="crypto"
        )
    
    async def get_event(self, event_id: str) -> Dict[str, Any]:
        """Get details for a specific event.
        
        Args:
            event_id: ID of the event to retrieve
            
        Returns:
            Event details
        """
        response = await self._request("GET", f"/validator/events/{event_id}")
        return response.get("data", {})
    
    async def get_community_prediction(self, event_id: str) -> Dict[str, Any]:
        """Get the community prediction for an event.
        
        Args:
            event_id: ID of the event
            
        Returns:
            Community prediction data
        """
        response = await self._request(
            "GET", 
            f"/validator/events/{event_id}/community_prediction"
        )
        return response.get("data", {})
    
    async def get_predictions(self, event_id: str) -> List[Dict[str, Any]]:
        """Get all predictions for an event.
        
        Args:
            event_id: ID of the event
            
        Returns:
            List of predictions
        """
        response = await self._request(
            "GET", 
            f"/validator/events/{event_id}/predictions"
        )
        return response.get("data", [])
    
    async def submit_prediction(
        self,
        event_id: str,
        prediction: Union[str, float, bool],
        confidence: float,
        reasoning: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Submit a prediction for an event.
        
        Args:
            event_id: ID of the event
            prediction: The prediction value (can be string for multiple choice,
                        float for numeric prediction, or boolean)
            confidence: Confidence score (0.0-1.0)
            reasoning: Explanation of the prediction
            metadata: Additional metadata about the prediction
            
        Returns:
            API response data
            
        Raises:
            ValueError: If the prediction submission fails
        """
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        data = {
            "prediction": prediction,
            "confidence": confidence
        }
        
        if reasoning:
            data["reasoning"] = reasoning
            
        if metadata:
            data["metadata"] = metadata
        
        try:
            response = await self._request(
                "POST",
                f"/validator/events/{event_id}/predict",
                json_data=data
            )
            return response.get("data", {})
        except (ValueError, aiohttp.ClientError) as e:
            logger.error(f"Failed to submit prediction for event {event_id}: {str(e)}")
            raise ValueError(f"Prediction submission failed: {str(e)}") 