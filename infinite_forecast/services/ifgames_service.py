"""
Infinite Games integration service for handling event processing and prediction submission.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Type

from infinite_forecast.api.core.config import get_api_settings
from infinite_forecast.api.core.logging import get_logger
from infinite_forecast.miner.forecasters.base import BaseForecaster, MinerEvent, Prediction
from infinite_forecast.miner.forecasters.crypto import CryptoForecaster
from infinite_forecast.miner.forecasters.fred import FredForecaster
from infinite_forecast.miner.forecasters.llm import LLMForecaster
from infinite_forecast.utils.api.ifgames import get_ifgames_client

# Initialize logger and settings
logger = get_logger(__name__)
settings = get_api_settings()


class IFGamesService:
    """Service for integrating with the Infinite Games platform."""

    def __init__(self):
        """Initialize the service."""
        self.logger = logger
        self.client = get_ifgames_client()
        
        # Register forecasters
        self.forecasters: Dict[str, Type[BaseForecaster]] = {
            "crypto": CryptoForecaster,
            "fred": FredForecaster,
            "llm": LLMForecaster,
            # Add more forecasters as needed
        }
        
        # Prediction cache
        self.prediction_cache: Dict[str, Dict[str, Any]] = {}
        
    async def get_eligible_events(self) -> List[MinerEvent]:
        """
        Get eligible events for forecasting.
        
        Returns:
            List of eligible events
        """
        # Get all events
        events_data = await self.client.get_all_events()
        
        # Convert to MinerEvent objects
        events = [self.client.event_response_to_miner_event(event) for event in events_data]
        
        # Filter events by eligibility (active and not yet past cutoff)
        now = datetime.utcnow()
        eligible_events = [
            event for event in events
            if (
                event.status == "active" and
                event.cutoff > now and
                # Only process events for which we have a forecaster
                event.market_type in self.forecasters
            )
        ]
        
        self.logger.info(f"Found {len(eligible_events)} eligible events out of {len(events)} total")
        return eligible_events
    
    async def process_events(self) -> None:
        """Process all eligible events and submit predictions."""
        # Get eligible events
        events = await self.get_eligible_events()
        
        # Process each event
        for event in events:
            try:
                # Check if we've already processed this event recently
                if self._is_recently_predicted(event.event_id):
                    self.logger.info(f"Skipping event {event.event_id} - already processed recently")
                    continue
                
                # Get the appropriate forecaster
                forecaster_class = self.forecasters.get(event.market_type)
                if not forecaster_class:
                    self.logger.warning(f"No forecaster found for event type: {event.market_type}")
                    continue
                
                # Instantiate the forecaster
                forecaster = forecaster_class()
                
                # Run the forecaster
                prediction = await forecaster.forecast(event)
                
                # Cache the prediction
                self._cache_prediction(event.event_id, prediction)
                
                # Log the prediction
                self.logger.info(
                    f"Generated prediction for event {event.event_id}: "
                    f"probability={prediction.probability:.4f}, "
                    f"confidence={prediction.confidence:.4f}"
                )
                
                # TODO: Submit the prediction to the platform
                # This would typically involve calling an API endpoint
                # await self._submit_prediction(event.event_id, prediction)
                
            except Exception as e:
                self.logger.error(f"Error processing event {event.event_id}: {e}")
    
    def _is_recently_predicted(self, event_id: str) -> bool:
        """
        Check if an event has been predicted recently.
        
        Args:
            event_id: Event ID
            
        Returns:
            True if predicted recently, False otherwise
        """
        if event_id not in self.prediction_cache:
            return False
        
        prediction_data = self.prediction_cache[event_id]
        timestamp = prediction_data.get("timestamp")
        
        # If no timestamp or older than 24 hours, not recent
        if not timestamp:
            return False
        
        now = datetime.utcnow()
        return (now - timestamp) < timedelta(hours=24)
    
    def _cache_prediction(self, event_id: str, prediction: Prediction) -> None:
        """
        Cache a prediction for an event.
        
        Args:
            event_id: Event ID
            prediction: Prediction object
        """
        self.prediction_cache[event_id] = {
            "prediction": prediction,
            "timestamp": datetime.utcnow(),
        }
    
    async def _submit_prediction(self, event_id: str, prediction: Prediction) -> Dict[str, Any]:
        """
        Submit a prediction to the platform.
        
        Args:
            event_id: Event ID
            prediction: Prediction object
            
        Returns:
            Response data
        """
        # This is where you would implement the API call to submit the prediction
        # Since the exact endpoint wasn't specified in the provided API docs,
        # this is left as a placeholder
        
        self.logger.info(f"Submitting prediction for event {event_id}")
        
        # Mock implementation
        await asyncio.sleep(0.2)
        return {
            "event_id": event_id,
            "prediction": prediction.probability,
            "confidence": prediction.confidence,
            "status": "submitted",
        }

    async def run_periodic(self, interval_seconds: int = 3600) -> None:
        """
        Run the service periodically.
        
        Args:
            interval_seconds: Interval between runs in seconds (default: 1 hour)
        """
        self.logger.info(f"Starting periodic runs every {interval_seconds} seconds")
        
        while True:
            try:
                await self.process_events()
            except Exception as e:
                self.logger.error(f"Error in periodic run: {e}")
            
            # Wait for the next interval
            await asyncio.sleep(interval_seconds)


# Create singleton instance
ifgames_service = IFGamesService()

# Convenience function to get service
def get_ifgames_service() -> IFGamesService:
    """
    Get Infinite Games service instance.
    
    Returns:
        IFGamesService instance
    """
    return ifgames_service 