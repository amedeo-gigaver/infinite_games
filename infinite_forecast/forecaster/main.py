"""
Main forecaster module for Infinite Games predictions.

This module provides a unified interface for running different forecasting strategies
against the Infinite Games prediction platform, with a focus on cryptocurrency events.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Type, Union

from infinite_forecast.api.client.infinite_games import InfiniteGamesClient
from infinite_forecast.api.core.logging import get_logger
from infinite_forecast.forecaster.crypto_forecaster import CryptoForecaster, CryptoEventData

logger = get_logger(__name__)


class ForecasterManager:
    """Manager class for coordinating forecasting operations."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://ifgames.win/api/v2",
        max_concurrent_forecasts: int = 5
    ):
        """Initialize the forecaster manager.
        
        Args:
            api_key: API key for Infinite Games platform
            base_url: Base URL for Infinite Games API
            max_concurrent_forecasts: Maximum number of concurrent forecasts to run
        """
        self.client = InfiniteGamesClient(api_key=api_key, base_url=base_url)
        self.max_concurrent_forecasts = max_concurrent_forecasts
        
        # Initialize forecasters
        self.crypto_forecaster = CryptoForecaster()
        
        # Track processed events
        self.processed_events = set()
        
    async def get_open_events(
        self,
        limit: int = 50,
        market_type: Optional[str] = None,
        processed_only: bool = False,
        unprocessed_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Get open events from Infinite Games platform.
        
        Args:
            limit: Maximum number of events to retrieve
            market_type: Optional filter for market type ("crypto", etc.)
            processed_only: Only return events that have been processed
            unprocessed_only: Only return events that have not been processed
            
        Returns:
            List of event dictionaries
        """
        # Get events from API
        if market_type == "crypto":
            events = await self.client.get_crypto_events(limit=limit)
        else:
            events = await self.client.get_events(limit=limit)
            
        # Filter events if requested
        if processed_only:
            events = [e for e in events if e.get("id") in self.processed_events]
        elif unprocessed_only:
            events = [e for e in events if e.get("id") not in self.processed_events]
            
        return events
    
    async def process_crypto_events(
        self,
        limit: int = 10,
        force_reprocess: bool = False
    ) -> List[Dict[str, Any]]:
        """Process open cryptocurrency events.
        
        Args:
            limit: Maximum number of events to process
            force_reprocess: Whether to force reprocessing of previously processed events
            
        Returns:
            List of processing results
        """
        # Get unprocessed crypto events
        if force_reprocess:
            events = await self.client.get_crypto_events(limit=limit)
        else:
            events = await self.get_open_events(
                limit=limit,
                market_type="crypto",
                unprocessed_only=True
            )
            
        if not events:
            logger.info("No new crypto events to process")
            return []
            
        logger.info(f"Processing {len(events)} cryptocurrency events")
        
        # Process events with rate limiting
        semaphore = asyncio.Semaphore(self.max_concurrent_forecasts)
        
        async def process_with_semaphore(event):
            async with semaphore:
                try:
                    result = await self.process_crypto_event(event)
                    # Mark as processed after successful processing
                    self.processed_events.add(event.get("id"))
                    return result
                except Exception as e:
                    logger.error(f"Error processing event {event.get('id')}: {str(e)}")
                    return {
                        "event_id": event.get("id"),
                        "error": str(e),
                        "success": False
                    }
        
        tasks = [process_with_semaphore(event) for event in events]
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def process_crypto_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single cryptocurrency event.
        
        Args:
            event: Event dictionary from Infinite Games API
            
        Returns:
            Processing result dictionary
        """
        event_id = event.get("id")
        logger.info(f"Processing crypto event {event_id}")
        
        try:
            # Parse event data
            try:
                event_data = CryptoEventData.from_event_dict(event)
            except ValueError as e:
                logger.error(f"Failed to parse event {event_id}: {str(e)}")
                return {
                    "event_id": event_id,
                    "error": f"Failed to parse event: {str(e)}",
                    "success": False
                }
            
            # Generate prediction
            prediction, confidence, reasoning = await self.crypto_forecaster.predict_price(event_data)
            
            # Format for submission
            submission_data = self.crypto_forecaster.format_prediction_for_submission(
                prediction, 
                confidence,
                event_data.options
            )
            
            # Submit prediction
            try:
                submission_result = await self.client.submit_prediction(
                    event_id=event_id,
                    prediction=submission_data["prediction"],
                    confidence=submission_data["confidence"],
                    reasoning=reasoning,
                    metadata={
                        "symbol": event_data.symbol,
                        "prediction_time": datetime.now().isoformat(),
                        "target_date": event_data.target_date.isoformat(),
                        "model": "crypto_forecaster_v1"
                    }
                )
                
                result = {
                    "event_id": event_id,
                    "symbol": event_data.symbol,
                    "prediction": submission_data["prediction"],
                    "confidence": submission_data["confidence"],
                    "target_date": event_data.target_date.isoformat(),
                    "reasoning": reasoning,
                    "submission_result": submission_result,
                    "success": True
                }
                
                logger.info(f"Successfully processed event {event_id}")
                return result
                
            except Exception as e:
                logger.error(f"Failed to submit prediction for event {event_id}: {str(e)}")
                return {
                    "event_id": event_id,
                    "symbol": event_data.symbol,
                    "prediction": submission_data["prediction"],
                    "confidence": submission_data["confidence"],
                    "reasoning": reasoning,
                    "error": f"Failed to submit prediction: {str(e)}",
                    "success": False
                }
                
        except Exception as e:
            logger.error(f"Error processing event {event_id}: {str(e)}")
            return {
                "event_id": event_id,
                "error": str(e),
                "success": False
            }
    
    async def run_periodic_forecasting(
        self,
        interval_seconds: int = 3600,  # Default: 1 hour
        limit_per_cycle: int = 10
    ):
        """Run the forecasting process periodically.
        
        Args:
            interval_seconds: Seconds between forecasting runs
            limit_per_cycle: Maximum events to process per cycle
        """
        logger.info(f"Starting periodic forecasting every {interval_seconds} seconds")
        
        while True:
            try:
                # Process crypto events
                await self.process_crypto_events(limit=limit_per_cycle)
                
                # Add other event types here as needed
                
            except Exception as e:
                logger.error(f"Error in forecasting cycle: {str(e)}")
                
            # Wait for next cycle
            await asyncio.sleep(interval_seconds)
    
    async def evaluate_performance(
        self,
        days: int = 30,
        market_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate forecasting performance over a period.
        
        Args:
            days: Number of days to look back
            market_type: Optional market type filter
            
        Returns:
            Performance metrics
        """
        if market_type == "crypto" or market_type is None:
            # Evaluate crypto forecaster performance
            crypto_performance = await self.crypto_forecaster.evaluate_historical_performance(days)
            
            # For now, just return crypto performance
            if market_type == "crypto":
                return crypto_performance
                
            # If no specific market type, include all available metrics
            return {
                "crypto": crypto_performance,
                # Add other market types here as they're implemented
            }
            
        # Market type not supported
        return {
            "error": f"Unsupported market type: {market_type}",
            "supported_types": ["crypto"]
        }


async def run_forecaster():
    """Run the forecaster as a standalone process."""
    # Create forecaster manager
    manager = ForecasterManager()
    
    # Process current events
    results = await manager.process_crypto_events()
    print(f"Processed {len(results)} events")
    
    for result in results:
        if result.get("success"):
            print(f"✅ Event {result['event_id']} - {result['symbol']}: "
                  f"{result['prediction']} (confidence: {result['confidence']:.2f})")
        else:
            print(f"❌ Event {result['event_id']}: {result.get('error', 'Unknown error')}")
    
    # Evaluate performance
    performance = await manager.evaluate_performance()
    if "crypto" in performance and "average_accuracy" in performance["crypto"]:
        print(f"\nAverage accuracy: {performance['crypto']['average_accuracy']:.2%}")


if __name__ == "__main__":
    asyncio.run(run_forecaster()) 