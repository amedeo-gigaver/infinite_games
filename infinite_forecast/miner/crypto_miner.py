"""
Cryptocurrency event miner for automatically processing crypto forecasting events.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from infinite_forecast.api.client.infinite_games import InfiniteGamesClient
from infinite_forecast.api.core.logging import get_logger
from infinite_forecast.miner.forecasters.crypto_forecaster import CryptoForecaster

logger = get_logger(__name__)

class CryptoMiner:
    """Service for mining cryptocurrency forecasting events from Infinite Games."""
    
    def __init__(
        self,
        poll_interval: int = 3600,  # Default: check every hour
        api_client: Optional[InfiniteGamesClient] = None,
        forecaster: Optional[CryptoForecaster] = None
    ):
        """Initialize the crypto miner service.
        
        Args:
            poll_interval: Seconds between polling for new events
            api_client: Optional client for Infinite Games API
            forecaster: Optional crypto forecaster instance
        """
        self.poll_interval = poll_interval
        self.api_client = api_client or InfiniteGamesClient()
        self.forecaster = forecaster or CryptoForecaster(api_client=self.api_client)
        self.is_running = False
        self.processed_events = set()
        
    async def start(self):
        """Start the mining service in the background."""
        if self.is_running:
            logger.warning("Crypto miner is already running")
            return
            
        self.is_running = True
        logger.info("Starting cryptocurrency event miner")
        
        try:
            while self.is_running:
                await self.process_cycle()
                await asyncio.sleep(self.poll_interval)
        except Exception as e:
            logger.error(f"Error in crypto miner main loop: {str(e)}")
            self.is_running = False
            raise
    
    def stop(self):
        """Stop the mining service."""
        logger.info("Stopping cryptocurrency event miner")
        self.is_running = False
    
    async def process_cycle(self):
        """Process one cycle of fetching and handling crypto events."""
        start_time = time.time()
        logger.info("Starting crypto event processing cycle")
        
        try:
            # Get all open crypto events
            events = await self.api_client.get_crypto_events(status="open")
            logger.info(f"Found {len(events)} open crypto events")
            
            # Process each event that hasn't been processed before
            new_events = [e for e in events if e.get("id") not in self.processed_events]
            
            if not new_events:
                logger.info("No new crypto events to process")
                return
                
            logger.info(f"Processing {len(new_events)} new crypto events")
            
            results = []
            for event in new_events:
                try:
                    event_id = event.get("id")
                    result = await self.process_event(event)
                    self.processed_events.add(event_id)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing event {event.get('id')}: {str(e)}")
            
            logger.info(f"Successfully processed {len(results)} crypto events")
            
        except Exception as e:
            logger.error(f"Error during crypto event processing cycle: {str(e)}")
        
        finally:
            duration = time.time() - start_time
            logger.info(f"Completed crypto event processing cycle in {duration:.2f} seconds")
    
    async def process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single crypto event.
        
        Args:
            event: Event data from Infinite Games API
            
        Returns:
            Dictionary with processing results
        """
        event_id = event.get("id")
        logger.info(f"Processing crypto event {event_id}")
        
        # Get full event details
        event_details = await self.api_client.get_event(event_id)
        
        if "error" in event_details:
            error_msg = f"Failed to get details for event {event_id}: {event_details['error']}"
            logger.error(error_msg)
            return {"event_id": event_id, "success": False, "error": error_msg}
        
        try:
            # Extract relevant information
            metadata = event_details.get("metadata", {})
            symbol = metadata.get("symbol", "BTC")
            close_time = event_details.get("close_time")
            
            if not close_time:
                raise ValueError(f"Event {event_id} has no close time")
                
            close_date = datetime.fromisoformat(close_time)
            
            # Check if the event is still open for predictions
            if close_date <= datetime.now():
                logger.warning(f"Event {event_id} is already closed for predictions")
                return {
                    "event_id": event_id,
                    "success": False,
                    "error": "Event is closed for predictions"
                }
            
            # Make prediction
            prediction = await self.forecaster.predict_crypto_price(symbol, close_date)
            
            # Submit prediction
            submission = await self.forecaster.submit_crypto_prediction(event_id, prediction)
            
            # Get community prediction for comparison
            community = await self.api_client.get_community_prediction(event_id)
            
            result = {
                "event_id": event_id,
                "symbol": symbol,
                "close_time": close_time,
                "our_prediction": prediction.dict() if prediction else None,
                "community_prediction": community,
                "submission_result": submission,
                "success": "error" not in submission,
                "error": submission.get("error")
            }
            
            if "error" not in submission:
                logger.info(f"Successfully submitted prediction for event {event_id}")
            else:
                logger.error(f"Failed to submit prediction for event {event_id}: {submission['error']}")
                
            return result
            
        except Exception as e:
            error_msg = f"Error processing event {event_id}: {str(e)}"
            logger.error(error_msg)
            return {"event_id": event_id, "success": False, "error": error_msg}
    
    async def run_once(self) -> List[Dict]:
        """Run the miner once to process all open events.
        
        Returns:
            List of results from processing each event
        """
        logger.info("Running one-time crypto event processing")
        
        try:
            # Get all open crypto events
            events = await self.api_client.get_crypto_events(status="open")
            logger.info(f"Found {len(events)} open crypto events")
            
            if not events:
                return []
                
            # Process each event
            results = []
            for event in events:
                try:
                    result = await self.process_event(event)
                    results.append(result)
                except Exception as e:
                    event_id = event.get("id", "unknown")
                    logger.error(f"Error processing event {event_id}: {str(e)}")
                    results.append({
                        "event_id": event_id,
                        "success": False,
                        "error": str(e)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error during one-time crypto event processing: {str(e)}")
            return [{"success": False, "error": str(e)}] 