"""
Base forecaster class for the miner.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from infinite_forecast.api.core.logging import get_logger

# Initialize logger
logger = get_logger(__name__)


class EventType(str, Enum):
    """Types of events that can be predicted."""
    
    LLM = "llm"
    FRED = "fred"
    CRYPTO = "crypto"
    EARNINGS = "earnings"
    POLYMARKET = "polymarket"


class MinerEvent:
    """Event model for miners."""
    
    def __init__(
        self,
        event_id: str,
        market_type: str,
        description: str,
        cutoff: datetime,
        starts: datetime,
        resolve_date: datetime,
        end_date: datetime,
        status: str = "active",
        outcome: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize miner event.
        
        Args:
            event_id: Unique event identifier
            market_type: Type of market for this event
            description: Human-readable description of the event
            cutoff: Time after which predictions cannot be modified
            starts: Time when the event starts
            resolve_date: Expected date of resolution
            end_date: End date for the event
            status: Current status of the event
            outcome: Actual outcome (0 or 1) if resolved
            metadata: Additional metadata for the event
        """
        self.event_id = event_id
        self.market_type = market_type
        self.description = description
        self.cutoff = cutoff
        self.starts = starts
        self.resolve_date = resolve_date
        self.end_date = end_date
        self.status = status
        self.outcome = outcome
        self.metadata = metadata or {}
        
    @property
    def event_type(self) -> EventType:
        """
        Get event type.
        
        Returns:
            EventType
        """
        # Extract event type from event ID
        # Format: <event_type>-<id>
        if "-" in self.event_id:
            try:
                type_str = self.event_id.split("-")[0]
                return EventType(type_str)
            except ValueError:
                pass
        
        # Fallback to market type
        try:
            return EventType(self.market_type)
        except ValueError:
            # Default to LLM if unknown
            return EventType.LLM
    
    @property
    def is_active(self) -> bool:
        """
        Check if event is active.
        
        Returns:
            True if active
        """
        return self.status == "active"
    
    @property
    def is_resolved(self) -> bool:
        """
        Check if event is resolved.
        
        Returns:
            True if resolved
        """
        return self.status == "resolved"
    
    @property
    def time_to_cutoff(self) -> timedelta:
        """
        Get time remaining until cutoff.
        
        Returns:
            Time remaining
        """
        now = datetime.utcnow()
        return max(timedelta(0), self.cutoff - now)
    
    @property
    def can_predict(self) -> bool:
        """
        Check if prediction is still allowed.
        
        Returns:
            True if prediction is allowed
        """
        return self.is_active and datetime.utcnow() < self.cutoff
    
    @property
    def is_crypto(self) -> bool:
        """
        Check if event is related to cryptocurrency.
        
        Returns:
            True if crypto-related
        """
        return self.event_type == EventType.CRYPTO
    
    @property
    def is_economic(self) -> bool:
        """
        Check if event is related to economic data.
        
        Returns:
            True if economic-related
        """
        return self.event_type == EventType.FRED
    
    @property
    def is_earnings(self) -> bool:
        """
        Check if event is related to earnings.
        
        Returns:
            True if earnings-related
        """
        return self.event_type == EventType.EARNINGS
    
    @property
    def is_political(self) -> bool:
        """
        Check if event is related to politics.
        
        Returns:
            True if politics-related
        """
        return self.event_type == EventType.LLM and (
            "politics" in self.metadata.get("categories", []) or
            "geopolitics" in self.metadata.get("categories", [])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert event to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "event_id": self.event_id,
            "market_type": self.market_type,
            "description": self.description,
            "cutoff": self.cutoff.isoformat(),
            "starts": self.starts.isoformat(),
            "resolve_date": self.resolve_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "status": self.status,
            "outcome": self.outcome,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MinerEvent":
        """
        Create event from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            MinerEvent object
        """
        # Convert ISO strings to datetime
        cutoff = datetime.fromisoformat(data["cutoff"])
        starts = datetime.fromisoformat(data["starts"])
        resolve_date = datetime.fromisoformat(data["resolve_date"])
        end_date = datetime.fromisoformat(data["end_date"])
        
        return cls(
            event_id=data["event_id"],
            market_type=data["market_type"],
            description=data["description"],
            cutoff=cutoff,
            starts=starts,
            resolve_date=resolve_date,
            end_date=end_date,
            status=data.get("status", "active"),
            outcome=data.get("outcome"),
            metadata=data.get("metadata", {}),
        )


class Prediction:
    """Prediction model for forecasters."""
    
    def __init__(
        self,
        event_id: str,
        probability: float,
        timestamp: datetime = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize prediction.
        
        Args:
            event_id: Event ID
            probability: Probability of the event occurring (0-1)
            timestamp: Time when the prediction was made
            confidence: Confidence in the prediction (0-1)
            metadata: Additional metadata for the prediction
        """
        self.event_id = event_id
        self.probability = max(0.0, min(1.0, probability))
        self.timestamp = timestamp or datetime.utcnow()
        self.confidence = confidence
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert prediction to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "event_id": self.event_id,
            "probability": self.probability,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Prediction":
        """
        Create prediction from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Prediction object
        """
        # Convert ISO string to datetime
        timestamp = datetime.fromisoformat(data["timestamp"])
        
        return cls(
            event_id=data["event_id"],
            probability=data["probability"],
            timestamp=timestamp,
            confidence=data.get("confidence"),
            metadata=data.get("metadata", {}),
        )


class BaseForecaster(ABC):
    """Base forecaster class."""
    
    def __init__(
        self,
        event: MinerEvent,
        logger=None,
        cache_ttl: int = 3600,  # 1 hour
        recalculation_interval: int = 14400,  # 4 hours
    ):
        """
        Initialize forecaster.
        
        Args:
            event: Event to forecast
            logger: Logger instance
            cache_ttl: Cache TTL in seconds
            recalculation_interval: Interval between recalculations in seconds
        """
        self.event = event
        self.logger = logger or get_logger(self.__class__.__name__)
        self.cache_ttl = cache_ttl
        self.recalculation_interval = recalculation_interval
        
        # Previous prediction
        self.last_prediction: Optional[Prediction] = None
        self.last_prediction_time: Optional[datetime] = None
        
        # Performance metrics
        self.prediction_count = 0
        self.total_processing_time = 0.0
        
    async def predict(self) -> Prediction:
        """
        Generate a prediction for the event.
        
        Returns:
            Prediction object
        """
        # Check if event can still be predicted
        if not self.event.can_predict:
            if self.last_prediction:
                self.logger.info(
                    f"Event {self.event.event_id} past cutoff, returning last prediction"
                )
                return self.last_prediction
            else:
                self.logger.warning(
                    f"Event {self.event.event_id} past cutoff and no previous prediction"
                )
                # Return middle probability with low confidence
                return Prediction(
                    event_id=self.event.event_id,
                    probability=0.5,
                    confidence=0.1,
                    metadata={"reason": "past_cutoff_no_previous_prediction"},
                )
        
        # Check if we should use cached prediction
        if self.last_prediction and self.last_prediction_time:
            # Calculate time since last prediction
            time_since_last = datetime.utcnow() - self.last_prediction_time
            
            # If we're within recalculation interval, return cached prediction
            if time_since_last.total_seconds() < self.recalculation_interval:
                self.logger.debug(
                    f"Using cached prediction for {self.event.event_id}, "
                    f"last prediction {time_since_last.total_seconds():.1f}s ago"
                )
                return self.last_prediction
        
        # Generate new prediction
        self.logger.info(f"Generating new prediction for {self.event.event_id}")
        start_time = time.time()
        
        try:
            # Run forecaster-specific prediction logic
            prediction = await self._run()
            
            # Update metrics
            self.prediction_count += 1
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            # Cache prediction
            self.last_prediction = prediction
            self.last_prediction_time = datetime.utcnow()
            
            self.logger.info(
                f"Generated prediction {prediction.probability:.4f} for "
                f"{self.event.event_id} in {processing_time:.2f}s"
            )
            
            return prediction
            
        except Exception as e:
            self.logger.error(
                f"Error generating prediction for {self.event.event_id}: {e}",
                exc_info=True,
            )
            
            # Return middle probability with low confidence on error
            return Prediction(
                event_id=self.event.event_id,
                probability=0.5,
                confidence=0.1,
                metadata={"error": str(e)},
            )
    
    @abstractmethod
    async def _run(self) -> Prediction:
        """
        Forecaster-specific prediction logic.
        
        Returns:
            Prediction object
        """
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get forecaster metrics.
        
        Returns:
            Dictionary with metrics
        """
        return {
            "event_id": self.event.event_id,
            "forecaster": self.__class__.__name__,
            "prediction_count": self.prediction_count,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": (
                self.total_processing_time / self.prediction_count
                if self.prediction_count > 0
                else 0.0
            ),
            "last_prediction_time": (
                self.last_prediction_time.isoformat()
                if self.last_prediction_time
                else None
            ),
        }


class DummyForecaster(BaseForecaster):
    """Dummy forecaster for testing."""
    
    async def _run(self) -> Prediction:
        """
        Generate a dummy prediction.
        
        Returns:
            Prediction object
        """
        # Add small delay to simulate work
        await asyncio.sleep(0.1)
        
        # For testing, use different strategies based on event type
        if self.event.is_crypto:
            probability = 0.65
        elif self.event.is_economic:
            probability = 0.45
        elif self.event.is_earnings:
            probability = 0.55
        else:
            probability = 0.5
            
        return Prediction(
            event_id=self.event.event_id,
            probability=probability,
            confidence=0.7,
            metadata={"model": "dummy"},
        ) 