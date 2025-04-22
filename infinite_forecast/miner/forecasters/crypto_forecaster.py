"""
Cryptocurrency price forecaster for predicting future crypto prices.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
from pydantic import BaseModel, Field

from infinite_forecast.miner.utils.market_data import (
    get_crypto_historical_data,
    get_price_trend,
    get_market_sentiment,
    calculate_technical_indicators
)
from infinite_forecast.api.client.infinite_games import InfiniteGamesClient
from infinite_forecast.api.core.logging import get_logger

logger = get_logger(__name__)

class CryptoPrediction(BaseModel):
    """Model for cryptocurrency price predictions."""
    symbol: str = Field(..., description="Cryptocurrency symbol (e.g., BTC)")
    target_date: datetime = Field(..., description="Target date for prediction")
    predicted_price: float = Field(..., description="Predicted price in USD")
    confidence: float = Field(..., description="Confidence level (0.0-1.0)")
    reasoning: str = Field(..., description="Reasoning behind prediction")
    technical_indicators: Dict[str, float] = Field(
        default_factory=dict, 
        description="Technical indicators used in prediction"
    )
    recent_trend: str = Field(..., description="Recent price trend")
    market_sentiment: Dict[str, float] = Field(
        default_factory=dict, 
        description="Market sentiment indicators"
    )

class CryptoForecaster:
    """Forecaster for cryptocurrency price predictions."""
    
    def __init__(self, api_client: Optional[InfiniteGamesClient] = None):
        """Initialize the crypto forecaster.
        
        Args:
            api_client: Optional client for Infinite Games API
        """
        self.api_client = api_client or InfiniteGamesClient()
        
    async def _analyze_historical_data(
        self, 
        symbol: str, 
        days: int = 30
    ) -> Tuple[List[Dict], Dict, str, Dict]:
        """Analyze historical data for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., BTC)
            days: Number of days of historical data to analyze
            
        Returns:
            Tuple containing:
            - Historical data
            - Technical indicators
            - Price trend
            - Market sentiment
        """
        # Get historical data
        historical_data = await asyncio.to_thread(
            get_crypto_historical_data,
            symbol,
            days
        )
        
        if not historical_data:
            logger.error(f"Failed to get historical data for {symbol}")
            return [], {}, "unknown", {}
            
        # Calculate technical indicators
        indicators = await asyncio.to_thread(
            calculate_technical_indicators,
            historical_data
        )
        
        # Get price trend
        trend = await asyncio.to_thread(
            get_price_trend,
            historical_data
        )
        
        # Get market sentiment
        sentiment = await asyncio.to_thread(
            get_market_sentiment,
            symbol
        )
        
        return historical_data, indicators, trend, sentiment
    
    async def _calculate_prediction(
        self, 
        symbol: str, 
        target_date: datetime,
        historical_data: List[Dict],
        indicators: Dict,
        trend: str,
        sentiment: Dict
    ) -> CryptoPrediction:
        """Calculate price prediction based on available data.
        
        Args:
            symbol: Cryptocurrency symbol
            target_date: Target date for prediction
            historical_data: Historical price data
            indicators: Technical indicators
            trend: Price trend
            sentiment: Market sentiment
            
        Returns:
            Prediction model with price and confidence
        """
        if not historical_data:
            logger.error(f"Cannot make prediction for {symbol} without historical data")
            return None
            
        # Extract closing prices
        prices = [entry["price"] for entry in historical_data]
        
        # Calculate basic statistics
        current_price = prices[-1]
        avg_price = sum(prices) / len(prices)
        std_dev = np.std(prices)
        
        # Calculate days to target
        days_to_target = (target_date - datetime.now()).days
        
        # Base prediction on recent trend
        if trend == "strong_up":
            predicted_change = 0.05 * days_to_target  # 5% per day
            confidence = 0.8
        elif trend == "up":
            predicted_change = 0.02 * days_to_target  # 2% per day
            confidence = 0.7
        elif trend == "neutral":
            predicted_change = 0.005 * days_to_target  # 0.5% per day
            confidence = 0.6
        elif trend == "down":
            predicted_change = -0.01 * days_to_target  # -1% per day
            confidence = 0.7
        elif trend == "strong_down":
            predicted_change = -0.03 * days_to_target  # -3% per day
            confidence = 0.8
        else:
            predicted_change = 0
            confidence = 0.5
            
        # Adjust based on technical indicators
        if indicators.get("rsi", 50) > 70:  # Overbought
            predicted_change *= 0.8
            confidence *= 0.9
        elif indicators.get("rsi", 50) < 30:  # Oversold
            predicted_change *= 1.2
            confidence *= 0.9
            
        # Adjust based on MACD
        if indicators.get("macd_signal", 0) > 0:
            predicted_change *= 1.1
        else:
            predicted_change *= 0.9
            
        # Adjust based on sentiment
        sentiment_score = sentiment.get("sentiment_score", 0.5)
        predicted_change *= (0.5 + sentiment_score)
        
        # Calculate final predicted price
        predicted_price = current_price * (1 + predicted_change)
        
        # Cap confidence
        confidence = min(max(confidence, 0.3), 0.9)
        
        # Generate reasoning
        reasoning = self._generate_prediction_reasoning(
            symbol, trend, indicators, sentiment, predicted_change
        )
        
        return CryptoPrediction(
            symbol=symbol,
            target_date=target_date,
            predicted_price=predicted_price,
            confidence=confidence,
            reasoning=reasoning,
            technical_indicators=indicators,
            recent_trend=trend,
            market_sentiment=sentiment
        )
    
    def _generate_prediction_reasoning(
        self, 
        symbol: str, 
        trend: str, 
        indicators: Dict, 
        sentiment: Dict, 
        predicted_change: float
    ) -> str:
        """Generate human-readable reasoning for prediction.
        
        Args:
            symbol: Cryptocurrency symbol
            trend: Price trend
            indicators: Technical indicators
            sentiment: Market sentiment
            predicted_change: Predicted price change
            
        Returns:
            String with reasoning
        """
        trend_desc = {
            "strong_up": "strongly upward",
            "up": "upward",
            "neutral": "neutral",
            "down": "downward",
            "strong_down": "strongly downward"
        }.get(trend, "uncertain")
        
        rsi = indicators.get("rsi", 50)
        rsi_desc = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
        
        volatility = indicators.get("volatility", 0.0)
        volatility_desc = "high" if volatility > 0.05 else "moderate" if volatility > 0.02 else "low"
        
        sentiment_score = sentiment.get("sentiment_score", 0.5)
        sentiment_desc = "bullish" if sentiment_score > 0.6 else "bearish" if sentiment_score < 0.4 else "neutral"
        
        change_pct = predicted_change * 100
        direction = "increase" if change_pct > 0 else "decrease"
        
        reasoning = (
            f"Based on analysis of {symbol}, the price shows a {trend_desc} trend. "
            f"Technical indicators suggest {rsi_desc} conditions with {volatility_desc} volatility. "
            f"Market sentiment appears {sentiment_desc}. "
            f"These factors suggest a projected {abs(change_pct):.2f}% {direction} in price."
        )
        
        return reasoning
    
    async def predict_crypto_price(
        self, 
        symbol: str, 
        target_date: Optional[datetime] = None,
        days_ahead: int = 7
    ) -> CryptoPrediction:
        """Predict cryptocurrency price for a future date.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., BTC)
            target_date: Target date for prediction (default: days_ahead from now)
            days_ahead: Days ahead to predict if target_date not specified
            
        Returns:
            Prediction model with price and confidence
        """
        if not target_date:
            target_date = datetime.now() + timedelta(days=days_ahead)
            
        # Analyze historical data
        historical_data, indicators, trend, sentiment = await self._analyze_historical_data(symbol)
        
        # Calculate prediction
        prediction = await self._calculate_prediction(
            symbol, target_date, historical_data, indicators, trend, sentiment
        )
        
        return prediction
    
    async def submit_crypto_prediction(
        self, 
        event_id: str, 
        prediction: CryptoPrediction
    ) -> Dict:
        """Submit a crypto prediction to Infinite Games.
        
        Args:
            event_id: ID of the event to submit prediction for
            prediction: Prediction model
            
        Returns:
            Response from Infinite Games API
        """
        # Format prediction for API
        prediction_data = {
            "prediction": {
                "value": prediction.predicted_price,
                "confidence": prediction.confidence
            },
            "reasoning": prediction.reasoning,
            "metadata": {
                "technical_indicators": prediction.technical_indicators,
                "recent_trend": prediction.recent_trend,
                "market_sentiment": prediction.market_sentiment
            }
        }
        
        # Submit to API
        response = await self.api_client.submit_prediction(event_id, prediction_data)
        
        if "error" in response:
            logger.error(f"Failed to submit prediction for event {event_id}: {response['error']}")
        else:
            logger.info(f"Successfully submitted prediction for event {event_id}")
            
        return response
    
    async def process_crypto_events(self) -> List[Dict]:
        """Process all open crypto events.
        
        Returns:
            List of processed events with predictions
        """
        results = []
        
        # Get open crypto events
        events = await self.api_client.get_crypto_events(status="open")
        
        for event in events:
            try:
                event_id = event.get("id")
                event_details = await self.api_client.get_event(event_id)
                
                if "error" in event_details:
                    logger.error(f"Failed to get details for event {event_id}")
                    continue
                    
                # Extract relevant information
                symbol = event_details.get("metadata", {}).get("symbol", "BTC")
                close_date = datetime.fromisoformat(event_details.get("close_time"))
                
                # Make prediction
                prediction = await self.predict_crypto_price(symbol, close_date)
                
                # Submit prediction
                submission = await self.submit_crypto_prediction(event_id, prediction)
                
                results.append({
                    "event_id": event_id,
                    "symbol": symbol,
                    "prediction": prediction.dict(),
                    "submission_result": submission
                })
                
            except Exception as e:
                logger.error(f"Error processing event: {str(e)}")
                
        return results 