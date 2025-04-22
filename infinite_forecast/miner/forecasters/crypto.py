"""
Cryptocurrency-specific forecaster for predicting crypto market events.
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from infinite_forecast.api.core.config import get_miner_config
from infinite_forecast.api.core.logging import get_logger
from infinite_forecast.miner.forecasters.base import BaseForecaster, MinerEvent, Prediction
from infinite_forecast.miner.forecasters.llm import LLMForecaster
from infinite_forecast.utils.llm.client import get_llm_client

# Initialize logger and config
logger = get_logger(__name__)
miner_config = get_miner_config()


class CryptoMarketData:
    """Mock class for fetching cryptocurrency market data."""
    
    def __init__(self):
        """Initialize crypto market data provider."""
        self.logger = get_logger(__name__)
        
    async def get_price_history(
        self, 
        symbol: str, 
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get price history for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC')
            days: Number of days of history to retrieve
            
        Returns:
            List of price data points
        """
        self.logger.info(f"Fetching price history for {symbol} (last {days} days)")
        
        # In a real implementation, this would call a crypto API
        # For now, return mock data
        await asyncio.sleep(0.2)  # Simulate API call
        
        # Generate mock price data
        base_price = self._get_base_price(symbol)
        volatility = self._get_volatility(symbol)
        
        # Generate a slightly upward trend with random noise
        now = datetime.utcnow()
        data = []
        
        for i in range(days):
            # Create a slightly trending time series with some random noise
            timestamp = now - timedelta(days=days-i)
            trend = 0.001 * i  # Slight upward trend
            noise = np.random.normal(0, volatility)
            price = base_price * (1 + trend + noise)
            
            # Add some volume data
            volume = np.random.randint(
                int(base_price * 1000), 
                int(base_price * 5000)
            )
            
            data.append({
                "timestamp": timestamp.isoformat(),
                "price": round(price, 2),
                "volume": volume,
                "market_cap": round(price * volume * 10, 2),
            })
            
        return data
        
    async def get_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get market sentiment for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC')
            
        Returns:
            Sentiment data
        """
        self.logger.info(f"Fetching market sentiment for {symbol}")
        
        # In a real implementation, this would call a sentiment API
        # For now, return mock data
        await asyncio.sleep(0.1)  # Simulate API call
        
        # Generate mock sentiment data
        if symbol.upper() in ["BTC", "ETH"]:
            sentiment = 0.65  # Slightly positive
        else:
            sentiment = 0.45  # Slightly negative
            
        return {
            "sentiment_score": sentiment,
            "bullish_signals": np.random.randint(5, 15),
            "bearish_signals": np.random.randint(3, 10),
            "neutral_signals": np.random.randint(8, 20),
            "social_volume": np.random.randint(1000, 10000),
        }
        
    def _get_base_price(self, symbol: str) -> float:
        """Get base price for symbol."""
        prices = {
            "BTC": 50000.0,
            "ETH": 3000.0,
            "SOL": 100.0,
            "ADA": 1.2,
            "DOT": 15.0,
        }
        return prices.get(symbol.upper(), 10.0)
        
    def _get_volatility(self, symbol: str) -> float:
        """Get volatility factor for symbol."""
        volatility = {
            "BTC": 0.03,
            "ETH": 0.04,
            "SOL": 0.06,
            "ADA": 0.05,
            "DOT": 0.05,
        }
        return volatility.get(symbol.upper(), 0.05)


def extract_crypto_symbol(event: MinerEvent) -> Optional[str]:
    """
    Extract cryptocurrency symbol from event.
    
    Args:
        event: Event to extract from
        
    Returns:
        Cryptocurrency symbol or None if not found
    """
    # Check metadata first
    if "symbol" in event.metadata:
        return event.metadata["symbol"]
        
    if "crypto_symbol" in event.metadata:
        return event.metadata["crypto_symbol"]
        
    if "asset" in event.metadata:
        return event.metadata["asset"]
        
    # Try to extract from description using regex
    # Look for common patterns like "BTC will reach..." or "price of ETH will..."
    symbol_match = re.search(
        r'\b(BTC|ETH|SOL|ADA|DOT|XRP|LTC|BCH|BNB|DOGE|USDT|USDC)\b', 
        event.description, 
        re.IGNORECASE
    )
    
    if symbol_match:
        return symbol_match.group(1).upper()
        
    # If we can't find a specific symbol, return None
    return None


def get_crypto_system_prompt() -> str:
    """
    Get system prompt for crypto forecasting.
    
    Returns:
        System prompt
    """
    return """
You are an expert cryptocurrency analyst tasked with predicting price movements and market events.
Your goal is to provide an accurate probability estimate (from 0 to 1) for whether a crypto-related event will occur.

Guidelines:
- Analyze the historical price data and market sentiment carefully
- Consider technical indicators and market trends
- Be aware of upcoming protocol updates, regulatory news, and market events
- Express your prediction as a probability between 0 and 1
- Provide a brief explanation of your reasoning
- Be honest about uncertainty in volatile markets

Your response should be in the following JSON format:
{
    "probability": 0.X,
    "confidence": 0.X,
    "reasoning": "...",
    "technical_factors": ["..."],
    "fundamental_factors": ["..."]
}

Where:
- "probability" is your estimate of the likelihood (0-1) that the event will occur
- "confidence" is your certainty about your prediction (0-1)
- "reasoning" is a brief explanation of how you arrived at your prediction
- "technical_factors" lists technical analysis factors that influenced your prediction
- "fundamental_factors" lists fundamental factors that influenced your prediction
"""


class CryptoForecaster(BaseForecaster):
    """Forecaster specialized for cryptocurrency events."""
    
    def __init__(
        self,
        event: MinerEvent,
        use_llm: bool = True,
        llm_provider: str = "openai",
        llm_model: Optional[str] = None,
        use_market_data: bool = True,
        **kwargs,
    ):
        """
        Initialize crypto forecaster.
        
        Args:
            event: Event to forecast
            use_llm: Whether to use LLM for analysis
            llm_provider: LLM provider to use
            llm_model: LLM model to use (if None, use default)
            use_market_data: Whether to fetch market data
            **kwargs: Additional arguments to pass to parent
        """
        super().__init__(event, **kwargs)
        
        # Get config from miner.yaml
        crypto_config = miner_config.get("forecasters", {}).get("crypto", {})
        
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.llm_model = llm_model or crypto_config.get("model", "gpt-4")
        self.use_market_data = use_market_data
        
        # Create market data provider
        self.market_data = CryptoMarketData()
        
        # Create LLM client if using LLM
        if self.use_llm:
            self.llm_client = get_llm_client()
            
        # Extract crypto symbol from event
        self.symbol = extract_crypto_symbol(event)
        
        # Log initialization
        self.logger.info(
            f"Initialized crypto forecaster for {event.event_id} "
            f"(symbol: {self.symbol or 'unknown'})"
        )
    
    async def _get_price_data(self) -> Dict[str, Any]:
        """
        Get price data for analysis.
        
        Returns:
            Dictionary with price data
        """
        if not self.symbol or not self.use_market_data:
            return {"data_available": False}
            
        try:
            # Get price history
            history = await self.market_data.get_price_history(self.symbol, days=30)
            
            # Calculate some basic metrics
            if history:
                prices = [point["price"] for point in history]
                current_price = prices[-1]
                week_ago_price = prices[max(0, len(prices)-7)]
                month_ago_price = prices[0]
                
                week_change = (current_price - week_ago_price) / week_ago_price
                month_change = (current_price - month_ago_price) / month_ago_price
                
                # Calculate volatility (standard deviation of daily returns)
                daily_returns = [
                    (prices[i] - prices[i-1]) / prices[i-1] 
                    for i in range(1, len(prices))
                ]
                volatility = np.std(daily_returns) if daily_returns else 0
                
                return {
                    "data_available": True,
                    "current_price": current_price,
                    "week_change": week_change,
                    "month_change": month_change,
                    "volatility": volatility,
                    "history": history[-7:],  # Last week only to save space
                }
            
            return {"data_available": False}
            
        except Exception as e:
            self.logger.error(f"Error fetching price data: {e}", exc_info=True)
            return {"data_available": False, "error": str(e)}
    
    async def _get_sentiment_data(self) -> Dict[str, Any]:
        """
        Get sentiment data for analysis.
        
        Returns:
            Dictionary with sentiment data
        """
        if not self.symbol or not self.use_market_data:
            return {"data_available": False}
            
        try:
            # Get sentiment data
            sentiment = await self.market_data.get_market_sentiment(self.symbol)
            
            if sentiment:
                return {
                    "data_available": True,
                    **sentiment,
                }
            
            return {"data_available": False}
            
        except Exception as e:
            self.logger.error(f"Error fetching sentiment data: {e}", exc_info=True)
            return {"data_available": False, "error": str(e)}
    
    async def _run_technical_analysis(
        self, 
        price_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run technical analysis on price data.
        
        Args:
            price_data: Price data from _get_price_data
            
        Returns:
            Dictionary with technical analysis results
        """
        # In a real implementation, this would calculate technical indicators
        # For now, return mock analysis
        if not price_data.get("data_available", False):
            return {"signal": "neutral", "strength": 0.5}
            
        # Simple trend-based signal
        week_change = price_data.get("week_change", 0)
        month_change = price_data.get("month_change", 0)
        volatility = price_data.get("volatility", 0)
        
        # Calculate a basic signal based on recent price movement
        if week_change > 0.05 and month_change > 0.1:
            signal = "bullish"
            strength = min(0.7 + week_change, 0.9)
        elif week_change < -0.05 and month_change < -0.1:
            signal = "bearish"
            strength = min(0.7 + abs(week_change), 0.9)
        elif week_change > 0.02:
            signal = "slightly_bullish"
            strength = 0.6
        elif week_change < -0.02:
            signal = "slightly_bearish"
            strength = 0.6
        else:
            signal = "neutral"
            strength = 0.5
            
        # Adjust strength based on volatility
        if volatility > 0.05:
            strength = max(0.4, strength - 0.1)
            
        return {
            "signal": signal,
            "strength": strength,
            "week_change": week_change,
            "month_change": month_change,
            "volatility": volatility,
        }
    
    async def _run_llm_analysis(
        self,
        price_data: Dict[str, Any],
        sentiment_data: Dict[str, Any],
        technical_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run LLM analysis on all available data.
        
        Args:
            price_data: Price data
            sentiment_data: Sentiment data
            technical_analysis: Technical analysis results
            
        Returns:
            Dictionary with LLM analysis results
        """
        if not self.use_llm:
            return {"used": False}
            
        try:
            # Create prompt with all available data
            system_prompt = get_crypto_system_prompt()
            
            user_prompt = f"""
I need to forecast the following cryptocurrency event:

Description: {self.event.description}

Event Details:
- Event ID: {self.event.event_id}
- Symbol: {self.symbol or "Unknown"}
- Start date: {self.event.starts.strftime('%Y-%m-%d %H:%M:%S UTC')}
- End date: {self.event.end_date.strftime('%Y-%m-%d %H:%M:%S UTC')}
- Resolution date: {self.event.resolve_date.strftime('%Y-%m-%d %H:%M:%S UTC')}

"""

            # Add price data if available
            if price_data.get("data_available", False):
                user_prompt += f"""
Price Data:
- Current price: ${price_data['current_price']:.2f}
- 7-day change: {price_data['week_change']*100:.2f}%
- 30-day change: {price_data['month_change']*100:.2f}%
- Volatility (30-day): {price_data['volatility']:.4f}

Recent Price History:
"""
                for point in price_data.get("history", []):
                    date = datetime.fromisoformat(point["timestamp"]).strftime('%Y-%m-%d')
                    user_prompt += f"- {date}: ${point['price']:.2f} (Volume: {point['volume']:,})\n"
            
            # Add sentiment data if available
            if sentiment_data.get("data_available", False):
                user_prompt += f"""
Market Sentiment:
- Overall sentiment: {sentiment_data['sentiment_score']:.2f}
- Bullish signals: {sentiment_data['bullish_signals']}
- Bearish signals: {sentiment_data['bearish_signals']}
- Neutral signals: {sentiment_data['neutral_signals']}
- Social volume: {sentiment_data['social_volume']:,}
"""
            
            # Add technical analysis if available
            if technical_analysis:
                user_prompt += f"""
Technical Analysis:
- Signal: {technical_analysis['signal']}
- Signal strength: {technical_analysis['strength']:.2f}
"""
            
            # Add metadata if available
            if self.event.metadata:
                user_prompt += "\nAdditional Information:\n"
                for key, value in self.event.metadata.items():
                    if isinstance(value, (list, dict)):
                        user_prompt += f"- {key}: {json.dumps(value)}\n"
                    else:
                        user_prompt += f"- {key}: {value}\n"
            
            user_prompt += """
Based on all available information, provide a forecast for this cryptocurrency event in the requested JSON format.
Focus on being accurate, not overly confident. Cryptocurrency markets are volatile, so your confidence should reflect that.
"""
            
            # Call LLM
            self.logger.debug(f"Calling LLM for cryptocurrency analysis of {self.event.event_id}")
            response = await self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                provider=self.llm_provider,
                model=self.llm_model,
                temperature=0.1,  # Low temperature for more deterministic results
                max_tokens=1000,
            )
            
            # Parse response as JSON
            result = self._parse_llm_response(response.content)
            
            return {
                "used": True,
                "result": result,
                "tokens": response.usage.get("total_tokens", 0),
            }
            
        except Exception as e:
            self.logger.error(f"Error in LLM analysis: {e}", exc_info=True)
            return {"used": False, "error": str(e)}
    
    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """
        Parse LLM response.
        
        Args:
            content: LLM response content
            
        Returns:
            Parsed response
            
        Raises:
            ValueError: If response couldn't be parsed
        """
        # Try to extract JSON from response
        json_match = re.search(r'({[\s\S]*})', content)
        
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                
                # Validate result
                if "probability" not in result:
                    raise ValueError("Missing 'probability' in response")
                    
                # Ensure probability is a float between 0 and 1
                result["probability"] = float(result["probability"])
                if not 0 <= result["probability"] <= 1:
                    self.logger.warning(
                        f"Invalid probability value: {result['probability']}, clamping to [0,1]"
                    )
                    result["probability"] = max(0, min(1, result["probability"]))
                
                # Ensure confidence is a float between 0 and 1
                if "confidence" in result:
                    result["confidence"] = float(result["confidence"])
                    if not 0 <= result["confidence"] <= 1:
                        self.logger.warning(
                            f"Invalid confidence value: {result['confidence']}, clamping to [0,1]"
                        )
                        result["confidence"] = max(0, min(1, result["confidence"]))
                else:
                    # Default confidence
                    result["confidence"] = 0.7
                
                # Ensure reasoning is present
                if "reasoning" not in result:
                    result["reasoning"] = "No reasoning provided"
                
                return result
                
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in response: {e}")
        
        # Fallback: try to extract probability directly
        probability_match = re.search(r'probability[:\s]*([0-9.]+)', content, re.IGNORECASE)
        if probability_match:
            try:
                probability = float(probability_match.group(1))
                
                # Extract confidence if present
                confidence_match = re.search(r'confidence[:\s]*([0-9.]+)', content, re.IGNORECASE)
                confidence = 0.7  # Default
                if confidence_match:
                    confidence = float(confidence_match.group(1))
                
                return {
                    "probability": max(0, min(1, probability)),
                    "confidence": max(0, min(1, confidence)),
                    "reasoning": content.strip(),
                }
                
            except ValueError:
                pass
        
        # If all else fails
        raise ValueError("Could not extract probability from LLM response")
    
    async def _run(self) -> Prediction:
        """
        Generate prediction for cryptocurrency event.
        
        Returns:
            Prediction object
        """
        # If no symbol found and LLM is enabled, fall back to LLM forecaster
        if not self.symbol and self.use_llm:
            self.logger.info(
                f"No crypto symbol found for {self.event.event_id}, "
                "falling back to LLM forecaster"
            )
            llm_forecaster = LLMForecaster(
                event=self.event,
                llm_provider=self.llm_provider,
                llm_model=self.llm_model,
            )
            return await llm_forecaster._run()
        
        # Get market data
        price_data = await self._get_price_data()
        sentiment_data = await self._get_sentiment_data()
        
        # Run technical analysis
        technical_analysis = await self._run_technical_analysis(price_data)
        
        # If using LLM, run LLM analysis
        llm_result = {}
        if self.use_llm:
            llm_result = await self._run_llm_analysis(
                price_data, 
                sentiment_data, 
                technical_analysis
            )
        
        # Determine final prediction
        if llm_result.get("used", False) and "result" in llm_result:
            # Use LLM result if available
            result = llm_result["result"]
            
            return Prediction(
                event_id=self.event.event_id,
                probability=result["probability"],
                confidence=result["confidence"],
                metadata={
                    "model": self.llm_model,
                    "provider": self.llm_provider,
                    "reasoning": result.get("reasoning", ""),
                    "technical_factors": result.get("technical_factors", []),
                    "fundamental_factors": result.get("fundamental_factors", []),
                    "price_data_available": price_data.get("data_available", False),
                    "sentiment_data_available": sentiment_data.get("data_available", False),
                    "tokens": llm_result.get("tokens", 0),
                },
            )
        else:
            # Use technical analysis for prediction
            signal = technical_analysis.get("signal", "neutral")
            strength = technical_analysis.get("strength", 0.5)
            
            # Map signal to probability
            if signal == "bullish":
                probability = 0.75
            elif signal == "slightly_bullish":
                probability = 0.65
            elif signal == "bearish":
                probability = 0.25
            elif signal == "slightly_bearish":
                probability = 0.35
            else:  # neutral
                probability = 0.5
                
            # Calculate confidence based on data availability
            confidence = 0.6  # Base confidence
            if not price_data.get("data_available", False):
                confidence -= 0.2
            if not sentiment_data.get("data_available", False):
                confidence -= 0.1
                
            return Prediction(
                event_id=self.event.event_id,
                probability=probability,
                confidence=max(0.1, confidence),
                metadata={
                    "technical_signal": signal,
                    "signal_strength": strength,
                    "week_change": technical_analysis.get("week_change"),
                    "month_change": technical_analysis.get("month_change"),
                    "volatility": technical_analysis.get("volatility"),
                    "price_data_available": price_data.get("data_available", False),
                    "sentiment_data_available": sentiment_data.get("data_available", False),
                    "llm_used": False,
                },
            ) 