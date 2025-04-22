"""
Utility functions for fetching and analyzing market data for forecasters.
"""

import asyncio
from typing import Dict, List, Optional, Union
import numpy as np
from datetime import datetime, timedelta

import aiohttp
from pycoingecko import CoinGeckoAPI

from infinite_forecast.api.core.logging import get_logger

logger = get_logger(__name__)

# Initialize API client
coingecko = CoinGeckoAPI()


async def get_crypto_historical_data(coin_id: str, days: int = 30) -> List[Dict]:
    """Fetch historical price data for a cryptocurrency.
    
    Args:
        coin_id: CoinGecko ID of the cryptocurrency
        days: Number of days of historical data
        
    Returns:
        List of dictionaries with price data
    """
    try:
        # Attempt to get data from CoinGecko
        market_data = coingecko.get_coin_market_chart_by_id(
            id=coin_id,
            vs_currency="usd",
            days=days
        )
        
        # Process and structure the data
        prices = market_data.get("prices", [])
        volumes = market_data.get("total_volumes", [])
        
        result = []
        for i, (timestamp, price) in enumerate(prices):
            volume = volumes[i][1] if i < len(volumes) else 0
            dt = datetime.fromtimestamp(timestamp/1000)
            
            result.append({
                "date": dt.strftime("%Y-%m-%d"),
                "timestamp": timestamp,
                "price": price,
                "volume": volume
            })
        
        return result
    
    except Exception as e:
        logger.error(f"Error fetching historical data for {coin_id}: {str(e)}")
        # Return empty list on error
        return []


def get_price_trend(historical_data: List[Dict], window: int = 7) -> str:
    """Analyze price trend from historical data.
    
    Args:
        historical_data: List of price data dictionaries
        window: Number of days to analyze trend
        
    Returns:
        String indicating trend: "strong_up", "up", "neutral", "down", "strong_down"
    """
    if not historical_data or len(historical_data) < window:
        return "neutral"
    
    # Extract prices for analysis
    prices = [entry["price"] for entry in historical_data[-window:]]
    
    # Calculate simple metrics
    start_price = prices[0]
    end_price = prices[-1]
    price_change = (end_price - start_price) / start_price if start_price > 0 else 0
    
    # Calculate moving averages
    if len(prices) >= 5:
        ma5 = sum(prices[-5:]) / 5
    else:
        ma5 = end_price
        
    if len(prices) >= 10:
        ma10 = sum(prices[-10:]) / 10
    else:
        ma10 = end_price
    
    # Determine trend based on price change and moving averages
    if price_change > 0.1 and end_price > ma5 > ma10:
        return "strong_up"
    elif price_change > 0.02 and end_price > ma5:
        return "up"
    elif -0.02 <= price_change <= 0.02:
        return "neutral"
    elif price_change < -0.02 and end_price < ma5:
        return "down"
    elif price_change < -0.1 and end_price < ma5 < ma10:
        return "strong_down"
    else:
        return "neutral"


async def get_market_sentiment(coin_id: str) -> Dict[str, float]:
    """Get market sentiment indicators for a cryptocurrency.
    
    Args:
        coin_id: CoinGecko ID of the cryptocurrency
        
    Returns:
        Dictionary with sentiment scores
    """
    try:
        # Get community data and developer activity as sentiment indicators
        coin_data = coingecko.get_coin_by_id(
            id=coin_id,
            localization=False,
            tickers=False,
            market_data=True,
            community_data=True,
            developer_data=True
        )
        
        # Extract relevant metrics
        community = coin_data.get("community_data", {})
        dev = coin_data.get("developer_data", {})
        market = coin_data.get("market_data", {})
        
        # Normalize sentiment metrics
        twitter_followers = community.get("twitter_followers", 0)
        reddit_subscribers = community.get("reddit_subscribers", 0)
        github_commits = dev.get("commit_count_4_weeks", 0)
        
        # Market metrics
        market_cap_change = market.get("market_cap_change_percentage_24h", 0)
        price_change = market.get("price_change_percentage_24h", 0)
        
        # Simple sentiment score (0-1 scale)
        sentiment_score = min(1.0, max(0.0, 0.5 + (price_change / 100)))
        
        # Simplified activity score (0-1 scale)
        # Higher github activity is positive
        activity_score = min(1.0, github_commits / 30) if github_commits else 0.5
        
        return {
            "sentiment": sentiment_score,
            "activity": activity_score,
            "price_change_24h": price_change,
            "market_cap_change_24h": market_cap_change
        }
        
    except Exception as e:
        logger.error(f"Error fetching sentiment for {coin_id}: {str(e)}")
        # Return neutral values on error
        return {
            "sentiment": 0.5,
            "activity": 0.5,
            "price_change_24h": 0,
            "market_cap_change_24h": 0
        }


def calculate_technical_indicators(historical_data: List[Dict]) -> Dict[str, float]:
    """Calculate technical indicators from historical price data.
    
    Args:
        historical_data: List of price data dictionaries
        
    Returns:
        Dictionary with technical indicators
    """
    if not historical_data or len(historical_data) < 14:
        return {
            "rsi": 50,
            "macd": 0,
            "volatility": 0.02
        }
    
    # Extract price data
    prices = np.array([entry["price"] for entry in historical_data])
    
    # Calculate RSI (Relative Strength Index)
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Average gain and loss over 14 periods
    avg_gain = np.mean(gain[-14:])
    avg_loss = np.mean(loss[-14:])
    
    if avg_loss == 0:
        rsi = 100
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    
    # Calculate MACD (Moving Average Convergence Divergence)
    if len(prices) >= 26:
        ema12 = np.mean(prices[-12:])
        ema26 = np.mean(prices[-26:])
        macd = ema12 - ema26
    else:
        macd = 0
    
    # Calculate volatility (standard deviation of returns)
    returns = delta / prices[:-1]
    volatility = np.std(returns) if len(returns) > 0 else 0.02
    
    return {
        "rsi": rsi,
        "macd": macd,
        "volatility": volatility
    } 