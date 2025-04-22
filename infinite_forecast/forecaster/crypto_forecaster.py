"""
Cryptocurrency price forecaster for the Infinite Games platform.

This module provides forecasting capabilities for cryptocurrency price events, using
historical data and market metrics to make predictions about future prices.
"""

import asyncio
import datetime
import json
import logging
import math
import os
import sqlite3
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
from dateutil.parser import parse as parse_date

from infinite_forecast.api.client.infinite_games import InfiniteGamesClient
from infinite_forecast.api.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CryptoEventData:
    """Data class for cryptocurrency event information."""
    
    event_id: str
    symbol: str
    current_price: Optional[float]
    target_date: datetime.datetime
    options: Optional[List[str]] = None
    
    @classmethod
    def from_event_dict(cls, event_dict: Dict[str, Any]) -> "CryptoEventData":
        """Create a CryptoEventData object from an event dictionary.
        
        Args:
            event_dict: Dictionary containing event information
            
        Returns:
            CryptoEventData object
            
        Raises:
            ValueError: If required data is missing or invalid
        """
        try:
            # Extract the event ID
            event_id = event_dict.get("id") or event_dict.get("event_id")
            if not event_id:
                raise ValueError("Event dictionary missing event ID")
            
            # Extract relevant data from the event
            question = event_dict.get("question", "")
            description = event_dict.get("description", "")
            
            # Extract cryptocurrency symbol from question or description
            symbol = cls._extract_symbol(question, description)
            if not symbol:
                raise ValueError(f"Could not extract cryptocurrency symbol from event: {event_id}")
            
            # Extract current price if available
            current_price = cls._extract_current_price(question, description)
            
            # Extract target date
            close_time = event_dict.get("close_time")
            if close_time:
                target_date = parse_date(close_time)
            else:
                # If no close time, use event deadline or default to 24 hours from now
                deadline = event_dict.get("deadline")
                if deadline:
                    target_date = parse_date(deadline)
                else:
                    target_date = datetime.datetime.now() + datetime.timedelta(days=1)
            
            # Extract options for multiple choice questions
            options = event_dict.get("options", [])
            
            return cls(
                event_id=event_id,
                symbol=symbol.upper(),
                current_price=current_price,
                target_date=target_date,
                options=options if options else None
            )
            
        except Exception as e:
            logger.error(f"Error creating CryptoEventData from event: {str(e)}")
            raise ValueError(f"Failed to parse crypto event: {str(e)}")
    
    @staticmethod
    def _extract_symbol(question: str, description: str) -> Optional[str]:
        """Extract cryptocurrency symbol from question or description.
        
        Args:
            question: Event question text
            description: Event description text
            
        Returns:
            Cryptocurrency symbol or None if not found
        """
        # Common pattern: "What will be the price of {SYMBOL} on..."
        text = f"{question} {description}".upper()
        
        # Check for common cryptocurrencies
        crypto_symbols = ["BTC", "ETH", "ADA", "SOL", "DOT", "AVAX", "MATIC", 
                         "LINK", "XRP", "LTC", "BCH", "DOGE", "SHIB", "UNI"]
        
        for symbol in crypto_symbols:
            if symbol in text:
                return symbol
                
        # Look for patterns like "$BTC" or "[BTC]"
        import re
        matches = re.findall(r'[$]([A-Z]{3,5})|[[]([A-Z]{3,5})[]]', text)
        if matches:
            # Flatten matches and filter empty matches
            flat_matches = [m for sublist in matches for m in sublist if m]
            if flat_matches:
                return flat_matches[0]
                
        return None
    
    @staticmethod
    def _extract_current_price(question: str, description: str) -> Optional[float]:
        """Extract current cryptocurrency price from question or description.
        
        Args:
            question: Event question text
            description: Event description text
            
        Returns:
            Current price or None if not found
        """
        text = f"{question} {description}"
        
        # Look for patterns like "currently at $45,000" or "current price: $45,000"
        import re
        
        # Pattern for prices with commas and decimals
        matches = re.findall(r'currently at [\$€£]([0-9,]+\.?[0-9]*)|current price:? [\$€£]([0-9,]+\.?[0-9]*)', 
                            text, re.IGNORECASE)
        
        if matches:
            # Flatten matches and filter empty matches
            flat_matches = [m for sublist in matches for m in sublist if m]
            if flat_matches:
                # Remove commas and convert to float
                try:
                    return float(flat_matches[0].replace(',', ''))
                except ValueError:
                    return None
        
        return None


class CryptoForecaster:
    """Forecaster for cryptocurrency price events.
    
    This class provides methods for analyzing historical price data,
    calculating volatility, and making predictions about future prices.
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        cache_duration: int = 24 * 60 * 60,  # 24 hours in seconds
        api_key: Optional[str] = None
    ):
        """Initialize the cryptocurrency forecaster.
        
        Args:
            db_path: Path to SQLite database for caching price data
            cache_duration: How long to cache price data in seconds
            api_key: CoinGecko API key
        """
        self.api_key = api_key or os.getenv("COINGECKO_API_KEY", "")
        self.cache_duration = cache_duration
        
        # Default database path in user's home directory
        if db_path is None:
            home_dir = os.path.expanduser("~")
            if not os.path.exists(os.path.join(home_dir, ".infinite_forecast")):
                os.makedirs(os.path.join(home_dir, ".infinite_forecast"), exist_ok=True)
            self.db_path = os.path.join(home_dir, ".infinite_forecast", "crypto_cache.db")
        else:
            self.db_path = db_path
            
        # Initialize database
        self._setup_db()
        
        # Initialize Infinite Games client
        self.infinite_client = InfiniteGamesClient()
    
    def _setup_db(self):
        """Set up SQLite database for caching price data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Table for historical price data
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                symbol TEXT,
                date TEXT,
                price REAL,
                last_updated INTEGER,
                PRIMARY KEY (symbol, date)
            )
            ''')
            
            # Table for volatility data
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS volatility_data (
                symbol TEXT PRIMARY KEY,
                volatility REAL,
                last_updated INTEGER
            )
            ''')
            
            # Table for predictions
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                event_id TEXT PRIMARY KEY,
                symbol TEXT,
                prediction REAL,
                confidence REAL,
                target_date TEXT,
                reasoning TEXT,
                created_at INTEGER
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"Database initialized at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {str(e)}")
    
    async def get_historical_prices(
        self,
        symbol: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get historical price data for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "BTC")
            days: Number of days of historical data to fetch
            
        Returns:
            List of dictionaries with date and price information
        """
        current_time = int(datetime.datetime.now().timestamp())
        cache_expiry = current_time - self.cache_duration
        
        # Check cache first
        prices_from_cache = self._get_cached_prices(symbol, days, cache_expiry)
        if prices_from_cache:
            logger.info(f"Using cached price data for {symbol}")
            return prices_from_cache
        
        # Fetch from API if cache miss or expired
        logger.info(f"Fetching historical prices for {symbol} for past {days} days")
        
        # Normalize symbol for API call
        api_symbol = symbol.lower()
        if api_symbol == "btc":
            api_symbol = "bitcoin"
        elif api_symbol == "eth":
            api_symbol = "ethereum"
        
        # CoinGecko API endpoint
        url = f"https://api.coingecko.com/api/v3/coins/{api_symbol}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": days,
            "interval": "daily"
        }
        
        headers = {}
        if self.api_key:
            headers["x-cg-pro-api-key"] = self.api_key
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 429:
                        logger.warning("Rate limit exceeded for CoinGecko API")
                        # Try to use cache even if expired
                        fallback_prices = self._get_cached_prices(symbol, days, None)
                        if fallback_prices:
                            logger.info(f"Using expired cache for {symbol} due to rate limiting")
                            return fallback_prices
                        return []
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"API error ({response.status}): {error_text}")
                        return []
                    
                    data = await response.json()
                    
                    # Process price data
                    prices = data.get("prices", [])
                    result = []
                    
                    for timestamp, price in prices:
                        date = datetime.datetime.fromtimestamp(timestamp / 1000)
                        date_str = date.strftime("%Y-%m-%d")
                        
                        # Cache this price point
                        self._cache_price(symbol, date_str, price, current_time)
                        
                        result.append({
                            "date": date_str,
                            "price": price
                        })
                    
                    return result
                    
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching historical prices: {str(e)}")
            return []
    
    def _get_cached_prices(
        self,
        symbol: str,
        days: int,
        cache_expiry: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Get cached price data from database.
        
        Args:
            symbol: Cryptocurrency symbol
            days: Number of days of historical data
            cache_expiry: Timestamp for cache expiration, or None to ignore
            
        Returns:
            List of price data dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate date range
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=days)
            
            query = '''
            SELECT date, price FROM price_data
            WHERE symbol = ? AND date >= ?
            '''
            params = [symbol, start_date.strftime("%Y-%m-%d")]
            
            if cache_expiry is not None:
                query += " AND last_updated >= ?"
                params.append(cache_expiry)
            
            query += " ORDER BY date"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to list of dictionaries
            result = [{"date": date, "price": price} for date, price in rows]
            
            # Only return if we have enough data
            if len(result) >= days * 0.8:  # At least 80% of days have data
                return result
            return []
            
        except sqlite3.Error as e:
            logger.error(f"Error retrieving cached prices: {str(e)}")
            return []
    
    def _cache_price(
        self,
        symbol: str,
        date: str,
        price: float,
        timestamp: int
    ):
        """Cache price data in the database.
        
        Args:
            symbol: Cryptocurrency symbol
            date: Date string in YYYY-MM-DD format
            price: Price value
            timestamp: Current timestamp
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                '''
                INSERT OR REPLACE INTO price_data (symbol, date, price, last_updated)
                VALUES (?, ?, ?, ?)
                ''',
                (symbol, date, price, timestamp)
            )
            
            conn.commit()
            conn.close()
            
        except sqlite3.Error as e:
            logger.error(f"Error caching price data: {str(e)}")
    
    async def calculate_volatility(
        self,
        symbol: str,
        days: int = 30,
        force_refresh: bool = False
    ) -> float:
        """Calculate historical volatility for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol
            days: Number of days to use for calculation
            force_refresh: Whether to force recalculation
            
        Returns:
            Annualized volatility as a percentage
        """
        current_time = int(datetime.datetime.now().timestamp())
        cache_expiry = current_time - self.cache_duration
        
        # Check cache first if not forcing refresh
        if not force_refresh:
            volatility = self._get_cached_volatility(symbol, cache_expiry)
            if volatility is not None:
                return volatility
        
        # Get historical price data
        prices = await self.get_historical_prices(symbol, days)
        if not prices:
            logger.warning(f"No price data available for {symbol}")
            return 0.0
            
        # Calculate daily returns
        price_values = [p["price"] for p in prices]
        returns = []
        
        for i in range(1, len(price_values)):
            daily_return = math.log(price_values[i] / price_values[i-1])
            returns.append(daily_return)
            
        if not returns:
            logger.warning(f"Not enough price data to calculate volatility for {symbol}")
            return 0.0
            
        # Calculate standard deviation of returns
        std_dev = np.std(returns)
        
        # Annualize volatility (multiply by sqrt of trading days in a year)
        annualized_volatility = std_dev * math.sqrt(365) * 100  # as percentage
        
        # Cache the result
        self._cache_volatility(symbol, annualized_volatility, current_time)
        
        return annualized_volatility
    
    def _get_cached_volatility(self, symbol: str, cache_expiry: int) -> Optional[float]:
        """Get cached volatility from database.
        
        Args:
            symbol: Cryptocurrency symbol
            cache_expiry: Timestamp for cache expiration
            
        Returns:
            Cached volatility or None if not found or expired
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                '''
                SELECT volatility FROM volatility_data
                WHERE symbol = ? AND last_updated >= ?
                ''',
                (symbol, cache_expiry)
            )
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return row[0]
            return None
            
        except sqlite3.Error as e:
            logger.error(f"Error retrieving cached volatility: {str(e)}")
            return None
    
    def _cache_volatility(self, symbol: str, volatility: float, timestamp: int):
        """Cache volatility in the database.
        
        Args:
            symbol: Cryptocurrency symbol
            volatility: Calculated volatility value
            timestamp: Current timestamp
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                '''
                INSERT OR REPLACE INTO volatility_data (symbol, volatility, last_updated)
                VALUES (?, ?, ?)
                ''',
                (symbol, volatility, timestamp)
            )
            
            conn.commit()
            conn.close()
            
        except sqlite3.Error as e:
            logger.error(f"Error caching volatility data: {str(e)}")
    
    async def predict_price(
        self,
        event_data: CryptoEventData
    ) -> Tuple[Union[float, str], float, str]:
        """Predict future price of a cryptocurrency.
        
        Args:
            event_data: Data about the crypto event
            
        Returns:
            Tuple of (prediction, confidence, reasoning)
            where prediction can be a float price or string option
        """
        symbol = event_data.symbol
        target_date = event_data.target_date
        
        # Get historical price data and current price
        prices = await self.get_historical_prices(symbol, days=30)
        if not prices:
            logger.error(f"Could not retrieve historical prices for {symbol}")
            return 0.0, 0.3, "Insufficient historical data"
        
        # Get current price
        current_price = event_data.current_price
        if current_price is None:
            current_price = prices[-1]["price"]
        
        # Calculate days until target
        days_until_target = (target_date - datetime.datetime.now()).days
        
        # Calculate volatility
        volatility = await self.calculate_volatility(symbol)
        
        # Calculate mean daily return
        price_values = [p["price"] for p in prices]
        daily_returns = []
        
        for i in range(1, len(price_values)):
            daily_return = (price_values[i] / price_values[i-1]) - 1
            daily_returns.append(daily_return)
        
        mean_daily_return = np.mean(daily_returns) if daily_returns else 0
        
        # Project future price
        projected_price = current_price * (1 + mean_daily_return) ** days_until_target
        
        # Calculate confidence based on volatility and time to target
        # Higher volatility and longer time horizons reduce confidence
        base_confidence = 0.8  # Start with base confidence
        volatility_factor = volatility / 100  # Convert to decimal
        time_factor = days_until_target / 365  # Normalize by year
        
        confidence = base_confidence - (volatility_factor * time_factor)
        confidence = max(0.2, min(0.95, confidence))  # Bound between 0.2 and 0.95
        
        # Prepare reasoning
        reasoning = f"Based on historical analysis of {symbol} price movement over the past 30 days, "
        reasoning += f"with a mean daily return of {mean_daily_return:.4%} and volatility of {volatility:.2f}%, "
        reasoning += f"the projected price on {target_date.strftime('%Y-%m-%d')} is ${projected_price:.2f}. "
        
        # Handle multiple choice options if provided
        if event_data.options:
            prediction_option, option_confidence = self._map_price_to_option(
                projected_price, event_data.options
            )
            # Adjust confidence based on option mapping
            confidence = min(confidence, option_confidence)
            return prediction_option, confidence, reasoning
        
        return projected_price, confidence, reasoning
    
    def _map_price_to_option(
        self,
        projected_price: float,
        options: List[str]
    ) -> Tuple[str, float]:
        """Map a projected price to one of the provided options.
        
        Args:
            projected_price: The projected price value
            options: List of option strings
            
        Returns:
            Tuple of (selected option, confidence)
        """
        # First, see if any options are explicit price ranges
        price_ranges = []
        for option in options:
            # Try to parse options as price ranges (e.g., "$40,000-$45,000")
            import re
            match = re.search(r'[\$€£]([0-9,.]+)[-—–][\$€£]?([0-9,.]+)', option)
            if match:
                try:
                    lower = float(match.group(1).replace(',', ''))
                    upper = float(match.group(2).replace(',', ''))
                    price_ranges.append((lower, upper, option))
                except ValueError:
                    continue
        
        # If we found price ranges, select the appropriate one
        if price_ranges:
            for lower, upper, option in price_ranges:
                if lower <= projected_price <= upper:
                    # High confidence if price is in middle of range, lower near boundaries
                    position_in_range = (projected_price - lower) / (upper - lower)
                    # Highest confidence when position is 0.5 (middle of range)
                    range_confidence = 1.0 - abs(0.5 - position_in_range) * 0.4
                    return option, range_confidence
            
            # If projected price is outside all ranges, pick closest range
            closest_option = min(
                price_ranges,
                key=lambda x: min(abs(projected_price - x[0]), abs(projected_price - x[1]))
            )[2]
            return closest_option, 0.4  # Lower confidence for out-of-range prediction
        
        # If options don't appear to be price ranges, use keyword matching
        direction_options = {
            "up": ["up", "increase", "higher", "rise", "bull", "positive"],
            "down": ["down", "decrease", "lower", "fall", "drop", "bear", "negative"],
            "same": ["same", "unchanged", "stable", "flat", "neutral"]
        }
        
        # Get the last two prices to determine trend
        price_trend = 0.0  # Default to flat
        
        # Determine which option best matches our prediction
        if price_trend > 0.02:  # More than 2% increase
            direction = "up"
        elif price_trend < -0.02:  # More than 2% decrease
            direction = "down"
        else:
            direction = "same"
            
        # Find matching option for our direction
        for option in options:
            option_lower = option.lower()
            if any(keyword in option_lower for keyword in direction_options[direction]):
                return option, 0.6
        
        # If no clear match, return the option that seems most neutral
        for option in options:
            option_lower = option.lower()
            if any(keyword in option_lower for keyword in direction_options["same"]):
                return option, 0.5
                
        # If still no match, default to first option with low confidence
        return options[0], 0.4
    
    def format_prediction_for_submission(
        self,
        prediction: Union[float, str],
        confidence: float,
        options: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Format a prediction for submission to the Infinite Games API.
        
        Args:
            prediction: Numeric price prediction or string option
            confidence: Confidence level
            options: Optional list of available options
            
        Returns:
            Formatted prediction data
        """
        # If prediction is a string option and options are provided
        if isinstance(prediction, str) and options:
            # Make sure the prediction is exactly one of the options
            if prediction not in options:
                closest_match = min(options, key=lambda x: 
                                   sum(1 for a, b in zip(x.lower(), prediction.lower()) if a != b))
                prediction = closest_match
            
            return {
                "prediction": prediction,
                "confidence": confidence
            }
            
        # If prediction is a numeric value, format it properly
        if isinstance(prediction, (int, float, Decimal)):
            # Round to 2 decimal places for most cryptocurrencies
            formatted_value = float(round(Decimal(str(prediction)), 2))
            
            return {
                "prediction": formatted_value,
                "confidence": confidence
            }
        
        # Default case
        return {
            "prediction": prediction,
            "confidence": confidence
        }
    
    def _cache_prediction(
        self,
        event_id: str,
        symbol: str,
        prediction: Union[float, str],
        confidence: float,
        target_date: datetime.datetime,
        reasoning: str
    ):
        """Cache a prediction in the database.
        
        Args:
            event_id: ID of the event
            symbol: Cryptocurrency symbol
            prediction: Prediction value
            confidence: Confidence level
            target_date: Target date for the prediction
            reasoning: Reasoning behind the prediction
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convert prediction to string if it's not a number
            if isinstance(prediction, (int, float, Decimal)):
                prediction_value = float(prediction)
            else:
                prediction_value = None
                
            current_time = int(datetime.datetime.now().timestamp())
            
            cursor.execute(
                '''
                INSERT OR REPLACE INTO predictions
                (event_id, symbol, prediction, confidence, target_date, reasoning, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    event_id,
                    symbol,
                    prediction_value,
                    confidence,
                    target_date.isoformat(),
                    reasoning,
                    current_time
                )
            )
            
            conn.commit()
            conn.close()
            
        except sqlite3.Error as e:
            logger.error(f"Error caching prediction: {str(e)}")

    async def process_event(self, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process a cryptocurrency event and make a prediction.
        
        Args:
            event_dict: Dictionary containing event information
            
        Returns:
            Dictionary with prediction information
            
        Raises:
            ValueError: If the event cannot be processed
        """
        try:
            # Parse event data
            event_data = CryptoEventData.from_event_dict(event_dict)
            
            # Make prediction
            prediction, confidence, reasoning = await self.predict_price(event_data)
            
            # Format prediction for submission
            submission_data = self.format_prediction_for_submission(
                prediction, 
                confidence,
                event_data.options
            )
            
            # Add reasoning and metadata
            submission_data["reasoning"] = reasoning
            submission_data["metadata"] = {
                "symbol": event_data.symbol,
                "prediction_time": datetime.datetime.now().isoformat(),
                "target_date": event_data.target_date.isoformat(),
                "model": "crypto_forecaster_v1"
            }
            
            # Cache prediction
            self._cache_prediction(
                event_data.event_id,
                event_data.symbol,
                prediction,
                confidence,
                event_data.target_date,
                reasoning
            )
            
            return submission_data
            
        except Exception as e:
            logger.error(f"Error processing crypto event: {str(e)}")
            raise ValueError(f"Failed to process crypto event: {str(e)}")

    async def fetch_and_process_crypto_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch and process open cryptocurrency events.
        
        Args:
            limit: Maximum number of events to process
            
        Returns:
            List of processed events with predictions
        """
        results = []
        
        try:
            # Fetch open crypto events
            events = await self.infinite_client.get_crypto_events(limit=limit)
            
            # Process each event
            for event in events:
                try:
                    prediction_data = await self.process_event(event)
                    
                    # Submit prediction to Infinite Games
                    event_id = event.get("id", "")
                    submission_result = await self.infinite_client.submit_prediction(
                        event_id=event_id,
                        prediction=prediction_data["prediction"],
                        confidence=prediction_data["confidence"],
                        reasoning=prediction_data["reasoning"],
                        metadata=prediction_data["metadata"]
                    )
                    
                    results.append({
                        "event_id": event_id,
                        "prediction": prediction_data,
                        "submission_result": submission_result
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing event {event.get('id', '')}: {str(e)}")
                    results.append({
                        "event_id": event.get("id", ""),
                        "error": str(e)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error fetching and processing crypto events: {str(e)}")
            return []

    async def evaluate_historical_performance(self, days: int = 30) -> Dict[str, Any]:
        """Evaluate the performance of historical predictions.
        
        Args:
            days: Number of days of history to evaluate
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get predictions from the specified time period
            cutoff_time = int((datetime.datetime.now() - datetime.timedelta(days=days)).timestamp())
            
            cursor.execute(
                '''
                SELECT event_id, symbol, prediction, confidence, target_date
                FROM predictions
                WHERE created_at >= ?
                ''',
                (cutoff_time,)
            )
            
            predictions = cursor.fetchall()
            conn.close()
            
            if not predictions:
                return {"error": "No historical predictions found"}
            
            # Get actual outcomes for these predictions
            results = []
            for event_id, symbol, prediction, confidence, target_date in predictions:
                try:
                    # Get event details from API
                    event = await self.infinite_client.get_event(event_id)
                    
                    # If event is resolved, compare prediction to outcome
                    if event.get("status") == "resolved":
                        outcome = event.get("outcome")
                        if outcome is not None:
                            # Calculate accuracy
                            if isinstance(prediction, (int, float)) and isinstance(outcome, (int, float)):
                                # For numeric predictions, calculate percentage error
                                error = abs(prediction - outcome) / outcome
                                accuracy = max(0, 1 - error)
                            elif isinstance(prediction, str) and isinstance(outcome, str):
                                # For categorical predictions, binary accuracy
                                accuracy = 1.0 if prediction == outcome else 0.0
                            else:
                                accuracy = None
                                
                            results.append({
                                "event_id": event_id,
                                "symbol": symbol,
                                "prediction": prediction,
                                "confidence": confidence,
                                "outcome": outcome,
                                "accuracy": accuracy
                            })
                except Exception as e:
                    logger.error(f"Error evaluating prediction for event {event_id}: {str(e)}")
            
            # Calculate performance metrics
            if not results:
                return {"error": "No resolved predictions found"}
                
            # Filter results with accuracy values
            valid_results = [r for r in results if r["accuracy"] is not None]
            
            if not valid_results:
                return {"error": "No predictions with calculable accuracy found"}
                
            # Calculate average accuracy
            avg_accuracy = sum(r["accuracy"] for r in valid_results) / len(valid_results)
            
            # Calculate calibration (correlation between confidence and accuracy)
            if len(valid_results) > 1:
                confidences = [r["confidence"] for r in valid_results]
                accuracies = [r["accuracy"] for r in valid_results]
                
                from scipy.stats import pearsonr
                calibration_corr, p_value = pearsonr(confidences, accuracies)
                
                calibration = {
                    "correlation": calibration_corr,
                    "p_value": p_value,
                    "significant": p_value < 0.05
                }
            else:
                calibration = {"error": "Need more than one prediction for calibration"}
            
            return {
                "num_predictions": len(valid_results),
                "average_accuracy": avg_accuracy,
                "calibration": calibration,
                "details": valid_results
            }
            
        except Exception as e:
            logger.error(f"Error evaluating historical performance: {str(e)}")
            return {"error": str(e)}


async def main():
    """Run the crypto forecaster."""
    forecaster = CryptoForecaster()
    results = await forecaster.fetch_and_process_crypto_events()
    print(json.dumps(results, indent=2))
    
    # Evaluate performance
    performance = await forecaster.evaluate_historical_performance()
    print("\nPerformance Metrics:")
    print(json.dumps(performance, indent=2))


if __name__ == "__main__":
    asyncio.run(main()) 