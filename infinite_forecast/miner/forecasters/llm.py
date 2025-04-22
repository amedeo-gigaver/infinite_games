"""
LLM-based forecaster for predicting events.
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

from infinite_forecast.api.core.config import get_miner_config
from infinite_forecast.api.core.logging import get_logger
from infinite_forecast.miner.forecasters.base import BaseForecaster, MinerEvent, Prediction
from infinite_forecast.utils.llm.client import get_llm_client

# Initialize logger and config
logger = get_logger(__name__)
miner_config = get_miner_config()


def get_system_prompt() -> str:
    """
    Get system prompt for LLM.
    
    Returns:
        System prompt
    """
    return """
You are an expert forecaster tasked with predicting the likelihood of future events.
Your goal is to provide an accurate probability estimate (from 0 to 1) for whether an event will occur.

Guidelines:
- Analyze the event description carefully
- Consider relevant context and historical data
- Be objective and avoid any personal bias
- Express your prediction as a probability between 0 and 1
- Provide a brief explanation of your reasoning
- Be honest about uncertainty when information is limited

Your response should be in the following JSON format:
{
    "probability": 0.X,
    "confidence": 0.X,
    "reasoning": "...",
    "sources": ["..."]
}

Where:
- "probability" is your estimate of the likelihood (0-1) that the event will occur
- "confidence" is your certainty about your prediction (0-1)
- "reasoning" is a brief explanation of how you arrived at your prediction
- "sources" lists any specific information sources you're drawing from
"""


def get_user_prompt(event: MinerEvent) -> str:
    """
    Get user prompt for LLM based on event.
    
    Args:
        event: Event to forecast
        
    Returns:
        User prompt
    """
    prompt = f"""
I need to forecast the following event:

Description: {event.description}

Event Details:
- Event ID: {event.event_id}
- Type: {event.event_type.value}
- Start date: {event.starts.strftime('%Y-%m-%d %H:%M:%S UTC')}
- End date: {event.end_date.strftime('%Y-%m-%d %H:%M:%S UTC')}
- Resolution date: {event.resolve_date.strftime('%Y-%m-%d %H:%M:%S UTC')}
"""

    # Add metadata if available
    if event.metadata:
        prompt += "\nAdditional Information:\n"
        for key, value in event.metadata.items():
            if isinstance(value, (list, dict)):
                prompt += f"- {key}: {json.dumps(value)}\n"
            else:
                prompt += f"- {key}: {value}\n"

    # Add specific instructions based on event type
    if event.event_type.value == "crypto":
        prompt += """
For cryptocurrency events:
- Consider current market trends and sentiment
- Factor in historical volatility
- Check for any scheduled events (e.g., protocol upgrades, regulations)
- Consider broader market conditions
"""
    elif event.event_type.value == "fred":
        prompt += """
For economic indicator events:
- Consider recent economic data and trends
- Factor in analyst forecasts
- Consider policy decisions and statements
- Check for seasonal patterns
"""
    elif event.event_type.value == "earnings":
        prompt += """
For earnings events:
- Consider company's recent performance and guidance
- Factor in analyst estimates
- Consider industry trends and competitors
- Check for recent news or announcements
"""
    elif event.event_type.value == "llm" and event.is_political:
        prompt += """
For political events:
- Consider recent statements from key political figures
- Factor in historical precedent
- Consider public opinion polls
- Check for scheduled events or deadlines
"""

    prompt += """
Based on all available information, provide a forecast for this event in the requested JSON format.
Focus on being accurate, not overly confident. If information is limited, your confidence should reflect that.
"""

    return prompt


class LLMForecaster(BaseForecaster):
    """Forecaster that uses an LLM to predict events."""
    
    def __init__(
        self,
        event: MinerEvent,
        llm_provider: str = "openai",
        llm_model: Optional[str] = None,
        temperature: float = 0.1,
        **kwargs,
    ):
        """
        Initialize LLM forecaster.
        
        Args:
            event: Event to forecast
            llm_provider: LLM provider to use
            llm_model: LLM model to use (if None, use default)
            temperature: Temperature parameter for LLM
            **kwargs: Additional arguments to pass to parent
        """
        super().__init__(event, **kwargs)
        
        # Get config from miner.yaml
        llm_config = miner_config.get("forecasters", {}).get("llm", {})
        
        self.llm_provider = llm_provider
        self.llm_model = llm_model or llm_config.get("model", "gpt-4")
        self.temperature = temperature
        self.max_tokens = llm_config.get("max_tokens", 1000)
        
        # LLM client
        self.llm_client = get_llm_client()
        
        self.logger.info(
            f"Initialized LLM forecaster for {event.event_id} "
            f"using {self.llm_provider}/{self.llm_model}"
        )
    
    async def _run(self) -> Prediction:
        """
        Generate prediction using LLM.
        
        Returns:
            Prediction object
        """
        # Get prompts
        system_prompt = get_system_prompt()
        user_prompt = get_user_prompt(self.event)
        
        # Call LLM
        self.logger.debug(f"Calling LLM for {self.event.event_id}")
        response = await self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            provider=self.llm_provider,
            model=self.llm_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        # Parse response
        try:
            result = self._parse_response(response.content)
            
            # Create prediction
            prediction = Prediction(
                event_id=self.event.event_id,
                probability=result["probability"],
                confidence=result["confidence"],
                metadata={
                    "model": self.llm_model,
                    "provider": self.llm_provider,
                    "reasoning": result["reasoning"],
                    "sources": result.get("sources", []),
                    "tokens": response.usage.get("total_tokens", 0),
                },
            )
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}", exc_info=True)
            self.logger.debug(f"Raw response: {response.content}")
            
            # Return fallback prediction
            return Prediction(
                event_id=self.event.event_id,
                probability=0.5,
                confidence=0.1,
                metadata={
                    "model": self.llm_model,
                    "provider": self.llm_provider,
                    "error": str(e),
                    "raw_response": response.content[:500],  # Truncate long responses
                },
            )
    
    def _parse_response(self, content: str) -> Dict[str, Any]:
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
        
        # No JSON found, try to extract probability directly
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
                    "sources": [],
                }
                
            except ValueError:
                pass
        
        # Fallback: try to extract any number that could be a probability
        number_match = re.search(r'([0-9.]+)%', content)
        if number_match:
            try:
                probability = float(number_match.group(1)) / 100
                return {
                    "probability": max(0, min(1, probability)),
                    "confidence": 0.5,  # Lower confidence for this method
                    "reasoning": content.strip(),
                    "sources": [],
                }
            except ValueError:
                pass
        
        # If all else fails
        raise ValueError("Could not extract probability from LLM response")


class EnhancedLLMForecaster(LLMForecaster):
    """Enhanced LLM forecaster with additional data retrieval."""
    
    def __init__(
        self,
        event: MinerEvent,
        search_news: bool = True,
        num_news_results: int = 3,
        **kwargs,
    ):
        """
        Initialize enhanced LLM forecaster.
        
        Args:
            event: Event to forecast
            search_news: Whether to search for news
            num_news_results: Number of news results to retrieve
            **kwargs: Additional arguments to pass to parent
        """
        super().__init__(event, **kwargs)
        
        self.search_news = search_news
        self.num_news_results = num_news_results
    
    async def _get_news_context(self) -> str:
        """
        Get news context for the event.
        
        Returns:
            News context string
        """
        # In a real implementation, this would call a news API
        # For now, return a mock response
        self.logger.info(f"Getting news context for {self.event.event_id}")
        
        # Add small delay to simulate API call
        await asyncio.sleep(0.5)
        
        # Mock news results based on event type
        if self.event.is_crypto:
            return """
Recent news about crypto markets:
1. "Bitcoin price continues to rise after recent ETF approval" - CoinDesk, 2023-04-15
2. "Ethereum 2.0 upgrade postponed to Q3" - CryptoNews, 2023-04-12
3. "Major exchange reports 20% increase in trading volume" - Bloomberg, 2023-04-10
"""
        elif self.event.is_economic:
            return """
Recent economic news:
1. "Federal Reserve signals potential rate hike in June" - Wall Street Journal, 2023-04-14
2. "Inflation rate exceeds expectations at 3.8%" - Reuters, 2023-04-11
3. "Unemployment falls to 3.6%, lowest in 18 months" - Financial Times, 2023-04-09
"""
        elif self.event.is_earnings:
            return """
Recent company news:
1. "Tech giant exceeds Q1 earnings expectations" - CNBC, 2023-04-15
2. "Industry analysts predict strong performance in upcoming earnings" - Barron's, 2023-04-13
3. "Company announces new product line ahead of earnings call" - Bloomberg, 2023-04-10
"""
        else:
            return """
Recent relevant news:
1. "Political tensions rise ahead of upcoming summit" - BBC, 2023-04-15
2. "New policy proposal gains bipartisan support" - Washington Post, 2023-04-12
3. "Experts predict continued market volatility" - MarketWatch, 2023-04-10
"""
    
    async def _run(self) -> Prediction:
        """
        Generate prediction using enhanced LLM approach.
        
        Returns:
            Prediction object
        """
        # Get news context if enabled
        news_context = ""
        if self.search_news:
            try:
                news_context = await self._get_news_context()
            except Exception as e:
                self.logger.error(f"Error getting news context: {e}", exc_info=True)
                # Continue without news context
        
        # Get standard prompts
        system_prompt = get_system_prompt()
        user_prompt = get_user_prompt(self.event)
        
        # Add news context if available
        if news_context:
            user_prompt += f"\n\nHere is some recent relevant news that may help with your prediction:\n{news_context}"
        
        # Call LLM with enhanced prompt
        self.logger.debug(f"Calling LLM for {self.event.event_id} with enhanced prompt")
        response = await self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            provider=self.llm_provider,
            model=self.llm_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        # Parse response as in parent class
        try:
            result = self._parse_response(response.content)
            
            # Create prediction
            prediction = Prediction(
                event_id=self.event.event_id,
                probability=result["probability"],
                confidence=result["confidence"],
                metadata={
                    "model": self.llm_model,
                    "provider": self.llm_provider,
                    "reasoning": result["reasoning"],
                    "sources": result.get("sources", []),
                    "tokens": response.usage.get("total_tokens", 0),
                    "enhanced": True,
                    "news_used": bool(news_context),
                },
            )
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error parsing enhanced LLM response: {e}", exc_info=True)
            
            # Fall back to parent implementation
            return await super()._run() 