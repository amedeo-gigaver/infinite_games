"""
LLM client with support for multiple providers, rate limiting, and caching.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from infinite_forecast.api.core.config import get_api_settings
from infinite_forecast.api.core.logging import get_logger

# Initialize settings and logger
settings = get_api_settings()
logger = get_logger(__name__)


class RateLimit:
    """Rate limiter implementation."""
    
    def __init__(self, rate: int, period: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            rate: Maximum number of requests per period
            period: Time period in seconds
        """
        self.rate = rate
        self.period = period
        self.tokens = rate
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
        
    async def acquire(self):
        """
        Acquire a rate limit token.
        
        Returns:
            True if a token was acquired, False otherwise
        """
        async with self.lock:
            # Refill tokens if needed
            now = time.time()
            elapsed = now - self.last_refill
            if elapsed >= self.period:
                refill_amount = int(elapsed / self.period) * self.rate
                self.tokens = min(self.rate, self.tokens + refill_amount)
                self.last_refill = now
            
            # Check if we have tokens
            if self.tokens > 0:
                self.tokens -= 1
                return True
            
            return False
            
    async def wait(self):
        """
        Wait until a token is available.
        """
        while not await self.acquire():
            await asyncio.sleep(0.1)


class LLMResponse:
    """Response object from LLM API calls."""
    
    def __init__(
        self,
        content: str,
        model: str,
        provider: str,
        usage: Dict[str, int],
        created_at: datetime = None,
    ):
        """
        Initialize LLM response.
        
        Args:
            content: Response content
            model: Model used
            provider: Provider used
            usage: Token usage information
            created_at: Creation timestamp
        """
        self.content = content
        self.model = model
        self.provider = provider
        self.usage = usage
        self.created_at = created_at or datetime.utcnow()
        
    def to_dict(self) -> Dict:
        """
        Convert response to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "usage": self.usage,
            "created_at": self.created_at.isoformat(),
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> "LLMResponse":
        """
        Create response from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            LLMResponse object
        """
        return cls(
            content=data["content"],
            model=data["model"],
            provider=data["provider"],
            usage=data["usage"],
            created_at=datetime.fromisoformat(data["created_at"]),
        )


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(
        self, 
        model: str,
        api_key: str,
        max_requests_per_minute: int = 60,
    ):
        """
        Initialize LLM provider.
        
        Args:
            model: Model name
            api_key: API key
            max_requests_per_minute: Rate limit
        """
        self.model = model
        self.api_key = api_key
        self.rate_limiter = RateLimit(max_requests_per_minute)
        
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """
        Generate response from LLM.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLMResponse object
        """
        pass
    
    async def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        """
        Stream response from LLM.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Yields:
            Response chunks
        """
        raise NotImplementedError("Streaming not implemented for this provider")


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        max_requests_per_minute: int = 60,
        timeout: int = 30,
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            model: Model name
            api_key: API key
            max_requests_per_minute: Rate limit
            timeout: Timeout in seconds
        """
        super().__init__(
            model=model,
            api_key=api_key or settings.openai_api_key,
            max_requests_per_minute=max_requests_per_minute,
        )
        self.client = openai.AsyncOpenAI(api_key=self.api_key)
        self.timeout = timeout
        
    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(
            (openai.APITimeoutError, openai.APIConnectionError)
        ),
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """
        Generate response from OpenAI.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLMResponse object
        """
        # Wait for rate limit
        await self.rate_limiter.wait()
        
        # Create messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Call API
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.timeout,
            )
            
            # Extract content and usage
            content = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            
            return LLMResponse(
                content=content,
                model=self.model,
                provider="openai",
                usage=usage,
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        """
        Stream response from OpenAI.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Yields:
            Response chunks
        """
        # Wait for rate limit
        await self.rate_limiter.wait()
        
        # Create messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Call API
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.timeout,
                stream=True,
            )
            
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI API streaming error: {e}")
            raise


class AnthropicProvider(LLMProvider):
    """Anthropic API provider."""
    
    def __init__(
        self,
        model: str = "claude-3-opus-20240229",
        api_key: Optional[str] = None,
        max_requests_per_minute: int = 60,
        timeout: int = 30,
    ):
        """
        Initialize Anthropic provider.
        
        Args:
            model: Model name
            api_key: API key
            max_requests_per_minute: Rate limit
            timeout: Timeout in seconds
        """
        super().__init__(
            model=model,
            api_key=api_key or settings.anthropic_api_key,
            max_requests_per_minute=max_requests_per_minute,
        )
        self.timeout = timeout
        
    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(
            (httpx.TimeoutException, httpx.ConnectError)
        ),
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """
        Generate response from Anthropic.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLMResponse object
        """
        # Wait for rate limit
        await self.rate_limiter.wait()
        
        # Create request
        request_data = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        if system_prompt:
            request_data["system"] = system_prompt
        
        # Call API
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
                json=request_data,
            )
            
            if response.status_code != 200:
                logger.error(f"Anthropic API error: {response.text}")
                response.raise_for_status()
                
            response_data = response.json()
            
            # Extract content and usage
            content = response_data["content"][0]["text"]
            usage = {
                "input_tokens": response_data["usage"]["input_tokens"],
                "output_tokens": response_data["usage"]["output_tokens"],
                "total_tokens": (
                    response_data["usage"]["input_tokens"] +
                    response_data["usage"]["output_tokens"]
                ),
            }
            
            return LLMResponse(
                content=content,
                model=self.model,
                provider="anthropic",
                usage=usage,
            )


class PerplexityProvider(LLMProvider):
    """Perplexity API provider."""
    
    def __init__(
        self,
        model: str = "pplx-7b-online",
        api_key: Optional[str] = None,
        max_requests_per_minute: int = 60,
        timeout: int = 30,
    ):
        """
        Initialize Perplexity provider.
        
        Args:
            model: Model name
            api_key: API key
            max_requests_per_minute: Rate limit
            timeout: Timeout in seconds
        """
        super().__init__(
            model=model,
            api_key=api_key or settings.perplexity_api_key,
            max_requests_per_minute=max_requests_per_minute,
        )
        self.timeout = timeout
        
    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(
            (httpx.TimeoutException, httpx.ConnectError)
        ),
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """
        Generate response from Perplexity.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLMResponse object
        """
        # Wait for rate limit
        await self.rate_limiter.wait()
        
        # Create messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Call API
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            
            if response.status_code != 200:
                logger.error(f"Perplexity API error: {response.text}")
                response.raise_for_status()
                
            response_data = response.json()
            
            # Extract content and usage
            content = response_data["choices"][0]["message"]["content"]
            usage = {
                "prompt_tokens": response_data["usage"]["prompt_tokens"],
                "completion_tokens": response_data["usage"]["completion_tokens"],
                "total_tokens": response_data["usage"]["total_tokens"],
            }
            
            return LLMResponse(
                content=content,
                model=self.model,
                provider="perplexity",
                usage=usage,
            )


class LLMClient:
    """Client for interacting with multiple LLM providers."""
    
    def __init__(self):
        """Initialize LLM client."""
        self.providers = {}
        self._initialize_providers()
        self.cache = {}
        
    def _initialize_providers(self):
        """Initialize available providers."""
        # Initialize OpenAI if API key is available
        if settings.openai_api_key:
            self.providers["openai"] = {
                "gpt-4": OpenAIProvider(model="gpt-4"),
                "gpt-4-turbo": OpenAIProvider(model="gpt-4-turbo"),
                "gpt-3.5-turbo": OpenAIProvider(model="gpt-3.5-turbo"),
            }
            
        # Initialize Anthropic if API key is available
        if settings.anthropic_api_key:
            self.providers["anthropic"] = {
                "claude-3-opus": AnthropicProvider(model="claude-3-opus-20240229"),
                "claude-3-sonnet": AnthropicProvider(model="claude-3-sonnet-20240229"),
                "claude-3-haiku": AnthropicProvider(model="claude-3-haiku-20240307"),
            }
            
        # Initialize Perplexity if API key is available
        if settings.perplexity_api_key:
            self.providers["perplexity"] = {
                "pplx-7b-online": PerplexityProvider(model="pplx-7b-online"),
                "pplx-70b-online": PerplexityProvider(model="pplx-70b-online"),
                "mistral-7b": PerplexityProvider(model="mistral-7b-instruct"),
                "llama-3-70b": PerplexityProvider(model="llama-3-70b-instruct"),
            }
            
    def get_provider(self, provider: str, model: str) -> LLMProvider:
        """
        Get provider instance.
        
        Args:
            provider: Provider name
            model: Model name
            
        Returns:
            LLMProvider instance
            
        Raises:
            ValueError: If provider or model not available
        """
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not available")
            
        if model not in self.providers[provider]:
            raise ValueError(f"Model {model} not available for provider {provider}")
            
        return self.providers[provider][model]
    
    def get_cache_key(
        self,
        provider: str,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate cache key.
        
        Args:
            provider: Provider name
            model: Model name
            prompt: User prompt
            system_prompt: System prompt
            temperature: Temperature parameter
            
        Returns:
            Cache key
        """
        key_data = {
            "provider": provider,
            "model": model,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
        }
        return json.dumps(key_data, sort_keys=True)
    
    async def generate(
        self,
        prompt: str,
        provider: str = "openai",
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        use_cache: bool = True,
        cache_ttl: int = 3600,  # 1 hour
    ) -> LLMResponse:
        """
        Generate response from LLM.
        
        Args:
            prompt: User prompt
            provider: Provider name
            model: Model name (if None, use default)
            system_prompt: System prompt
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            use_cache: Whether to use cache
            cache_ttl: Cache TTL in seconds
            
        Returns:
            LLMResponse object
        """
        # Select default model if not specified
        if model is None:
            if provider == "openai":
                model = "gpt-4"
            elif provider == "anthropic":
                model = "claude-3-opus"
            elif provider == "perplexity":
                model = "pplx-70b-online"
                
        # Check if provider and model are available
        provider_instance = self.get_provider(provider, model)
        
        # Check cache if enabled
        if use_cache:
            cache_key = self.get_cache_key(
                provider=provider,
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
            )
            
            if cache_key in self.cache:
                cached_response, timestamp = self.cache[cache_key]
                if datetime.utcnow() < timestamp + timedelta(seconds=cache_ttl):
                    logger.debug(f"Cache hit for {provider}/{model}")
                    return cached_response
                
        # Generate response
        response = await provider_instance.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Cache response if enabled
        if use_cache:
            self.cache[cache_key] = (response, datetime.utcnow())
            
        return response
    
    async def stream(
        self,
        prompt: str,
        provider: str = "openai",
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        """
        Stream response from LLM.
        
        Args:
            prompt: User prompt
            provider: Provider name
            model: Model name (if None, use default)
            system_prompt: System prompt
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Yields:
            Response chunks
        """
        # Select default model if not specified
        if model is None:
            if provider == "openai":
                model = "gpt-4"
            elif provider == "anthropic":
                model = "claude-3-opus"
            elif provider == "perplexity":
                model = "pplx-70b-online"
                
        # Check if provider and model are available
        provider_instance = self.get_provider(provider, model)
        
        # Stream response
        async for chunk in provider_instance.stream(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield chunk


# Create singleton instance
llm_client = LLMClient()

# Convenience function to get client
def get_llm_client() -> LLMClient:
    """
    Get LLM client instance.
    
    Returns:
        LLMClient instance
    """
    return llm_client 