"""
Rate limiting utilities for API calls.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast

from infinite_forecast.api.core.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class RateLimitRule:
    """Rate limit rule configuration."""
    
    # Maximum number of requests per period
    requests: int
    
    # Period in seconds
    period: int
    
    # Current token count
    remaining: int = field(init=False)
    
    # Last refill timestamp
    last_refill: float = field(init=False)
    
    def __post_init__(self):
        """Initialize remaining tokens and last refill timestamp."""
        self.remaining = self.requests
        self.last_refill = time.time()
        
    def refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        if elapsed >= self.period:
            # Calculate number of periods elapsed
            periods = int(elapsed / self.period)
            # Refill tokens
            self.remaining = min(self.remaining + periods * self.requests, self.requests)
            # Update last refill timestamp
            self.last_refill = now - (elapsed % self.period)
            
    def acquire(self) -> bool:
        """
        Acquire a token if available.
        
        Returns:
            True if a token was acquired, False otherwise
        """
        self.refill()
        
        if self.remaining > 0:
            self.remaining -= 1
            return True
        
        return False


class RateLimiter:
    """Rate limiter implementation."""
    
    def __init__(self):
        """Initialize rate limiter."""
        self.rules: Dict[str, RateLimitRule] = {}
        self.locks: Dict[str, asyncio.Lock] = {}
        
    def add_rule(self, key: str, requests: int, period: int = 60):
        """
        Add a rate limit rule.
        
        Args:
            key: Rule identifier
            requests: Maximum number of requests per period
            period: Period in seconds
        """
        self.rules[key] = RateLimitRule(requests=requests, period=period)
        self.locks[key] = asyncio.Lock()
        
    async def acquire(self, key: str) -> bool:
        """
        Acquire a token for the specified rule.
        
        Args:
            key: Rule identifier
            
        Returns:
            True if a token was acquired, False otherwise
            
        Raises:
            KeyError: If rule doesn't exist
        """
        if key not in self.rules:
            raise KeyError(f"Rate limit rule '{key}' does not exist")
        
        async with self.locks[key]:
            return self.rules[key].acquire()
            
    async def wait(self, key: str, timeout: Optional[float] = None) -> bool:
        """
        Wait until a token is available or timeout expires.
        
        Args:
            key: Rule identifier
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if a token was acquired, False if timeout expired
            
        Raises:
            KeyError: If rule doesn't exist
        """
        if key not in self.rules:
            raise KeyError(f"Rate limit rule '{key}' does not exist")
        
        # If no timeout, wait indefinitely
        if timeout is None:
            while True:
                if await self.acquire(key):
                    return True
                await asyncio.sleep(0.1)
        
        # With timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            if await self.acquire(key):
                return True
            await asyncio.sleep(0.1)
        
        return False
    
    def get_remaining(self, key: str) -> int:
        """
        Get remaining tokens for the specified rule.
        
        Args:
            key: Rule identifier
            
        Returns:
            Remaining tokens
            
        Raises:
            KeyError: If rule doesn't exist
        """
        if key not in self.rules:
            raise KeyError(f"Rate limit rule '{key}' does not exist")
        
        # Refill tokens if needed
        self.rules[key].refill()
        
        return self.rules[key].remaining
    
    def get_wait_time(self, key: str) -> float:
        """
        Get estimated wait time until a token is available.
        
        Args:
            key: Rule identifier
            
        Returns:
            Estimated wait time in seconds
            
        Raises:
            KeyError: If rule doesn't exist
        """
        if key not in self.rules:
            raise KeyError(f"Rate limit rule '{key}' does not exist")
        
        # Refill tokens if needed
        self.rules[key].refill()
        
        # If tokens available, wait time is 0
        if self.rules[key].remaining > 0:
            return 0.0
        
        # Calculate time until next refill
        now = time.time()
        elapsed = now - self.rules[key].last_refill
        time_until_refill = self.rules[key].period - elapsed
        
        return max(0.0, time_until_refill)


# Create singleton instance
rate_limiter = RateLimiter()

# Configure common limits
rate_limiter.add_rule("openai", requests=3500, period=60)  # 3500 RPM for most models
rate_limiter.add_rule("openai_gpt4", requests=500, period=60)  # 500 RPM for GPT-4
rate_limiter.add_rule("anthropic", requests=100, period=60)  # 100 RPM for Claude
rate_limiter.add_rule("perplexity", requests=60, period=60)  # 60 RPM for Perplexity

# Convenience function to get rate limiter
def get_rate_limiter() -> RateLimiter:
    """
    Get rate limiter instance.
    
    Returns:
        RateLimiter instance
    """
    return rate_limiter


def rate_limited(key: str, timeout: Optional[float] = None):
    """
    Decorator to apply rate limiting to functions.
    
    Args:
        key: Rate limit rule identifier
        timeout: Maximum time to wait in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            """Async function wrapper."""
            # Wait for rate limit
            acquired = await rate_limiter.wait(key, timeout)
            
            if not acquired:
                logger.warning(f"Rate limit timeout for '{key}'")
                raise Exception(f"Rate limit timeout for '{key}'")
            
            # Call function
            return await func(*args, **kwargs)
            
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            """Sync function wrapper."""
            # Create event loop if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Wait for rate limit
            acquired = loop.run_until_complete(rate_limiter.wait(key, timeout))
            
            if not acquired:
                logger.warning(f"Rate limit timeout for '{key}'")
                raise Exception(f"Rate limit timeout for '{key}'")
            
            # Call function
            return func(*args, **kwargs)
        
        # Choose wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return cast(Callable[..., T], async_wrapper)
        else:
            return cast(Callable[..., T], sync_wrapper)
        
    return decorator 