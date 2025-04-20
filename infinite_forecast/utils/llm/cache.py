"""
Cache system for LLM responses with SQLite backend.
"""

import asyncio
import json
import sqlite3
import time
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union, cast

import aiosqlite
from pydantic import BaseModel

from infinite_forecast.api.core.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

T = TypeVar("T")


class CacheConfig(BaseModel):
    """Configuration for the cache."""
    
    enabled: bool = True
    ttl: int = 3600  # 1 hour in seconds
    db_path: str = "cache/llm_cache.db"
    max_size: int = 10000  # Maximum number of items to keep in cache


class LLMCache:
    """Cache for LLM responses with SQLite backend."""
    
    def __init__(self, config: CacheConfig = CacheConfig()):
        """
        Initialize LLM cache.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        self.db_path = Path(config.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = None
        self.lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize the cache database."""
        if not self.config.enabled:
            return
        
        async with self.lock:
            if self.connection is None:
                self.connection = await aiosqlite.connect(self.db_path)
                await self._create_tables()
        
    async def _create_tables(self):
        """Create tables if they don't exist."""
        if not self.connection:
            raise RuntimeError("Database connection not initialized")
        
        async with self.connection.cursor() as cursor:
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS llm_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE,
                    data TEXT,
                    created_at TIMESTAMP,
                    expires_at TIMESTAMP,
                    provider TEXT,
                    model TEXT,
                    tokens INTEGER
                )
            """)
            
            # Create index on expires_at for efficient cleanup
            await cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at ON llm_cache (expires_at)
            """)
            
            # Create index on key for efficient lookup
            await cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_key ON llm_cache (key)
            """)
            
            await self.connection.commit()
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data or None if not found
        """
        if not self.config.enabled or not self.connection:
            return None
        
        try:
            async with self.connection.cursor() as cursor:
                # Get item and check if it's expired
                await cursor.execute(
                    "SELECT data, expires_at FROM llm_cache WHERE key = ?",
                    (key,)
                )
                row = await cursor.fetchone()
                
                if row is None:
                    return None
                
                data, expires_at = row
                
                # Check if expired
                if datetime.utcnow() > datetime.fromisoformat(expires_at):
                    # Delete expired item
                    await cursor.execute(
                        "DELETE FROM llm_cache WHERE key = ?",
                        (key,)
                    )
                    await self.connection.commit()
                    return None
                
                return json.loads(data)
                
        except Exception as e:
            logger.error(f"Error getting item from cache: {e}")
            return None
    
    async def set(
        self,
        key: str,
        data: Dict[str, Any],
        ttl: Optional[int] = None,
        provider: str = "unknown",
        model: str = "unknown",
        tokens: int = 0,
    ):
        """
        Set an item in the cache.
        
        Args:
            key: Cache key
            data: Data to cache
            ttl: Time to live in seconds (overrides default)
            provider: LLM provider
            model: LLM model
            tokens: Number of tokens used
        """
        if not self.config.enabled or not self.connection:
            return
        
        try:
            # Calculate expiration
            ttl = ttl or self.config.ttl
            now = datetime.utcnow()
            expires_at = now + timedelta(seconds=ttl)
            
            async with self.connection.cursor() as cursor:
                # Insert or replace
                await cursor.execute(
                    """
                    INSERT OR REPLACE INTO llm_cache
                    (key, data, created_at, expires_at, provider, model, tokens)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key,
                        json.dumps(data),
                        now.isoformat(),
                        expires_at.isoformat(),
                        provider,
                        model,
                        tokens,
                    )
                )
                
                await self.connection.commit()
                
                # Check if we need to clean up
                await self._clean_if_needed()
                
        except Exception as e:
            logger.error(f"Error setting item in cache: {e}")
    
    async def delete(self, key: str):
        """
        Delete an item from the cache.
        
        Args:
            key: Cache key
        """
        if not self.config.enabled or not self.connection:
            return
        
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(
                    "DELETE FROM llm_cache WHERE key = ?",
                    (key,)
                )
                await self.connection.commit()
                
        except Exception as e:
            logger.error(f"Error deleting item from cache: {e}")
    
    async def clear(self):
        """Clear all items from the cache."""
        if not self.config.enabled or not self.connection:
            return
        
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute("DELETE FROM llm_cache")
                await self.connection.commit()
                
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    async def _clean_if_needed(self):
        """Clean up expired items and enforce max size."""
        if not self.connection:
            return
        
        try:
            async with self.connection.cursor() as cursor:
                # Delete expired items
                now = datetime.utcnow().isoformat()
                await cursor.execute(
                    "DELETE FROM llm_cache WHERE expires_at < ?",
                    (now,)
                )
                
                # Check if we need to enforce max size
                await cursor.execute("SELECT COUNT(*) FROM llm_cache")
                count = (await cursor.fetchone())[0]
                
                if count > self.config.max_size:
                    # Delete oldest items
                    to_delete = count - self.config.max_size
                    await cursor.execute(
                        """
                        DELETE FROM llm_cache
                        WHERE id IN (
                            SELECT id FROM llm_cache
                            ORDER BY created_at ASC
                            LIMIT ?
                        )
                        """,
                        (to_delete,)
                    )
                
                await self.connection.commit()
                
        except Exception as e:
            logger.error(f"Error cleaning cache: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with statistics
        """
        if not self.config.enabled or not self.connection:
            return {
                "enabled": False,
                "count": 0,
                "size": 0,
                "providers": {},
                "models": {},
            }
        
        try:
            stats = {
                "enabled": True,
                "count": 0,
                "size": 0,
                "providers": {},
                "models": {},
            }
            
            async with self.connection.cursor() as cursor:
                # Get count
                await cursor.execute("SELECT COUNT(*) FROM llm_cache")
                stats["count"] = (await cursor.fetchone())[0]
                
                # Get total size (approximate)
                await cursor.execute("SELECT SUM(LENGTH(data)) FROM llm_cache")
                size = (await cursor.fetchone())[0]
                stats["size"] = size if size else 0
                
                # Get provider stats
                await cursor.execute(
                    """
                    SELECT provider, COUNT(*), SUM(tokens)
                    FROM llm_cache
                    GROUP BY provider
                    """
                )
                for provider, count, tokens in await cursor.fetchall():
                    stats["providers"][provider] = {
                        "count": count,
                        "tokens": tokens if tokens else 0,
                    }
                
                # Get model stats
                await cursor.execute(
                    """
                    SELECT model, COUNT(*), SUM(tokens)
                    FROM llm_cache
                    GROUP BY model
                    """
                )
                for model, count, tokens in await cursor.fetchall():
                    stats["models"][model] = {
                        "count": count,
                        "tokens": tokens if tokens else 0,
                    }
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {
                "enabled": True,
                "error": str(e),
                "count": 0,
                "size": 0,
                "providers": {},
                "models": {},
            }
    
    async def close(self):
        """Close the cache connection."""
        if self.connection:
            await self.connection.close()
            self.connection = None


# Create singleton instance
llm_cache = LLMCache()

# Initialize on import
async def initialize_cache():
    """Initialize the cache."""
    await llm_cache.initialize()

# Convenience function to get cache
def get_llm_cache() -> LLMCache:
    """
    Get LLM cache instance.
    
    Returns:
        LLMCache instance
    """
    return llm_cache


def async_cached(ttl: Optional[int] = None):
    """
    Decorator to cache async function results.
    
    Args:
        ttl: Time to live in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key_parts = [func.__name__]
            for arg in args:
                key_parts.append(str(arg))
            for k, v in sorted(kwargs.items()):
                key_parts.append(f"{k}={v}")
            
            cache_key = ":".join(key_parts)
            
            # Check cache
            cache_result = await llm_cache.get(cache_key)
            if cache_result is not None:
                return cache_result
            
            # Call function
            result = await func(*args, **kwargs)
            
            # Cache result
            if result is not None:
                await llm_cache.set(
                    key=cache_key,
                    data=result if isinstance(result, dict) else {"result": result},
                    ttl=ttl,
                )
            
            return result
        
        return wrapper
    
    return decorator 