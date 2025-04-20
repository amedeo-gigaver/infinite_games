"""
Security utilities for the FastAPI application.
"""

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from infinite_forecast.api.core.config import get_api_settings
from infinite_forecast.api.core.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

# Initialize API key header scheme
api_settings = get_api_settings()
API_KEY_HEADER = APIKeyHeader(name=api_settings.api_key_header, auto_error=False)


async def validate_api_key(
    api_key: str = Security(API_KEY_HEADER),
) -> bool:
    """
    Validate the API key provided in the request.
    
    Args:
        api_key: API key from request header
        
    Returns:
        True if valid
        
    Raises:
        HTTPException: If API key is invalid
    """
    # If API key is not configured, skip validation
    if not api_settings.api_key:
        return True
    
    # If API key is configured but not provided
    if not api_key:
        logger.warning("API key required but not provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": api_settings.api_key_header},
        )
    
    # If API key is provided but doesn't match
    if api_key != api_settings.api_key:
        logger.warning("Invalid API key provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": api_settings.api_key_header},
        )
    
    return True


def get_api_key_dependency():
    """
    Get a FastAPI dependency for API key validation.
    
    Returns:
        Dependency function
    """
    return Depends(validate_api_key) 