"""
Configuration module for the FastAPI application.
Loads configuration from YAML files and environment variables.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class LogConfig(BaseModel):
    """Logging configuration."""
    
    level: str = "INFO"
    file: Optional[str] = None
    rotation: Optional[str] = None
    retention: Optional[str] = None


class APISettings(BaseSettings):
    """API configuration settings loaded from environment variables."""
    
    # API settings
    api_title: str = "Infinite Forecast API"
    api_description: str = "API for Advanced forecasting miners, event resolvers, and trading tools"
    api_version: str = "0.1.0"
    debug: bool = False
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Security
    api_key_header: str = "X-API-Key"
    api_key: Optional[str] = None
    
    # External API keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    perplexity_api_key: Optional[str] = None
    serper_api_key: Optional[str] = None
    
    # Database
    db_url: str = "sqlite:///./infinite_forecast.db"
    
    # Logging
    log_config: LogConfig = LogConfig()
    
    class Config:
        """Pydantic config."""
        
        env_file = ".env"
        env_prefix = "IF_"


@lru_cache()
def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    config_file = Path(config_path)
    if not config_file.exists():
        return {}
    
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


@lru_cache()
def get_miner_config() -> Dict[str, Any]:
    """
    Get miner configuration.
    
    Returns:
        Dictionary containing the miner configuration
    """
    config_dir = Path(__file__).parent.parent.parent.parent / "config"
    return load_yaml_config(str(config_dir / "miner.yaml"))


@lru_cache()
def get_resolver_config() -> Dict[str, Any]:
    """
    Get resolver configuration.
    
    Returns:
        Dictionary containing the resolver configuration
    """
    config_dir = Path(__file__).parent.parent.parent.parent / "config"
    return load_yaml_config(str(config_dir / "resolver.yaml"))


@lru_cache()
def get_generator_config() -> Dict[str, Any]:
    """
    Get generator configuration.
    
    Returns:
        Dictionary containing the generator configuration
    """
    config_dir = Path(__file__).parent.parent.parent.parent / "config"
    return load_yaml_config(str(config_dir / "generator.yaml"))


@lru_cache()
def get_api_settings() -> APISettings:
    """
    Get API settings.
    
    Returns:
        APISettings object
    """
    return APISettings() 