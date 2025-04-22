# Infinite Forecast

Advanced forecasting miners, event resolvers, and trading tools for the Infinite Games platform.

## Overview

Infinite Forecast is a comprehensive solution for the Infinite Games platform, focusing on three key components:

1. **Advanced Miner**: Highly accurate prediction system for future events, optimized for early forecasting across various event types (geopolitical, economic, cryptocurrency, etc.)

2. **LLM Event Resolver**: Cost-efficient fact verification system with >90% accuracy and costs under $0.1 per event

3. **RSS Event Generator**: Real-time event generation from stock market news using RSS feeds

## Installation

### Prerequisites

- Python 3.10 or higher
- Poetry (dependency management)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/infinite-forecast.git
cd infinite-forecast
```

2. Install dependencies:
```bash
poetry install
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Usage

### Running the Miner

```bash
# Run the miner on testnet
poetry run python -m infinite_forecast.miner.client --netuid 155 --subtensor.network test --wallet.name miner --wallet.hotkey default

# Run the miner on mainnet
poetry run python -m infinite_forecast.miner.client --netuid 6 --wallet.name miner --wallet.hotkey default
```

### Running the Event Resolver

```bash
# Start the resolver service
poetry run python -m infinite_forecast.resolver.client
```

### Running the Event Generator

```bash
# Start the event generator service
poetry run python -m infinite_forecast.generator.client
```

### Starting the API

```bash
# Start the FastAPI server
poetry run uvicorn infinite_forecast.api.main:app --reload
```

### Running the UI (Optional)

```bash
# Start the Streamlit interface
poetry run streamlit run infinite_forecast/ui/app.py
```

## Architecture

The system follows a modular architecture with clear separation of concerns:

- **API Layer**: FastAPI-based REST API for interacting with all services
- **Miner Module**: Prediction services for various event types
- **Resolver Module**: Fact verification and resolution services
- **Generator Module**: Event generation from RSS feeds
- **Utilities**: Shared components for LLM integration, data processing, etc.

## Configuration

Configuration is managed via YAML files in the `config/` directory:

- `miner.yaml`: Miner-specific settings
- `resolver.yaml`: Resolver settings
- `generator.yaml`: Generator settings

## License

This project is licensed under the MIT License - see the LICENSE file for details. 