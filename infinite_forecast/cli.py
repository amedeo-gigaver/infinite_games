#!/usr/bin/env python3
"""
Command-line interface for Infinite Forecast tools.

This module provides a command-line utility for interacting with
the Infinite Games platform and running forecasting operations.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Any

from infinite_forecast.api.client.infinite_games import InfiniteGamesClient
from infinite_forecast.api.core.logging import setup_logging, get_logger
from infinite_forecast.forecaster.main import ForecasterManager

logger = get_logger(__name__)


def setup_parser() -> argparse.ArgumentParser:
    """Set up command-line argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Infinite Forecast - Cryptocurrency Prediction Tools"
    )
    
    # Global options
    parser.add_argument(
        "--api-key", 
        help="Infinite Games API key (can also use INFINITE_GAMES_API_KEY env var)"
    )
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level"
    )
    
    # Create subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List events command
    list_parser = subparsers.add_parser("list", help="List open events")
    list_parser.add_argument(
        "--market-type", 
        choices=["crypto", "all"],
        default="all",
        help="Filter by market type"
    )
    list_parser.add_argument(
        "--limit", 
        type=int, 
        default=10,
        help="Maximum number of events to list"
    )
    list_parser.add_argument(
        "--format", 
        choices=["text", "json"],
        default="text",
        help="Output format"
    )
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions for events")
    predict_parser.add_argument(
        "--event-id",
        help="Specific event ID to predict (otherwise processes multiple events)"
    )
    predict_parser.add_argument(
        "--market-type", 
        choices=["crypto", "all"],
        default="all",
        help="Filter by market type"
    )
    predict_parser.add_argument(
        "--limit", 
        type=int, 
        default=5,
        help="Maximum number of events to process"
    )
    predict_parser.add_argument(
        "--format", 
        choices=["text", "json"],
        default="text",
        help="Output format"
    )
    
    # Performance command
    perf_parser = subparsers.add_parser("performance", help="Evaluate forecasting performance")
    perf_parser.add_argument(
        "--days", 
        type=int, 
        default=30,
        help="Number of days to analyze"
    )
    perf_parser.add_argument(
        "--market-type", 
        choices=["crypto", "all"],
        default="all",
        help="Filter by market type"
    )
    perf_parser.add_argument(
        "--format", 
        choices=["text", "json"],
        default="text",
        help="Output format"
    )
    
    # Service command
    service_parser = subparsers.add_parser("service", help="Run as a background service")
    service_parser.add_argument(
        "--interval", 
        type=int, 
        default=3600,
        help="Polling interval in seconds"
    )
    service_parser.add_argument(
        "--events-per-cycle", 
        type=int, 
        default=10,
        help="Maximum events to process per cycle"
    )
    
    return parser


async def list_events(args: argparse.Namespace) -> None:
    """List open events from the platform.
    
    Args:
        args: Command-line arguments
    """
    api_key = args.api_key or os.getenv("INFINITE_GAMES_API_KEY")
    manager = ForecasterManager(api_key=api_key)
    
    # Get events based on market type filter
    if args.market_type == "crypto":
        events = await manager.client.get_crypto_events(limit=args.limit)
    else:
        events = await manager.client.get_events(limit=args.limit)
    
    if args.format == "json":
        print(json.dumps(events, indent=2))
    else:
        # Text format
        if not events:
            print("No events found")
            return
            
        print(f"\nFound {len(events)} events:")
        print(f"{'ID':<12} {'Market':<8} {'Description':<60} {'Target Date':<20}")
        print("-" * 100)
        
        for event in events:
            event_id = event.get("id", "unknown")
            market_type = event.get("market_type", "unknown")
            description = event.get("description", "No description")
            
            # Truncate long descriptions
            if len(description) > 57:
                description = description[:54] + "..."
                
            # Parse target date
            target_date = "Unknown"
            if "close_time" in event:
                target_date = event["close_time"].split("T")[0]
                
            print(f"{event_id:<12} {market_type:<8} {description:<60} {target_date:<20}")


async def make_predictions(args: argparse.Namespace) -> None:
    """Make predictions for events.
    
    Args:
        args: Command-line arguments
    """
    api_key = args.api_key or os.getenv("INFINITE_GAMES_API_KEY")
    manager = ForecasterManager(api_key=api_key)
    
    results = []
    
    # Process single event if specified
    if args.event_id:
        event = await manager.client.get_event(args.event_id)
        
        if "error" in event:
            print(f"Error getting event {args.event_id}: {event['error']}")
            return
            
        # Check market type
        if args.market_type != "all" and event.get("market_type") != args.market_type:
            print(f"Event {args.event_id} is not a {args.market_type} event")
            return
            
        # Process the event
        if event.get("market_type") == "crypto":
            result = await manager.process_crypto_event(event)
            results.append(result)
        else:
            print(f"Unsupported market type: {event.get('market_type')}")
            return
    else:
        # Process multiple events
        if args.market_type == "crypto" or args.market_type == "all":
            crypto_results = await manager.process_crypto_events(limit=args.limit)
            results.extend(crypto_results)
        
        # Add other market types here as they're implemented
    
    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        # Text format
        if not results:
            print("No predictions made")
            return
            
        print(f"\nProcessed {len(results)} events:")
        for result in results:
            event_id = result.get("event_id", "unknown")
            success = result.get("success", False)
            
            if success:
                symbol = result.get("symbol", "unknown")
                prediction = result.get("prediction", "unknown")
                confidence = result.get("confidence", 0)
                
                print(f"✅ Event {event_id} - {symbol}: {prediction} (confidence: {confidence:.2f})")
            else:
                error = result.get("error", "Unknown error")
                print(f"❌ Event {event_id}: {error}")


async def evaluate_performance(args: argparse.Namespace) -> None:
    """Evaluate forecasting performance.
    
    Args:
        args: Command-line arguments
    """
    api_key = args.api_key or os.getenv("INFINITE_GAMES_API_KEY")
    manager = ForecasterManager(api_key=api_key)
    
    # Get performance metrics
    market_type = None if args.market_type == "all" else args.market_type
    performance = await manager.evaluate_performance(days=args.days, market_type=market_type)
    
    if args.format == "json":
        print(json.dumps(performance, indent=2))
    else:
        # Text format
        print("\nForecasting Performance Metrics:")
        print("=" * 40)
        
        if "error" in performance:
            print(f"Error: {performance['error']}")
            return
            
        if "crypto" in performance:
            crypto_perf = performance["crypto"]
            print("Cryptocurrency Forecasting:")
            print("-" * 30)
            
            if "error" in crypto_perf:
                print(f"  Error: {crypto_perf['error']}")
            else:
                print(f"  Number of predictions: {crypto_perf.get('num_predictions', 0)}")
                if "average_accuracy" in crypto_perf:
                    print(f"  Average accuracy: {crypto_perf['average_accuracy']:.2%}")
                
                if "calibration" in crypto_perf:
                    calib = crypto_perf["calibration"]
                    if isinstance(calib, dict) and "correlation" in calib:
                        print(f"  Confidence calibration: {calib['correlation']:.3f}")
                        print(f"  Calibration is significant: {calib.get('significant', False)}")
        
        # Add other market types here as they're implemented


async def run_service(args: argparse.Namespace) -> None:
    """Run as a background service.
    
    Args:
        args: Command-line arguments
    """
    api_key = args.api_key or os.getenv("INFINITE_GAMES_API_KEY")
    manager = ForecasterManager(api_key=api_key)
    
    print(f"Starting Infinite Forecast service with {args.interval} second polling interval")
    print("Press Ctrl+C to stop")
    
    try:
        await manager.run_periodic_forecasting(
            interval_seconds=args.interval,
            limit_per_cycle=args.events_per_cycle
        )
    except KeyboardInterrupt:
        print("\nService stopped by user")
    except Exception as e:
        print(f"\nService stopped due to error: {str(e)}")


async def main() -> None:
    """Main entry point for the CLI."""
    parser = setup_parser()
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(level=getattr(logging, args.log_level))
    
    # Check for API key
    if not args.api_key and not os.getenv("INFINITE_GAMES_API_KEY"):
        logger.warning(
            "No API key provided. Set --api-key or INFINITE_GAMES_API_KEY environment variable."
        )
    
    # Run the appropriate command
    if args.command == "list":
        await list_events(args)
    elif args.command == "predict":
        await make_predictions(args)
    elif args.command == "performance":
        await evaluate_performance(args)
    elif args.command == "service":
        await run_service(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main()) 