#!/usr/bin/env python3
"""
Paper trading script for RL trading system.

This script provides a command-line interface for running paper trading
sessions with trained RL models.
"""
import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config_loader import Settings
from trading.paper_trading import PaperTradingEngine, PaperTradingConfig
from utils.logging import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run paper trading session")
    
    parser.add_argument("--config", default="configs/settings.yaml", help="Configuration file")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--symbol", default="MES", help="Trading symbol")
    parser.add_argument("--exchange", default="CME", help="Trading exchange")
    parser.add_argument("--currency", default="USD", help="Trading currency")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial capital")
    parser.add_argument("--duration", type=int, help="Duration in minutes")
    parser.add_argument("--output", default="paper_trading_results", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (no actual trading)")
    
    return parser.parse_args()


async def main():
    """Main function."""
    args = parse_args()
    
    # Load settings
    settings = Settings.from_paths(args.config)
    
    # Create paper trading configuration
    config = PaperTradingConfig(
        model_path=args.model,
        trading_symbol=args.symbol,
        trading_exchange=args.exchange,
        trading_currency=args.currency,
        initial_capital=args.capital,
        output_dir=args.output
    )
    
    # Create paper trading engine
    engine = PaperTradingEngine(settings, config)
    
    try:
        # Initialize
        logger.info("Initializing paper trading engine...")
        if not await engine.initialize():
            logger.error("Failed to initialize paper trading engine")
            return 1
        
        # Print configuration
        logger.info("Paper Trading Configuration:")
        logger.info(f"  Model: {args.model}")
        logger.info(f"  Symbol: {args.symbol}")
        logger.info(f"  Exchange: {args.exchange}")
        logger.info(f"  Currency: {args.currency}")
        logger.info(f"  Initial Capital: ${args.capital:,.2f}")
        logger.info(f"  Duration: {args.duration or 'Indefinite'} minutes")
        logger.info(f"  Output Directory: {args.output}")
        logger.info(f"  Dry Run: {args.dry_run}")
        
        # Run trading session
        logger.info("Starting paper trading session...")
        await engine.run_trading_session(duration_minutes=args.duration)
        
        # Print summary
        summary = engine.get_trading_summary()
        logger.info("\nTrading Session Summary:")
        logger.info(f"  Final Equity: ${summary['final_equity']:,.2f}")
        logger.info(f"  Total Return: {summary['performance_metrics']['total_return']:.2%}")
        logger.info(f"  Annual Return: {summary['performance_metrics']['annual_return']:.2%}")
        logger.info(f"  Sharpe Ratio: {summary['performance_metrics']['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown: {summary['performance_metrics']['max_drawdown']:.2%}")
        logger.info(f"  Win Rate: {summary['performance_metrics']['win_rate']:.2%}")
        logger.info(f"  Total Trades: {summary['total_trades']}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        await engine.stop_trading_session()
        return 0
        
    except Exception as e:
        logger.error(f"Error in paper trading: {e}")
        await engine.stop_trading_session()
        return 1
    
    finally:
        await engine.stop_trading_session()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)