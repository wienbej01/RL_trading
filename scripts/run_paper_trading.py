#!/usr/bin/env python3
import argparse
import asyncio
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import Settings
from src.trading.paper_trading import PaperTradingEngine, PaperTradingConfig

def main():
    parser = argparse.ArgumentParser(description="Run paper trading session.")
    parser.add_argument("--config", type=str, default="configs/settings.yaml", help="Path to configuration file.")
    parser.add_argument("--duration", type=int, default=None, help="Duration of the trading session in minutes.")
    args = parser.parse_args()

    settings = Settings.from_paths(args.config)
    
    # Create PaperTradingConfig from settings
    paper_trading_config = PaperTradingConfig(
        model_path=settings.get('paper_trading', 'model_path', default="models/trained_model.zip"),
        trading_symbol=settings.get('paper_trading', 'trading_symbol', default="MES"),
        trading_exchange=settings.get('paper_trading', 'trading_exchange', default="CME"),
        trading_currency=settings.get('paper_trading', 'trading_currency', default="USD"),
        initial_capital=settings.get('paper_trading', 'initial_capital', default=100000.0),
        max_position_size=settings.get('paper_trading', 'max_position_size', default=10),
        risk_per_trade_frac=settings.get('risk', 'risk_per_trade_frac', default=0.02),
        stop_loss_r_multiple=settings.get('risk', 'stop_r_multiple', default=1.0),
        take_profit_r_multiple=settings.get('risk', 'tp_r_multiple', default=1.5),
        max_daily_loss_r=settings.get('risk', 'max_daily_loss_r', default=3.0),
        update_frequency=settings.get('paper_trading', 'update_frequency', default=60),
        data_lookback=settings.get('paper_trading', 'data_lookback', default=120),
        output_dir=settings.get('paper_trading', 'output_dir', default="paper_trading_results"),
    )

    engine = PaperTradingEngine(settings=settings, config=paper_trading_config)
    
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(engine.initialize())
        loop.run_until_complete(engine.run_trading_session(duration_minutes=args.duration))
    except KeyboardInterrupt:
        print("Paper trading session interrupted by user.")
    finally:
        loop.run_until_complete(engine.stop_trading_session())
        loop.close()

if __name__ == "__main__":
    main()
