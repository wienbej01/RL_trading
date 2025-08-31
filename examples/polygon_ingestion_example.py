#!/usr/bin/env python3
"""
Polygon Data Ingestion Examples

This script demonstrates various usage patterns for the Polygon data ingestion pipeline,
including programmatic usage, bulk operations, and incremental updates.

Examples include:
- Basic data fetching for single/multiple symbols
- Incremental updates
- Different data types (OHLCV, quotes, trades)
- Progress tracking and error handling
- Integration with existing data loader
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.polygon_ingestor import PolygonDataIngestor
from src.data.data_loader import UnifiedDataLoader
from src.utils.config_loader import Settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


async def example_basic_fetch():
    """Example: Basic data fetching for a single symbol."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Data Fetch")
    print("="*60)

    settings = Settings.from_paths('configs/settings.yaml')
    ingestor = PolygonDataIngestor(settings)

    try:
        # Fetch OHLCV data for AAPL
        symbols = ['AAPL']
        start_date = '2023-01-01'
        end_date = '2023-01-05'  # Small date range for demo

        print(f"Fetching OHLCV data for {symbols[0]} from {start_date} to {end_date}")

        results = await ingestor.fetch_historical_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            data_types=['ohlcv']
        )

        print("Results:", results)

    finally:
        await ingestor.aclose()


async def example_bulk_fetch():
    """Example: Bulk data fetching for multiple symbols."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Bulk Data Fetch")
    print("="*60)

    settings = Settings.from_paths('configs/settings.yaml')
    ingestor = PolygonDataIngestor(settings)

    try:
        # Fetch data for multiple symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        start_date = '2023-01-01'
        end_date = '2023-01-05'

        print(f"Fetching data for {len(symbols)} symbols: {', '.join(symbols)}")

        results = await ingestor.fetch_historical_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            data_types=['ohlcv', 'quotes']
        )

        print("Bulk fetch results:")
        for symbol, symbol_result in results['symbol_results'].items():
            print(f"  {symbol}: {symbol_result['records_fetched']} records, {len(symbol_result['errors'])} errors")

    finally:
        await ingestor.aclose()


async def example_incremental_update():
    """Example: Incremental data updates."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Incremental Update")
    print("="*60)

    settings = Settings.from_paths('configs/settings.yaml')
    ingestor = PolygonDataIngestor(settings)

    try:
        # First, do an initial fetch
        print("Performing initial data fetch...")
        await ingestor.fetch_historical_data(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-01-03',
            data_types=['ohlcv']
        )

        # Check status
        status = ingestor.get_ingestion_status('AAPL')
        print("Status after initial fetch:")
        print(f"  Last update: {status['metadata']['AAPL_ohlcv']['last_update']}")

        # Now perform incremental update
        print("\nPerforming incremental update...")
        results = await ingestor.fetch_incremental_update(
            symbols=['AAPL'],
            days_back=2
        )

        print("Incremental update results:", results)

    finally:
        await ingestor.aclose()


async def example_different_data_types():
    """Example: Fetching different data types."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Different Data Types")
    print("="*60)

    settings = Settings.from_paths('configs/settings.yaml')
    ingestor = PolygonDataIngestor(settings)

    try:
        symbol = 'AAPL'
        date = '2023-01-03'  # Single day for demo

        # Fetch different data types
        data_types = ['ohlcv', 'quotes', 'trades']

        for data_type in data_types:
            print(f"\nFetching {data_type} data for {symbol} on {date}")

            results = await ingestor.fetch_historical_data(
                symbols=[symbol],
                start_date=date,
                end_date=date,
                data_types=[data_type]
            )

            records = results['symbol_results'][symbol]['records_fetched']
            print(f"  Fetched {records} {data_type} records")

    finally:
        await ingestor.aclose()


async def example_dry_run():
    """Example: Dry run mode."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Dry Run Mode")
    print("="*60)

    settings = Settings.from_paths('configs/settings.yaml')
    ingestor = PolygonDataIngestor(settings)

    try:
        print("Performing dry run (no actual data fetching)...")

        results = await ingestor.fetch_historical_data(
            symbols=['AAPL', 'MSFT'],
            start_date='2023-01-01',
            end_date='2023-01-05',
            data_types=['ohlcv'],
            dry_run=True
        )

        print("Dry run results:", results)

    finally:
        await ingestor.aclose()


async def example_integration_with_data_loader():
    """Example: Integration with existing data loader."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Integration with Data Loader")
    print("="*60)

    settings = Settings.from_paths('configs/settings.yaml')

    # First, ingest some data
    ingestor = PolygonDataIngestor(settings)
    try:
        print("Ingesting data...")
        await ingestor.fetch_historical_data(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-01-05',
            data_types=['ohlcv']
        )
    finally:
        await ingestor.aclose()

    # Now load the data using the unified data loader
    print("\nLoading ingested data with UnifiedDataLoader...")
    data_loader = UnifiedDataLoader(settings, data_source='polygon')

    try:
        df = data_loader.load_data(
            symbol='AAPL',
            start_date='2023-01-01',
            end_date='2023-01-05',
            data_type='ohlcv'
        )

        print(f"Loaded DataFrame shape: {df.shape}")
        print("Columns:", list(df.columns))
        print("Date range:", df.index.min(), "to", df.index.max())
        print("\nFirst few rows:")
        print(df.head())

    except Exception as e:
        print(f"Error loading data: {e}")


async def example_error_handling():
    """Example: Error handling and recovery."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Error Handling")
    print("="*60)

    settings = Settings.from_paths('configs/settings.yaml')
    ingestor = PolygonDataIngestor(settings)

    try:
        # Try to fetch data for invalid symbols
        symbols = ['AAPL', 'INVALID_SYMBOL_123', 'MSFT']
        start_date = '2023-01-01'
        end_date = '2023-01-05'

        print(f"Fetching data for symbols: {symbols}")
        print("(Note: INVALID_SYMBOL_123 should fail)")

        results = await ingestor.fetch_historical_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            data_types=['ohlcv']
        )

        print("\nResults summary:")
        print(f"  Total symbols: {results['total_symbols']}")
        print(f"  Successful fetches: {results['successful_fetches']}")
        print(f"  Failed fetches: {results['failed_fetches']}")
        print(f"  Total records: {results['total_records']}")

        if results['errors']:
            print(f"\nErrors ({len(results['errors'])}):")
            for error in results['errors']:
                print(f"  - {error}")

    finally:
        await ingestor.aclose()


async def example_status_and_metadata():
    """Example: Checking ingestion status and metadata."""
    print("\n" + "="*60)
    print("EXAMPLE 8: Status and Metadata")
    print("="*60)

    settings = Settings.from_paths('configs/settings.yaml')
    ingestor = PolygonDataIngestor(settings)

    try:
        # Get overall status
        overall_status = ingestor.get_ingestion_status()
        print("Overall status:")
        print(f"  Total symbols: {overall_status['total_symbols']}")
        print(f"  Total data types: {overall_status['total_data_types']}")
        print(f"  Total records: {overall_status['total_records']}")

        # Get status for specific symbol
        if overall_status['total_symbols'] > 0:
            symbol = overall_status['symbols'][0]
            symbol_status = ingestor.get_ingestion_status(symbol)
            print(f"\nStatus for {symbol}:")
            for data_type_key, metadata in symbol_status['metadata'].items():
                data_type = data_type_key.split('_', 1)[1]
                print(f"  {data_type}: {metadata['total_records']} records, last update: {metadata['last_update']}")

    finally:
        await ingestor.aclose()


async def main():
    """Run all examples."""
    print("Polygon Data Ingestion Examples")
    print("===============================")

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    try:
        # Run examples
        await example_basic_fetch()
        await example_bulk_fetch()
        await example_incremental_update()
        await example_different_data_types()
        await example_dry_run()
        await example_integration_with_data_loader()
        await example_error_handling()
        await example_status_and_metadata()

        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(main())