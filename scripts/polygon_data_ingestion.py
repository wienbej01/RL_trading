#!/usr/bin/env python3
"""
Polygon Data Ingestion CLI

Command-line interface for the Polygon data ingestion pipeline.
Supports bulk data fetching, incremental updates, and comprehensive
progress tracking with rich console output.

Usage Examples:
    # Fetch data for specific symbols and date range
    python scripts/polygon_data_ingestion.py \
        --symbols AAPL,MSFT,GOOGL \
        --start-date 2023-01-01 \
        --end-date 2023-12-31 \
        --data-types ohlcv,quotes \
        --config configs/settings.yaml

    # Incremental update
    python scripts/polygon_data_ingestion.py --incremental --config configs/settings.yaml

    # Dry run to check what would be fetched
    python scripts/polygon_data_ingestion.py --dry-run --symbols AAPL

    # Get status of ingested data
    python scripts/polygon_data_ingestion.py --status --symbol AAPL
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.panel import Panel
from rich.text import Text

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.polygon_ingestor import PolygonDataIngestor
from src.utils.config_loader import Settings
from src.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()


class ProgressTracker:
    """Progress tracking for data ingestion operations."""

    def __init__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[bold green]{task.completed}/{task.total}"),
            TextColumn("[dim]{task.fields[status]}"),
            console=console,
            refresh_per_second=2
        )

        self.tasks = {}
        self.overall_task: Optional[TaskID] = None

    def start_overall_progress(self, total_operations: int, description: str = "Overall Progress"):
        """Start overall progress tracking."""
        self.overall_task = self.progress.add_task(
            description,
            total=total_operations,
            completed=0,
            status="Starting..."
        )

    def update_overall_progress(self, completed: int, status: str = ""):
        """Update overall progress."""
        if self.overall_task is not None:
            self.progress.update(
                self.overall_task,
                completed=completed,
                status=status
            )

    def add_symbol_task(self, symbol: str, data_types: List[str]) -> TaskID:
        """Add a task for symbol processing."""
        task_id = self.progress.add_task(
            f"Processing {symbol}",
            total=len(data_types),
            completed=0,
            status="Initializing..."
        )
        self.tasks[symbol] = task_id
        return task_id

    def update_symbol_progress(self, symbol: str, completed: int, status: str = ""):
        """Update progress for a specific symbol."""
        if symbol in self.tasks:
            self.progress.update(
                self.tasks[symbol],
                completed=completed,
                status=status
            )

    def __enter__(self):
        self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.__exit__(exc_type, exc_val, exc_tb)


def parse_symbols(symbol_str: str) -> List[str]:
    """Parse comma-separated symbol string."""
    return [s.strip().upper() for s in symbol_str.split(',') if s.strip()]


def parse_data_types(data_type_str: str) -> List[str]:
    """Parse comma-separated data type string."""
    return [dt.strip().lower() for dt in data_type_str.split(',') if dt.strip()]


def validate_date(date_str: str) -> str:
    """Validate and return date string in YYYY-MM-DD format."""
    try:
        # Try to parse the date
        parsed_date = pd.to_datetime(date_str)
        return parsed_date.strftime('%Y-%m-%d')
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD format.")


def display_status(status: dict, symbol: Optional[str] = None):
    """Display ingestion status in a formatted table."""
    if symbol:
        # Display detailed status for specific symbol
        console.print(f"\n[bold blue]Ingestion Status for {symbol}[/bold blue]")

        if 'metadata' in status and status['metadata']:
            table = Table(title=f"Data Types for {symbol}")
            table.add_column("Data Type", style="cyan")
            table.add_column("Last Update", style="green")
            table.add_column("Date Range", style="yellow")
            table.add_column("Total Records", style="magenta")
            table.add_column("Errors", style="red")

            for data_type_key, meta in status['metadata'].items():
                data_type = data_type_key.split('_', 1)[1] if '_' in data_type_key else data_type_key

                date_range = "N/A"
                if meta['date_range']['start'] and meta['date_range']['end']:
                    date_range = f"{meta['date_range']['start']} to {meta['date_range']['end']}"

                table.add_row(
                    data_type,
                    meta['last_update'] or "Never",
                    date_range,
                    str(meta['total_records']),
                    str(meta['error_count'])
                )

            console.print(table)
        else:
            console.print(f"[yellow]No ingestion metadata found for {symbol}[/yellow]")

    else:
        # Display overall status
        console.print("\n[bold blue]Overall Ingestion Status[/bold blue]")

        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Symbols", str(status.get('total_symbols', 0)))
        table.add_row("Total Data Types", str(status.get('total_data_types', 0)))
        table.add_row("Total Records", str(status.get('total_records', 0)))
        table.add_row("Total Errors", str(status.get('error_count', 0)))

        if status.get('symbols'):
            table.add_row("Symbols", ", ".join(status['symbols'][:5]) + ("..." if len(status['symbols']) > 5 else ""))

        if status.get('data_types'):
            table.add_row("Data Types", ", ".join(status['data_types']))

        console.print(table)


def display_results(results: dict, dry_run: bool = False):
    """Display ingestion results in a formatted way."""
    prefix = "[DRY RUN] " if dry_run else ""

    # Overall summary
    console.print(f"\n[bold green]{prefix}Ingestion Summary[/bold green]")

    summary_table = Table()
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Total Symbols", str(results.get('total_symbols', 0)))
    summary_table.add_row("Total Data Types", str(results.get('total_data_types', 0)))
    summary_table.add_row("Successful Fetches", str(results.get('successful_fetches', 0)))
    summary_table.add_row("Failed Fetches", str(results.get('failed_fetches', 0)))
    summary_table.add_row("Total Records", str(results.get('total_records', 0)))

    console.print(summary_table)

    # Symbol-level results
    if results.get('symbol_results'):
        console.print(f"\n[bold blue]{prefix}Symbol Results[/bold blue]")

        symbol_table = Table()
        symbol_table.add_column("Symbol", style="cyan")
        symbol_table.add_column("Data Types", style="green")
        symbol_table.add_column("Records", style="magenta")
        symbol_table.add_column("Errors", style="red")

        for symbol, symbol_result in results['symbol_results'].items():
            symbol_table.add_row(
                symbol,
                str(symbol_result.get('data_types_processed', 0)),
                str(symbol_result.get('records_fetched', 0)),
                str(len(symbol_result.get('errors', [])))
            )

        console.print(symbol_table)

    # Errors
    if results.get('errors'):
        console.print(f"\n[bold red]{prefix}Errors[/bold red]")
        for error in results['errors'][:10]:  # Show first 10 errors
            console.print(f"• {error}")

        if len(results['errors']) > 10:
            console.print(f"[dim]... and {len(results['errors']) - 10} more errors[/dim]")


async def run_ingestion(args: argparse.Namespace, settings: Settings):
    """Run the data ingestion process."""
    try:
        # Initialize ingestor
        ingestor = PolygonDataIngestor(settings)

        # Determine symbols
        if args.incremental and not args.symbols:
            # For incremental updates, get all tracked symbols
            symbols = None
        elif args.symbols:
            symbols = parse_symbols(args.symbols)
        else:
            console.print("[red]Error: Must specify --symbols or use --incremental[/red]")
            return 1

        # Determine data types
        data_types = parse_data_types(args.data_types) if args.data_types else ['ohlcv']

        # Determine date range
        if args.incremental:
            # For incremental, we'll let the ingestor determine the date range
            start_date = None
            end_date = None
        else:
            if not args.start_date or not args.end_date:
                console.print("[red]Error: Must specify --start-date and --end-date for non-incremental fetches[/red]")
                return 1

            start_date = args.start_date
            end_date = args.end_date

        # Progress tracking
        with ProgressTracker() as progress_tracker:
            if not args.dry_run:
                total_operations = len(symbols) * len(data_types) if symbols else len(data_types)
                progress_tracker.start_overall_progress(total_operations)

            def progress_callback(symbol: str, data_type: str, records: int):
                """Progress callback function."""
                if not args.dry_run:
                    status_msg = f"Fetched {records} {data_type} records"
                    progress_tracker.update_overall_progress(
                        progress_tracker.progress.tasks[progress_tracker.overall_task].completed + 1,
                        f"Processed {symbol} {data_type}"
                    )

            # Run ingestion
            if args.incremental:
                console.print("[bold blue]Starting incremental update...[/bold blue]")
                results = await ingestor.fetch_incremental_update(
                    symbols=symbols,
                    days_back=args.days_back,
                    data_types=data_types
                )
            else:
                console.print(f"[bold blue]Starting data fetch for {len(symbols)} symbols...[/bold blue]")
                results = await ingestor.fetch_historical_data(
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    data_types=data_types,
                    incremental=False,
                    dry_run=args.dry_run,
                    progress_callback=progress_callback
                )

        # Display results
        display_results(results, args.dry_run)

        # Cleanup
        await ingestor.aclose()

        return 0 if results.get('failed_fetches', 0) == 0 else 1

    except Exception as e:
        console.print(f"[red]Error during ingestion: {e}[/red]")
        logger.exception("Ingestion failed")
        return 1


async def run_status_check(args: argparse.Namespace, settings: Settings):
    """Run status check."""
    try:
        ingestor = PolygonDataIngestor(settings)

        if args.symbol:
            status = ingestor.get_ingestion_status(args.symbol)
        else:
            status = ingestor.get_ingestion_status()

        display_status(status, args.symbol)

        await ingestor.aclose()
        return 0

    except Exception as e:
        console.print(f"[red]Error checking status: {e}[/red]")
        logger.exception("Status check failed")
        return 1


async def run_clear_metadata(args: argparse.Namespace, settings: Settings):
    """Clear ingestion metadata."""
    try:
        ingestor = PolygonDataIngestor(settings)

        if args.symbol:
            console.print(f"[yellow]Clearing metadata for symbol: {args.symbol}[/yellow]")
        else:
            console.print("[yellow]Clearing all ingestion metadata[/yellow]")

        ingestor.clear_metadata(symbol=args.symbol, data_type=args.data_type)

        console.print("[green]Metadata cleared successfully[/green]")

        await ingestor.aclose()
        return 0

    except Exception as e:
        console.print(f"[red]Error clearing metadata: {e}[/red]")
        logger.exception("Metadata clear failed")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Polygon Data Ingestion CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch data for specific symbols
  %(prog)s --symbols AAPL,MSFT --start-date 2023-01-01 --end-date 2023-12-31

  # Incremental update
  %(prog)s --incremental

  # Dry run
  %(prog)s --dry-run --symbols AAPL

  # Check status
  %(prog)s --status --symbol AAPL

  # Clear metadata
  %(prog)s --clear-metadata --symbol AAPL
        """
    )

    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='configs/settings.yaml',
        help='Path to configuration file (default: configs/settings.yaml)'
    )

    # Main operation modes
    parser.add_argument(
        '--incremental',
        action='store_true',
        help='Perform incremental update for tracked symbols'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate the operation without actually fetching data'
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='Check ingestion status'
    )

    parser.add_argument(
        '--clear-metadata',
        action='store_true',
        help='Clear ingestion metadata'
    )

    # Data specification
    parser.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated list of symbols (e.g., AAPL,MSFT,GOOGL)'
    )

    parser.add_argument(
        '--symbol',
        type=str,
        help='Single symbol for status/metadata operations'
    )

    parser.add_argument(
        '--start-date',
        type=validate_date,
        help='Start date in YYYY-MM-DD format'
    )

    parser.add_argument(
        '--end-date',
        type=validate_date,
        help='End date in YYYY-MM-DD format'
    )

    parser.add_argument(
        '--data-types',
        type=str,
        default='ohlcv',
        help='Comma-separated list of data types (ohlcv,quotes,trades) (default: ohlcv)'
    )

    parser.add_argument(
        '--data-type',
        type=str,
        help='Single data type for metadata operations'
    )

    # Incremental options
    parser.add_argument(
        '--days-back',
        type=int,
        default=7,
        help='Number of days to look back for incremental updates (default: 7)'
    )

    # Logging
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load configuration
    try:
        settings = Settings.from_paths(args.config)
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        return 1

    # Determine operation
    if args.status:
        operation = run_status_check
    elif args.clear_metadata:
        operation = run_clear_metadata
    else:
        operation = run_ingestion

    # Run the operation
    return asyncio.run(operation(args, settings))


if __name__ == '__main__':
    sys.exit(main())