#!/usr/bin/env python3
"""
Monitoring dashboard script for RL trading system.

This script provides a command-line interface for running the
monitoring dashboard with real-time risk monitoring and analytics.
"""
import argparse
import sys
import signal
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config_loader import Settings
from monitoring.dashboard import MonitoringDashboard, DashboardConfig
from monitoring.risk_monitor import RiskMonitor, RiskLimits
from monitoring.analytics import TradingAnalytics
from utils.logging import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run monitoring dashboard")
    
    parser.add_argument("--config", default="configs/settings.yaml", help="Configuration file")
    parser.add_argument("--output", default="monitoring_dashboard", help="Output directory")
    parser.add_argument("--duration", type=int, help="Duration in seconds (0 for indefinite)")
    parser.add_argument("--interval", type=int, default=5, help="Update interval in seconds")
    parser.add_argument("--max-alerts", type=int, default=10, help="Maximum alerts to display")
    parser.add_argument("--max-trades", type=int, default=50, help="Maximum trades to display")
    parser.add_argument("--max-equity", type=int, default=1000, help="Maximum equity points to display")
    parser.add_argument("--enable-real-time", action="store_true", help="Enable real-time monitoring")
    parser.add_argument("--enable-alerts", action="store_true", help="Enable risk alerts")
    parser.add_argument("--enable-performance", action="store_true", help="Enable performance monitoring")
    parser.add_argument("--enable-risk", action="store_true", help="Enable risk monitoring")
    parser.add_argument("--export-format", choices=["json", "csv"], default="json", help="Export format")
    parser.add_argument("--export-interval", type=int, default=300, help="Export interval in seconds")
    
    return parser.parse_args()


def signal_handler(signum, frame):
    """Handle signal for graceful shutdown."""
    logger.info("Received shutdown signal")
    dashboard.stop_dashboard()
    dashboard.stop_real_time_monitoring()
    sys.exit(0)


def main():
    """Main function."""
    args = parse_args()
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Load settings
    settings = Settings.from_paths(args.config)
    
    # Create dashboard configuration
    config = DashboardConfig(
        update_interval=args.interval,
        max_alerts_display=args.max_alerts,
        max_trades_display=args.max_trades,
        max_equity_points=args.max_equity,
        enable_real_time=args.enable_real_time,
        enable_alerts=args.enable_alerts,
        enable_performance=args.enable_performance,
        enable_risk=args.enable_risk,
        output_dir=args.output
    )
    
    # Create monitoring dashboard
    dashboard = MonitoringDashboard(settings, config)
    
    try:
        logger.info("Starting monitoring dashboard...")
        
        # Print configuration
        logger.info("Monitoring Dashboard Configuration:")
        logger.info(f"  Update Interval: {args.interval} seconds")
        logger.info(f"  Output Directory: {args.output}")
        logger.info(f"  Duration: {args.duration or 'Indefinite'} seconds")
        logger.info(f"  Real-time Monitoring: {args.enable_real_time}")
        logger.info(f"  Risk Alerts: {args.enable_alerts}")
        logger.info(f"  Performance Monitoring: {args.enable_performance}")
        logger.info(f"  Risk Monitoring: {args.enable_risk}")
        logger.info(f"  Export Format: {args.export_format}")
        logger.info(f"  Export Interval: {args.export_interval} seconds")
        
        # Start dashboard
        dashboard.start_dashboard()
        
        # Start real-time monitoring if enabled
        if args.enable_real_time:
            dashboard.start_real_time_monitoring()
        
        # Add some sample data for demonstration
        add_sample_data(dashboard)
        
        # Main loop
        start_time = time.time()
        last_export_time = start_time
        
        while True:
            # Check if duration exceeded
            if args.duration and (time.time() - start_time) >= args.duration:
                logger.info("Duration reached")
                break
            
            # Export data periodically
            if args.export_interval and (time.time() - last_export_time) >= args.export_interval:
                export_path = Path(args.output) / f"dashboard_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{args.export_format}"
                dashboard.export_dashboard_data(str(export_path), args.export_format)
                last_export_time = time.time()
            
            # Print status periodically
            if int(time.time() - start_time) % 60 == 0:  # Every minute
                summary = dashboard.get_dashboard_summary()
                logger.info(f"Dashboard Status - Equity: ${summary.get('equity', 0):,.2f}, "
                          f"Position: {summary.get('current_position', 0)}, "
                          f"Alerts: {summary.get('active_alerts', 0)}, "
                          f"Trades: {summary.get('total_trades', 0)}")
            
            # Sleep
            time.sleep(1)
        
        # Generate final report
        final_report = Path(args.output) / "final_dashboard_report.json"
        dashboard.generate_dashboard_report(str(final_report))
        
        # Save snapshot
        snapshot_dir = Path(args.output) / "snapshot"
        dashboard.save_dashboard_snapshot(str(snapshot_dir))
        
        # Print final summary
        final_summary = dashboard.get_dashboard_summary()
        logger.info("\nFinal Dashboard Summary:")
        logger.info(f"  Final Equity: ${final_summary.get('equity', 0):,.2f}")
        logger.info(f"  Total P&L: ${final_summary.get('total_pnl', 0):,.2f}")
        logger.info(f"  Daily P&L: ${final_summary.get('daily_pnl', 0):,.2f}")
        logger.info(f"  Current Position: {final_summary.get('current_position', 0)}")
        logger.info(f"  Total Trades: {final_summary.get('total_trades', 0)}")
        logger.info(f"  Win Rate: {final_summary.get('win_rate', 0):.2%}")
        logger.info(f"  Sharpe Ratio: {final_summary.get('sharpe_ratio', 0):.2f}")
        logger.info(f"  Max Drawdown: {final_summary.get('max_drawdown', 0):.2%}")
        logger.info(f"  Active Alerts: {final_summary.get('active_alerts', 0)}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
        
    except Exception as e:
        logger.error(f"Error in monitoring dashboard: {e}")
        return 1
    
    finally:
        # Clean up
        dashboard.stop_dashboard()
        dashboard.stop_real_time_monitoring()
        logger.info("Monitoring dashboard stopped")


def add_sample_data(dashboard):
    """Add sample data for demonstration."""
    import random
    from datetime import datetime, timedelta
    
    # Add sample equity data
    base_equity = 100000
    current_time = datetime.now()
    
    for i in range(100):
        timestamp = current_time - timedelta(minutes=i * 5)
        equity = base_equity + random.uniform(-1000, 1000)
        pnl = random.uniform(-500, 500)
        
        dashboard.add_real_time_data('equity', {
            'timestamp': timestamp,
            'equity': equity,
            'pnl': pnl
        })
        
        # Add sample position data
        position = random.randint(-5, 5)
        dashboard.add_real_time_data('position', {
            'timestamp': timestamp,
            'position': position
        })
        
        # Add sample trade data
        if i % 10 == 0:  # Add trade every 10 data points
            trade = {
                'timestamp': timestamp,
                'symbol': 'MES',
                'action': random.choice(['BUY', 'SELL']),
                'quantity': random.randint(1, 3),
                'price': random.uniform(6400, 6500),
                'pnl': random.uniform(-200, 200)
            }
            dashboard.add_real_time_data('trade', trade)
        
        # Add sample alerts occasionally
        if i % 20 == 0:  # Add alert every 20 data points
            from monitoring.risk_monitor import RiskAlert, RiskLevel
            alert = RiskAlert(
                timestamp=timestamp,
                alert_type=random.choice(['POSITION_SIZE_EXCEEDED', 'DRAWDOWN_EXCEEDED', 'DAILY_LOSS_EXCEEDED']),
                severity=random.choice([RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]),
                message=f"Sample alert {i}",
                details={'sample': True}
            )
            dashboard.add_real_time_data('alert', alert)
    
    logger.info("Sample data added to dashboard")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)