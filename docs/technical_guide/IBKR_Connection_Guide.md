# IBKR Connection Guide

This guide explains how to connect to Interactive Brokers (IBKR) through the trade_system_modules from another trading repository.

## Prerequisites

- Python 3.8+
- Interactive Brokers Trader Workstation (TWS) running locally
- Paper trading account configured in TWS
- API connections enabled in TWS (File → Global Configuration → API → Settings)

## 1. Dependencies Installation

Install the required Python packages:

```bash
pip install ib_insync pydantic-settings
```

## 2. Environment Configuration

Create a `.env` file in your project root directory:

```env
# IBKR Connection Settings
IB_HOST=127.0.0.1
IB_PORT=7497
IB_CLIENT_ID=0

# Optional: Other settings if needed
GCS_BUCKET=your_bucket_name
GCP_PROJECT=your_gcp_project
POLYGON_API_KEY=your_polygon_key
```

**Note**: Port 7497 is for paper trading. Use port 7496 for live trading.

## 3. Module Setup

### Option A: Copy Module Files

Copy these files from the trade_system_modules repository to your project:

```
your_project/
├── src/
│   └── trade_system_modules/
│       ├── config/
│       │   └── settings.py
│       ├── execution/
│       │   └── ibkr_exec.py
│       └── data/
│           └── symbology.py
├── .env
└── your_trading_script.py
```

### Option B: Install as Package

From the trade_system_modules directory:

```bash
pip install -e .
```

## 4. Basic Connection Example

```python
#!/usr/bin/env python3
"""
Basic IBKR connection example using trade_system_modules
"""

import sys
import os
from pathlib import Path

# Add module path (if using Option A)
sys.path.append(str(Path(__file__).parent / 'src'))

from trade_system_modules.execution.ibkr_exec import IBExec
from trade_system_modules.config.settings import settings

def main():
    """Main trading function."""
    try:
        print("Initializing IBKR connection...")
        print(f"Connecting to: {settings.ib_host}:{settings.ib_port} (Client ID: {settings.ib_client_id})")

        # Create IBKR execution handler (connects automatically)
        ib_exec = IBExec()

        print("✅ Successfully connected to IBKR!")

        # Example operations:

        # 1. Check connection status
        print(f"Connection status: {ib_exec.ib.isConnected()}")

        # 2. Get account information
        try:
            accounts = ib_exec.ib.managedAccounts()
            print(f"Managed accounts: {accounts}")
        except Exception as e:
            print(f"Could not get accounts: {e}")

        # 3. Place a test market order (uncomment to test)
        # try:
        #     trade = ib_exec.place_market("AAPL", "BUY", 1)
        #     print(f"Test order placed: {trade}")
        # except Exception as e:
        #     print(f"Order failed: {e}")

        # 4. Get current positions
        try:
            positions = ib_exec.get_positions()
            print(f"Current positions: {len(positions)} positions")
            for pos in positions:
                print(f"  - {pos.contract.symbol}: {pos.position}")
        except Exception as e:
            print(f"Could not get positions: {e}")

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("Make sure TWS is running and API is enabled")

    finally:
        # Always disconnect when done
        try:
            if 'ib_exec' in locals():
                ib_exec.ib.disconnect()
                print("✅ Disconnected from IBKR")
        except:
            pass

if __name__ == "__main__":
    main()
```

## 5. Advanced Usage Examples

### Market Order
```python
# Buy 100 shares of AAPL at market
trade = ib_exec.place_market("AAPL", "BUY", 100)
print(f"Order ID: {trade.order.orderId}")
```

### Limit Order
```python
# Buy 100 shares of AAPL with limit price
trade = ib_exec.place_limit("AAPL", "BUY", 100, 150.00)
```

### Cancel Order
```python
# Cancel order by ID
success = ib_exec.cancel(order_id)
```

### Real-time Data Streaming
```python
from trade_system_modules.data.ibkr_live import IBLive

# Create live data handler
ib_live = IBLive()
ib_live.connect()

# Stream real-time bars for AAPL
bars = ib_live.stream_rt_bars("AAPL", duration="1 D", what="TRADES")

# Access the bars data
print(f"Latest bar: {bars[-1] if bars else 'No data'}")
```

## 6. Error Handling

Common connection issues and solutions:

### Connection Refused
```
Error: Connection refused
```
**Solution**: Ensure TWS is running and API connections are enabled.

### Invalid Client ID
```
Error: Client id 0 is already in use
```
**Solution**: Use a different client ID (1, 2, 3, etc.) or close other connections.

### Port Already in Use
```
Error: Port 7497 is already in use
```
**Solution**: Check if another application is using the port, or use a different port.

## 7. Configuration Options

### Paper Trading (Default)
```env
IB_PORT=7497
```

### Live Trading
```env
IB_PORT=7496
```

### Custom Host
```env
IB_HOST=192.168.1.100  # If TWS is on different machine
```

## 8. Troubleshooting

### Check TWS API Settings
1. Open TWS
2. Go to File → Global Configuration → API → Settings
3. Enable "Enable ActiveX and Socket Clients"
4. Set "Socket port" to 7497 (paper) or 7496 (live)
5. Enable "Allow connections from localhost only" (or your network)

### Test Connection Manually
```python
from ib_insync import IB

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=0)
print(f"Connected: {ib.isConnected()}")
ib.disconnect()
```

### Common Issues
- **TWS not running**: Start TWS and log in
- **API not enabled**: Enable API in TWS settings
- **Firewall blocking**: Allow Python through firewall
- **Multiple connections**: Use unique client IDs

## 9. Best Practices

1. **Always disconnect**: Call `ib.disconnect()` when done
2. **Handle exceptions**: Wrap IBKR calls in try-except blocks
3. **Use unique client IDs**: Different applications should use different IDs
4. **Monitor connection**: Check `ib.isConnected()` periodically
5. **Paper trading first**: Test with paper account before live trading

## 10. File Structure Reference

```
your_project/
├── .env                           # Environment variables
├── requirements.txt               # Python dependencies
├── src/
│   └── trade_system_modules/      # Copied module files
│       ├── __init__.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py        # Configuration management
│       ├── execution/
│       │   ├── __init__.py
│       │   └── ibkr_exec.py       # Order execution
│       └── data/
│           ├── __init__.py
│           ├── ibkr_live.py       # Live data streaming
│           └── symbology.py       # Symbol resolution
└── scripts/
    └── test_connection.py         # Connection test script
```

## Support

For issues with IBKR API:
- Check IBKR API documentation
- Review TWS logs
- Ensure latest TWS version
- Contact IBKR support for account-specific issues