#!/usr/bin/env python3
"""
Fix the NaN-to-integer conversion issue in _risk_sized_contracts method
"""

import re

# Fix the _risk_sized_contracts method in env_intraday_rl.py
with open('src/sim/env_intraday_rl.py', 'r') as f:
    content = f.read()

# Replace the problematic int() conversion
old_method = '''    def _risk_sized_contracts(self, price, atr):
        """Calculate position size based on risk management."""
        contracts = self.risk_manager.calculate_position_size(
            self.equity, price,
            price - atr * self.risk_manager.risk_config.stop_r_multiple,
            atr
        )

        # Handle NaN values
        if np.isnan(contracts) or np.isinf(contracts):
            contracts = 0.0

        return max(0, int(contracts))'''

new_method = '''    def _risk_sized_contracts(self, price, atr):
        """Calculate position size based on risk management."""
        contracts = self.risk_manager.calculate_position_size(
            self.equity, price,
            price - atr * self.risk_manager.risk_config.stop_r_multiple,
            atr
        )

        # Handle NaN values - comprehensive check
        if np.isnan(contracts) or np.isinf(contracts) or contracts is None:
            contracts = 0.0

        # Ensure contracts is a valid number before int conversion
        try:
            contracts_int = int(max(0, contracts))
        except (ValueError, OverflowError):
            logger.warning(f"Invalid contracts value: {contracts}, defaulting to 0")
            contracts_int = 0

        return contracts_int'''

content = content.replace(old_method, new_method)

with open('src/sim/env_intraday_rl.py', 'w') as f:
    f.write(content)

print("Fixed NaN-to-integer conversion issue in _risk_sized_contracts method")