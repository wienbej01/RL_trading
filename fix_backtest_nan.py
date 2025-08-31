#!/usr/bin/env python3
"""
Fix NaN handling in polygon_rl_backtest_example.py
"""

import re

# Read the backtest example file
with open('examples/polygon_rl_backtest_example.py', 'r') as f:
    content = f.read()

# Fix the equity curve calculation
old_code = '''        # Calculate episode return
        if len(equity_curve) >= 2:
            episode_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
        else:
            episode_return = 0.0'''

new_code = '''        # Calculate episode return
        if len(equity_curve) >= 2:
            start_equity = equity_curve.iloc[0]
            end_equity = equity_curve.iloc[-1]

            # Handle NaN values
            if np.isnan(start_equity) or np.isnan(end_equity) or start_equity == 0:
                episode_return = 0.0
            else:
                episode_return = (end_equity - start_equity) / start_equity
        else:
            episode_return = 0.0'''

content = content.replace(old_code, new_code)

# Fix the final equity calculation
old_code2 = '''            'final_equity': equity_curve.iloc[-1] if len(equity_curve) > 0 else env.cash'''

new_code2 = '''            'final_equity': equity_curve.iloc[-1] if len(equity_curve) > 0 and not np.isnan(equity_curve.iloc[-1]) else env.cash'''

content = content.replace(old_code2, new_code2)

with open('examples/polygon_rl_backtest_example.py', 'w') as f:
    f.write(content)

print("Fixed NaN handling in polygon_rl_backtest_example.py")