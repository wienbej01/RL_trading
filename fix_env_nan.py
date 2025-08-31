#!/usr/bin/env python3
"""
Fix NaN handling in env_intraday_rl.py
"""

import re

# Read the environment file
with open('src/sim/env_intraday_rl.py', 'r') as f:
    content = f.read()

# Fix the observation construction with NaN price handling
old_code = '''        next_ts = self.df.index[min(self.i, len(self.df) - 1)]
        obs = self._obs(next_ts, float(self.df["close"].iloc[min(self.i, len(self.df) - 1)]))'''

new_code = '''        next_ts = self.df.index[min(self.i, len(self.df) - 1)]
        next_price = float(self.df["close"].iloc[min(self.i, len(self.df) - 1)])

        # Handle NaN price values for observation
        if np.isnan(next_price) or np.isinf(next_price):
            next_price = 0.0

        obs = self._obs(next_ts, next_price)'''

content = content.replace(old_code, new_code)

with open('src/sim/env_intraday_rl.py', 'w') as f:
    f.write(content)

print("Fixed NaN handling in env_intraday_rl.py")