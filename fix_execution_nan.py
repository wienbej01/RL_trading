#!/usr/bin/env python3
"""
Fix NaN-to-integer conversion issues in execution.py
"""

import re

# Read the execution.py file
with open('src/sim/execution.py', 'r') as f:
    content = f.read()

# Fix the first location (line 245)
old_code1 = '''        # Determine fill quantity
        if abs(quantity) <= available_liquidity:
            fill_quantity = quantity
        else:
            fill_quantity = int(np.sign(quantity) * available_liquidity)'''

new_code1 = '''        # Determine fill quantity
        if abs(quantity) <= available_liquidity:
            fill_quantity = quantity
        else:
            # Handle NaN values in quantity
            if np.isnan(quantity) or np.isinf(quantity):
                fill_quantity = 0
            else:
                fill_quantity = int(np.sign(quantity) * available_liquidity)'''

# Fix the second location (line 279)
old_code2 = '''            # Determine current fill size
            current_fill = min(abs(remaining_quantity), display_size)
            current_fill = int(np.sign(remaining_quantity) * current_fill)'''

new_code2 = '''            # Determine current fill size
            current_fill = min(abs(remaining_quantity), display_size)
            # Handle NaN values in remaining_quantity
            if np.isnan(remaining_quantity) or np.isinf(remaining_quantity):
                current_fill = 0
            else:
                current_fill = int(np.sign(remaining_quantity) * current_fill)'''

content = content.replace(old_code1, new_code1)
content = content.replace(old_code2, new_code2)

with open('src/sim/execution.py', 'w') as f:
    f.write(content)

print("Fixed NaN-to-integer conversion issues in execution.py")