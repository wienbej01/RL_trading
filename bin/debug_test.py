#!/usr/bin/env python3
"""
Debug test script to isolate the unhashable type error
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import Settings

def debug_settings():
    print("=== Debugging Settings ===")

    try:
        # Load settings
        print("Loading settings...")
        settings = Settings.from_paths('configs/settings.yaml')
        print("✓ Settings loaded successfully")
        print(f"  Config keys: {list(settings._config.keys())}")
        print(f"  Config type: {type(settings._config)}")

        # Test the specific call that's failing
        print("\nTesting settings.get calls...")
        print("  Calling settings.get('data', {})...")
        data_section = settings.get('data', {})
        print(f"  ✓ data section: {data_section}")
        print(f"  ✓ data section type: {type(data_section)}")

        print("  Calling settings.get('data', {}).get('cache_enabled', True)...")
        cache_enabled = settings.get('data', {}).get('cache_enabled', True)
        print(f"  ✓ cache_enabled: {cache_enabled}")
        print(f"  ✓ cache_enabled type: {type(cache_enabled)}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_settings()
    print(f"\nDebug result: {'PASSED' if success else 'FAILED'}")