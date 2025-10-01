"""
Minimal compatibility shim for legacy `import gym` test code.
Re-exports gymnasium symbols so `import gym` works without the gym package.
"""
from gymnasium import *  # type: ignore
from gymnasium import spaces  # re-export common submodule

__all__ = [name for name in dir() if not name.startswith("_")]

