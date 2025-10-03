"""
Minimal compatibility shim for legacy `import gym`.
Forwards to gymnasium so code written for Gym v0.x can import.
"""
from gymnasium import *  # type: ignore
from gymnasium import spaces  # expose common submodule

__all__ = [name for name in dir() if not name.startswith("_")]

