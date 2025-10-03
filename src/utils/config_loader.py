# rl-intraday/src/utils/config_loader.py
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Backward-compat error type expected by older tests
class ConfigError(Exception):
    """Configuration parsing/validation error (compat shim)."""
    pass

logger = logging.getLogger(__name__)

# Default: <repo>/rl-intraday/configs/settings.yaml
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "settings.yaml"

# Simple module-level cache
_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_CONFIG_PATH_CACHED: Optional[Path] = None


def _read_yaml(path: Path) -> Dict[str, Any]:
    """Read a YAML file and return a dict, wrapping errors as ConfigError for compatibility.

    Note: Do not pre-check filesystem existence to allow tests to mock `open`/`yaml.safe_load`.
    """
    try:
        try:
            # Use builtins.open to allow tests to mock file I/O easily
            with open(str(path), "r") as f:
                try:
                    data = yaml.safe_load(f) or {}
                except yaml.YAMLError as e:
                    raise ConfigError("Error parsing YAML") from e
        except FileNotFoundError as e:
            raise ConfigError(f"Configuration file not found: {path}") from e

        if not isinstance(data, dict):
            raise ConfigError(
                f"Top-level YAML content must be a mapping (dict), got: {type(data).__name__}"
            )
        return data
    except ConfigError:
        # Re-raise
        raise
    except Exception as e:
        # Normalize unexpected I/O errors as ConfigError
        raise ConfigError(str(e)) from e


# Backward-compat function expected by tests
def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Compatibility wrapper that loads YAML into a dict and raises ConfigError on failures."""
    return _read_yaml(Path(path))


def _resolve_paths(cfg: Dict[str, Any], cfg_path: Path) -> Dict[str, Any]:
    """
    Resolve path-like entries under cfg['paths']:
    - If YAML lacks 'paths', supply sensible defaults.
    - Resolve any relative path against the YAML file's directory (cfg_path.parent),
      not the process current working directory.
    - Apply environment overrides if present.
    """
    base = cfg_path.parent               # e.g., <repo>/rl-intraday/configs
    project_root = base.parent           # e.g., <repo>/rl-intraday

    paths: Dict[str, Any] = dict(cfg.get("paths") or {})

    # Defaults (used if YAML has no 'paths' block)
    defaults = {
        "data_root": project_root / "data",
        "cache_dir": project_root / "data" / "cache",
        "polygon_raw_dir": project_root / "data" / "polygon" / "historical",
    }
    for k, v in defaults.items():
        paths.setdefault(k, str(v))

    # Optional env overrides
    env_overrides = {
        "data_root": os.getenv("RL_DATA_ROOT"),
        "cache_dir": os.getenv("RL_CACHE_DIR"),
        "polygon_raw_dir": os.getenv("RL_POLYGON_DIR"),
    }
    for k, v in env_overrides.items():
        if v:
            paths[k] = v

    # Resolve to absolute paths relative to project root
    resolved: Dict[str, str] = {}
    for k, v in paths.items():
        p = Path(v)
        if not p.is_absolute():
            p = (project_root / p).resolve()
        resolved[k] = str(p)

    cfg["paths"] = resolved

    # Useful meta
    meta = cfg.setdefault("__meta__", {})
    meta["config_file"] = str(cfg_path)
    meta["config_dir"] = str(base)
    meta["project_root"] = str(project_root)

    return cfg


def _apply_secret_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Allow environment variables to supply secrets if YAML doesn't.
    Supports:
      - POLYGON_API_KEY -> cfg['secrets']['polygon_api_key']
    """
    secrets = cfg.setdefault("secrets", {})
    if not secrets.get("polygon_api_key"):
        env_key = os.getenv("POLYGON_API_KEY")
        if env_key:
            secrets["polygon_api_key"] = env_key
    return cfg


def load_config(config_path: Optional[str | Path | Dict[str, Any]] = None, use_cache: bool = True) -> Dict[str, Any]:
    """
    Load configuration from YAML, resolve paths, and apply env overrides.

    Returns dict with:
      - cfg['paths'] as absolute strings
      - cfg['__meta__'] with 'config_file', 'config_dir', 'project_root'
      - cfg['secrets'] possibly filled from env
    """
    global _CONFIG_CACHE, _CONFIG_PATH_CACHED

    # Allow passing a pre-loaded dict for compatibility with some tests
    if isinstance(config_path, dict):
        # Use provided mapping verbatim for strict test equality
        cfg = dict(config_path)
        cfg_path = DEFAULT_CONFIG_PATH
    else:
        cfg_path = Path(config_path).resolve() if config_path else DEFAULT_CONFIG_PATH

        if use_cache and _CONFIG_CACHE is not None and _CONFIG_PATH_CACHED == cfg_path:
            return _CONFIG_CACHE

        raw = _read_yaml(cfg_path)
        cfg = _resolve_paths(raw, cfg_path)
    # Only apply secret/path overrides for file-based loads
    if not isinstance(config_path, dict):
        cfg = _apply_secret_overrides(cfg)

    logger.info("Loaded configuration from %s", cfg_path)
    try:
        logger.info("Resolved paths: %s", cfg.get("paths", {}))
    except Exception:
        pass

    if use_cache:
        _CONFIG_CACHE = cfg
        _CONFIG_PATH_CACHED = cfg_path

    return cfg


def get_config() -> Dict[str, Any]:
    """Convenience accessor using default path + cache."""
    return load_config(None, use_cache=True)


# ---------- Backward-compatible Settings class ----------

class _AttrDict(dict):
    """Dict that also allows attribute access (obj.key)."""
    __getattr__ = dict.get
    def __setattr__(self, k, v): self[k] = v
    def __delattr__(self, k): del self[k]


def _resolve_override_paths(base_dir: Path, overrides: Dict[str, Any]) -> Dict[str, str]:
    """Resolve override paths against base_dir, return as strings."""
    resolved: Dict[str, str] = {}
    for k, v in overrides.items():
        if v is None:
            continue
        p = Path(v)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        resolved[k] = str(p)
    return resolved


class Settings:
    """
    Backward-compatible wrapper that many modules import.

    Supported usage:
      - Settings()
      - Settings(config_path=".../settings.yaml")
      - Settings.from_yaml(".../settings.yaml")
      - Settings.from_paths(paths={"cache_dir": "...", "data_root": "..."})
      - Settings.from_paths(cache_dir="...", data_root="...")   # kwargs form
      - s.paths['cache_dir'] / s.paths.cache_dir
      - s.secrets['polygon_api_key'] / s.secrets.polygon_api_key
      - s.to_dict()
    """
    def __init__(
        self,
        config_path: Optional[str | Path | Dict[str, Any]] = None,
        use_cache: bool = True,
        paths_override: Optional[Dict[str, Any]] = None,
        secrets_override: Optional[Dict[str, Any]] = None,
    ):
        # avoid stale cache if applying overrides
        cfg = load_config(config_path, use_cache=(use_cache and not paths_override and not secrets_override))
        base_dir = Path(cfg.get("__meta__", {}).get("config_dir", DEFAULT_CONFIG_PATH.parent))

        # Apply path overrides (resolve relative to config dir)
        if paths_override:
            cfg_paths = dict(cfg.get("paths", {}))
            cfg_paths.update(_resolve_override_paths(base_dir, paths_override))
            cfg["paths"] = cfg_paths

        # Apply secret overrides (direct mapping, no path resolution)
        if secrets_override:
            secrets = dict(cfg.get("secrets", {}))
            secrets.update({k: v for k, v in secrets_override.items() if v is not None})
            cfg["secrets"] = secrets

        self._cfg = cfg
        self._config = self._cfg                    # legacy alias expected by callers
        self.config = self._cfg                     # sometimes accessed as .config
        self.paths = _AttrDict(cfg.get("paths", {}))
        self.secrets = _AttrDict(cfg.get("secrets", {}))
        self.meta = _AttrDict(cfg.get("__meta__", {}))

    @classmethod
    def from_yaml(cls, config_path: Optional[str | Path] = None) -> "Settings":
        return cls(config_path=config_path, use_cache=True)

    @classmethod
    def load(cls, config_path: Optional[str | Path] = None) -> "Settings":
        return cls(config_path=config_path, use_cache=True)

    @classmethod
    def from_paths(cls, *args, **kwargs) -> "Settings":
        """
        Back-compat shim supporting two modes:
          - File mode: Settings.from_paths("config1.yaml", "config2.yaml") merges YAMLs (later wins)
          - Override mode: Settings.from_paths({"data_root": "..."}, cache_dir="...")
        """
        # File mode if any positional arg is a string/Path
        if args and all(isinstance(a, (str, Path)) for a in args):
            merged_cfg: Dict[str, Any] = {}
            for p in args:
                cfg = _read_yaml(Path(p))
                # shallow merge: later files override earlier
                merged_cfg.update(cfg)
            return cls(config_path=merged_cfg, use_cache=False)

        # Override mode: merge dict + kwargs into paths_override
        merged: Dict[str, Any] = {}
        if args and isinstance(args[0], dict):
            merged.update({k: v for k, v in args[0].items() if v is not None})
        merged.update({k: v for k, v in kwargs.items() if v is not None})
        return cls(paths_override=merged, use_cache=False)

    def get(self, *keys, default=None):
        """
        Get nested configuration value.
        Supports an optional `default=` keyword. Raises ConfigError when missing and no default.
        """
        current: Any = self._cfg
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                if default is not None:
                    return default
                raise ConfigError("Configuration key not found")
        return current

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._cfg)


__all__ = [
    "load_config",
    "get_config",
    "Settings",
    "ConfigError",
    "load_yaml",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = get_config()
    print("CONFIG FILE :", cfg.get("__meta__", {}).get("config_file"))
    print("CONFIG DIR  :", cfg.get("__meta__", {}).get("config_dir"))
    print("PROJECT ROOT:", cfg.get("__meta__", {}).get("project_root"))
    print("PATHS       :", cfg.get("paths"))
    if "secrets" in cfg:
        redacted = {k: ("<set>" if bool(v) else "<empty>") for k, v in cfg["secrets"].items()}
        print("SECRETS     :", redacted)
