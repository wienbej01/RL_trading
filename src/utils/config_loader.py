# rl-intraday/src/utils/config_loader.py
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# Default: <repo>/rl-intraday/configs/settings.yaml
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "settings.yaml"

# Simple module-level cache
_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_CONFIG_PATH_CACHED: Optional[Path] = None


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML content must be a mapping (dict), got: {type(data).__name__}")
    return data


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

    # Resolve to absolute paths relative to YAML dir
    resolved: Dict[str, str] = {}
    for k, v in paths.items():
        p = Path(v)
        if not p.is_absolute():
            p = (base / p).resolve()
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


def load_config(config_path: Optional[str | Path] = None, use_cache: bool = True) -> Dict[str, Any]:
    """
    Load configuration from YAML, resolve paths, and apply env overrides.

    Returns dict with:
      - cfg['paths'] as absolute strings
      - cfg['__meta__'] with 'config_file', 'config_dir', 'project_root'
      - cfg['secrets'] possibly filled from env
    """
    global _CONFIG_CACHE, _CONFIG_PATH_CACHED

    cfg_path = Path(config_path).resolve() if config_path else DEFAULT_CONFIG_PATH

    if use_cache and _CONFIG_CACHE is not None and _CONFIG_PATH_CACHED == cfg_path:
        return _CONFIG_CACHE

    raw = _read_yaml(cfg_path)
    cfg = _resolve_paths(raw, cfg_path)
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
        config_path: Optional[str | Path] = None,
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
    def from_paths(cls, paths: Optional[Dict[str, Any]] = None, **kwargs) -> "Settings":
        """
        Back-compat shim used by some scripts:
          Settings.from_paths({"data_root": "...", "cache_dir": "..."})
          Settings.from_paths(data_root="...", cache_dir="...")
        """
        # Merge dict arg + kwargs, ignore None
        merged: Dict[str, Any] = {}
        if isinstance(paths, dict):
            merged.update({k: v for k, v in paths.items() if v is not None})
        merged.update({k: v for k, v in kwargs.items() if v is not None})
        return cls(paths_override=merged, use_cache=False)

    def get(self, *keys, default=None):
        """
        Get nested configuration value.
        Usage: settings.get('section', 'key') or settings.get('section', 'subsection', 'key')
        """
        current = self._cfg
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._cfg)


__all__ = ["load_config", "get_config", "Settings"]


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
