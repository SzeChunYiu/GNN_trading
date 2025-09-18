"""Application configuration loader with safe defaults.

This helper reads ``config/app.yaml`` (if present) and merges it with the
built-in defaults required by the guarded feature roll-out.  Callers should use
:func:`load_app_config` to obtain a dictionary with predictable keys without
having to worry about missing sections.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "data": {
        "start_date": None,
        "end_date": None,
        "providers": {
            "prices": ["stooq", "yfinance"],
            "fundamentals": "sec_xbrl",
            "news": ["gdelt", "sec_filings"],
        },
        "reporting_delay_days": 3,
    },
    "features": {
        "lookback": 120,
    },
    "labels": {
        "horizons": [5, 20, 50],
    },
    "edges": {
        "topk": 5,
        "use_sector": True,
    },
    "registry": {
        "path": "./models/gnn_crosssec",
    },
    "llm": {
        "enabled": False,
        "provider": "openai",
        "schema_strict": True,
    },
    "ui": {
        "enabled": False,
    },
}


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overlay`` into ``base`` and return a copy."""

    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_app_config(path: str | Path = "config/app.yaml") -> Dict[str, Any]:
    """Load application configuration from ``path`` if it exists."""

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg_path = Path(path)
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        cfg = _deep_merge(cfg, payload)
    return cfg


def ensure_directory(path: str | Path) -> Path:
    """Create ``path`` if necessary and return it as :class:`Path`."""

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def iter_company_dirs(root: str | Path) -> Iterable[Path]:
    """Yield company directories inside ``root`` if they exist."""

    root_path = Path(root)
    companies = root_path / "companies"
    if not companies.exists():
        return []
    return [p for p in companies.iterdir() if p.is_dir()]


__all__ = ["DEFAULT_CONFIG", "load_app_config", "ensure_directory", "iter_company_dirs"]
