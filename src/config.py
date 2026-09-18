"""Configuration loading and normalization."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "default.json"


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into `base` and return a new dict."""

    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the default config, optionally merged with a user config."""

    base = load_json(DEFAULT_CONFIG_PATH)
    if path is None:
        cfg = base
    else:
        cfg = deep_update(base, load_json(path))
    return normalize_config(cfg)


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    sleeves = list(config["universe"]["sleeves"])
    cash = config["universe"].get("cash_sleeve", "cash")
    if cash not in sleeves:
        sleeves.append(cash)
        config["universe"]["sleeves"] = sleeves

    strategic = config["universe"].get("strategic_weights", {})
    missing = [name for name in sleeves if name not in strategic]
    if missing:
        equal_missing = 1.0 / len(sleeves)
        for name in missing:
            strategic[name] = equal_missing

    total = float(sum(strategic.get(name, 0.0) for name in sleeves))
    if total <= 0:
        strategic = {name: 1.0 / len(sleeves) for name in sleeves}
    else:
        strategic = {name: float(strategic.get(name, 0.0)) / total for name in sleeves}
    config["universe"]["strategic_weights"] = strategic

    enabled = config.get("experiment", {}).get("enabled")
    if enabled is None:
        config["experiment"]["enabled"] = []
    return config


def get_sleeves(config: dict[str, Any]) -> list[str]:
    return list(config["universe"]["sleeves"])


def get_cash_sleeve(config: dict[str, Any]) -> str:
    return str(config["universe"].get("cash_sleeve", "cash"))
