"""Data loading and deterministic sample-data generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import get_cash_sleeve, get_sleeves


@dataclass(frozen=True)
class StrategyDataset:
    returns: pd.DataFrame
    macro: pd.DataFrame
    regimes: pd.Series
    source: str


REGIME_NAMES = ["calm", "growth", "inflation", "stress", "recovery"]


def load_dataset(config: dict[str, Any]) -> StrategyDataset:
    data_cfg = config["data"]
    returns_path = data_cfg.get("returns_csv")
    macro_path = data_cfg.get("macro_csv")

    if returns_path:
        returns = read_timeseries_csv(returns_path, data_cfg.get("date_column", "date"))
        returns = returns.reindex(columns=get_sleeves(config))
        missing = returns.columns[returns.isna().all()].tolist()
        if missing:
            raise ValueError(f"Missing required sleeve columns in returns CSV: {missing}")
        returns = returns.astype(float).sort_index().dropna(how="all")
        macro = (
            read_timeseries_csv(macro_path, data_cfg.get("date_column", "date"))
            if macro_path
            else derive_macro_from_returns(returns)
        )
        macro = macro.reindex(returns.index).ffill().fillna(0.0)
        regimes = infer_sample_regimes(macro)
        return StrategyDataset(returns=returns, macro=macro, regimes=regimes, source=str(returns_path))

    return generate_synthetic_dataset(config)


def read_timeseries_csv(path: str | Path, date_column: str = "date") -> pd.DataFrame:
    raw = pd.read_csv(Path(path).expanduser())
    if date_column in raw.columns:
        dates = pd.to_datetime(raw.pop(date_column))
    else:
        dates = pd.to_datetime(raw.iloc[:, 0])
        raw = raw.iloc[:, 1:]
    raw.index = pd.DatetimeIndex(dates, name="date")
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.sort_index()
    return numeric


def generate_synthetic_dataset(config: dict[str, Any]) -> StrategyDataset:
    """Create a deterministic, regime-rich strategy sleeve dataset.

    The sample is useful for testing the research pipeline end to end. It is not
    intended to be presented as live or firm-sourced investment evidence.
    """

    data_cfg = config["data"]
    sleeves = get_sleeves(config)
    cash_sleeve = get_cash_sleeve(config)
    rng = np.random.default_rng(int(data_cfg.get("seed", 42)))
    frequency = data_cfg.get("frequency", "M")
    if frequency == "M":
        frequency = "ME"
    dates = pd.date_range(
        data_cfg.get("start", "2010-01-31"),
        data_cfg.get("end", "2025-12-31"),
        freq=frequency,
    )

    states = simulate_regime_path(len(dates), rng)
    regimes = pd.Series([REGIME_NAMES[idx] for idx in states], index=dates, name="regime")

    sleeve_profiles = {
        "equity_beta": {"beta": 1.00, "trend": 0.20, "quality": 0.15, "carry": 0.05, "defense": -0.20, "vol": 0.045},
        "trend_following": {"beta": 0.20, "trend": 0.85, "quality": 0.05, "carry": 0.20, "defense": 0.10, "vol": 0.035},
        "value_quality": {"beta": 0.65, "trend": -0.05, "quality": 0.85, "carry": 0.10, "defense": 0.05, "vol": 0.032},
        "carry_macro": {"beta": 0.35, "trend": 0.15, "quality": 0.05, "carry": 0.90, "defense": -0.05, "vol": 0.030},
        "defensive_low_vol": {"beta": 0.20, "trend": 0.00, "quality": 0.35, "carry": 0.00, "defense": 0.90, "vol": 0.018},
        cash_sleeve: {"beta": 0.00, "trend": 0.00, "quality": 0.00, "carry": 0.00, "defense": 0.00, "vol": 0.001},
    }
    regime_factor_means = {
        "calm": np.array([0.006, 0.003, 0.003, 0.002, 0.001]),
        "growth": np.array([0.011, 0.002, 0.005, 0.004, -0.001]),
        "inflation": np.array([-0.003, 0.006, -0.001, 0.005, 0.001]),
        "stress": np.array([-0.028, 0.010, -0.011, -0.012, 0.006]),
        "recovery": np.array([0.016, -0.002, 0.006, 0.002, 0.000]),
    }
    regime_factor_vol = {
        "calm": np.array([0.020, 0.015, 0.012, 0.010, 0.006]),
        "growth": np.array([0.026, 0.016, 0.014, 0.012, 0.007]),
        "inflation": np.array([0.035, 0.020, 0.018, 0.020, 0.009]),
        "stress": np.array([0.070, 0.030, 0.035, 0.040, 0.016]),
        "recovery": np.array([0.045, 0.020, 0.020, 0.018, 0.010]),
    }

    factor_names = ["beta", "trend", "quality", "carry", "defense"]
    returns = pd.DataFrame(index=dates, columns=sleeves, dtype=float)
    macro_rows: list[dict[str, float]] = []

    prev_macro = np.zeros(6)
    for idx, date in enumerate(dates):
        regime = regimes.loc[date]
        means = regime_factor_means[regime]
        vols = regime_factor_vol[regime]
        common = rng.normal(means, vols)

        for sleeve in sleeves:
            profile = sleeve_profiles.get(
                sleeve,
                {"beta": 0.3, "trend": 0.2, "quality": 0.2, "carry": 0.2, "defense": 0.1, "vol": 0.030},
            )
            loading = np.array([profile[name] for name in factor_names])
            noise = rng.normal(0.0, profile["vol"])
            ret = float(loading @ common + noise)
            if sleeve == cash_sleeve:
                ret = 0.0015 + rng.normal(0.0, 0.00025)
            returns.loc[date, sleeve] = np.clip(ret, -0.35, 0.35)

        macro_target = macro_state_vector(regime)
        prev_macro = 0.82 * prev_macro + 0.18 * macro_target + rng.normal(0.0, 0.12, size=6)
        macro_rows.append(
            {
                "growth": prev_macro[0],
                "inflation": prev_macro[1],
                "rates": prev_macro[2],
                "credit_spread": prev_macro[3],
                "vix_proxy": max(5.0, 18.0 + 8.0 * prev_macro[4]),
                "liquidity_proxy": prev_macro[5],
            }
        )

    macro = pd.DataFrame(macro_rows, index=dates)
    macro.index.name = "date"
    returns.index.name = "date"
    return StrategyDataset(returns=returns, macro=macro, regimes=regimes, source="synthetic")


def simulate_regime_path(n_periods: int, rng: np.random.Generator) -> np.ndarray:
    transition = np.array(
        [
            [0.78, 0.10, 0.05, 0.03, 0.04],
            [0.13, 0.72, 0.08, 0.02, 0.05],
            [0.12, 0.08, 0.66, 0.10, 0.04],
            [0.10, 0.02, 0.08, 0.62, 0.18],
            [0.20, 0.25, 0.04, 0.04, 0.47],
        ]
    )
    states = np.zeros(n_periods, dtype=int)
    states[0] = 0
    for idx in range(1, n_periods):
        states[idx] = rng.choice(len(REGIME_NAMES), p=transition[states[idx - 1]])
    return states


def macro_state_vector(regime: str) -> np.ndarray:
    values = {
        "calm": [0.4, -0.2, -0.1, -0.4, -0.4, 0.4],
        "growth": [1.0, 0.0, 0.2, -0.5, -0.5, 0.6],
        "inflation": [0.0, 1.1, 0.9, 0.2, 0.5, -0.1],
        "stress": [-1.2, 0.4, -0.5, 1.4, 1.7, -1.2],
        "recovery": [0.5, -0.1, -0.2, 0.2, 0.2, 0.0],
    }
    return np.array(values[regime], dtype=float)


def derive_macro_from_returns(returns: pd.DataFrame) -> pd.DataFrame:
    equal = returns.mean(axis=1)
    vol = equal.rolling(12, min_periods=3).std().fillna(equal.std())
    trend = equal.rolling(6, min_periods=3).sum().fillna(0.0)
    corr = rolling_average_correlation(returns, 12).fillna(0.0)
    macro = pd.DataFrame(
        {
            "growth": trend,
            "inflation": 0.0,
            "rates": 0.0,
            "credit_spread": vol.rank(pct=True),
            "vix_proxy": 15.0 + 100.0 * vol,
            "liquidity_proxy": -corr,
        },
        index=returns.index,
    )
    return macro.fillna(0.0)


def infer_sample_regimes(macro: pd.DataFrame) -> pd.Series:
    vix = macro.get("vix_proxy", pd.Series(0.0, index=macro.index))
    growth = macro.get("growth", pd.Series(0.0, index=macro.index))
    inflation = macro.get("inflation", pd.Series(0.0, index=macro.index))
    labels = []
    for date in macro.index:
        if vix.loc[date] > vix.quantile(0.75):
            labels.append("stress")
        elif inflation.loc[date] > inflation.quantile(0.75):
            labels.append("inflation")
        elif growth.loc[date] > growth.quantile(0.65):
            labels.append("growth")
        else:
            labels.append("calm")
    return pd.Series(labels, index=macro.index, name="regime")


def rolling_average_correlation(returns: pd.DataFrame, window: int) -> pd.Series:
    values: list[float] = []
    min_rows = max(3, min(window, 6))
    for end in range(len(returns)):
        sample = returns.iloc[max(0, end + 1 - window) : end + 1]
        if len(sample) < min_rows:
            values.append(np.nan)
            continue
        valid_sample = sample.dropna(axis=1, thresh=min_rows)
        if valid_sample.shape[1] < 2:
            values.append(np.nan)
            continue
        corr = valid_sample.corr().to_numpy(dtype=float)
        mask = ~np.eye(corr.shape[0], dtype=bool)
        pairs = corr[mask]
        pairs = pairs[np.isfinite(pairs)]
        values.append(float(np.mean(pairs)) if len(pairs) else np.nan)
    return pd.Series(values, index=returns.index, name="avg_corr")
