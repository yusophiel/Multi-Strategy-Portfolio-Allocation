"""Evaluation metrics for forecasts and portfolios."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_performance_metrics(
    returns: pd.Series,
    periods_per_year: int = 12,
    cost: pd.Series | None = None,
    turnover: pd.Series | None = None,
    exposure: pd.Series | None = None,
) -> dict[str, float]:
    series = pd.Series(returns).dropna().astype(float)
    if series.empty:
        return {name: float("nan") for name in default_metric_names()}

    wealth = (1.0 + series).cumprod()
    n_periods = len(series)
    annual_return = float(wealth.iloc[-1] ** (periods_per_year / n_periods) - 1.0)
    annual_vol = float(series.std(ddof=1) * np.sqrt(periods_per_year)) if n_periods > 1 else 0.0
    sharpe = safe_ratio(annual_return, annual_vol)
    downside = series[series < 0.0]
    downside_vol = float(downside.std(ddof=1) * np.sqrt(periods_per_year)) if len(downside) > 1 else 0.0
    sortino = safe_ratio(annual_return, downside_vol)
    max_dd = max_drawdown(series)
    cvar_5 = conditional_value_at_risk(series, 0.05)

    out = {
        "periods": float(n_periods),
        "cumulative_return": float(wealth.iloc[-1] - 1.0),
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "cvar_5": cvar_5,
        "hit_rate": float((series > 0.0).mean()),
        "best_period": float(series.max()),
        "worst_period": float(series.min()),
    }
    out["avg_cost"] = float(pd.Series(cost).mean()) if cost is not None and len(cost) else 0.0
    out["avg_turnover"] = float(pd.Series(turnover).mean()) if turnover is not None and len(turnover) else 0.0
    out["avg_exposure"] = float(pd.Series(exposure).mean()) if exposure is not None and len(exposure) else 1.0
    return out


def default_metric_names() -> list[str]:
    return [
        "periods",
        "cumulative_return",
        "annual_return",
        "annual_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "cvar_5",
        "hit_rate",
        "best_period",
        "worst_period",
        "avg_cost",
        "avg_turnover",
        "avg_exposure",
    ]


def max_drawdown(returns: pd.Series) -> float:
    series = pd.Series(returns).dropna().astype(float)
    if series.empty:
        return float("nan")
    wealth = (1.0 + series).cumprod()
    peak = wealth.cummax()
    return float((wealth / peak - 1.0).min())


def conditional_value_at_risk(returns: pd.Series, alpha: float = 0.05) -> float:
    series = pd.Series(returns).dropna().astype(float)
    if series.empty:
        return float("nan")
    threshold = series.quantile(alpha)
    tail = series[series <= threshold]
    return float(tail.mean()) if not tail.empty else float(threshold)


def rank_ic_by_date(forecasts: pd.DataFrame, realized: pd.DataFrame) -> pd.Series:
    common_dates = forecasts.index.intersection(realized.index)
    values: dict[pd.Timestamp, float] = {}
    for date in common_dates:
        f = forecasts.loc[date].astype(float)
        r = realized.loc[date].astype(float)
        common = f.dropna().index.intersection(r.dropna().index)
        if len(common) < 3:
            values[date] = np.nan
            continue
        f_rank = f[common].rank()
        r_rank = r[common].rank()
        if f_rank.nunique() < 2 or r_rank.nunique() < 2:
            values[date] = np.nan
        else:
            values[date] = float(f_rank.corr(r_rank))
    return pd.Series(values, name="rank_ic")


def mean_rank_ic(forecasts: pd.DataFrame, realized: pd.DataFrame) -> float:
    values = rank_ic_by_date(forecasts, realized).dropna()
    if values.empty:
        return float("-inf")
    return float(values.mean())


def ranking_accuracy_by_date(forecasts: pd.DataFrame, realized: pd.DataFrame) -> pd.Series:
    common_dates = forecasts.index.intersection(realized.index)
    output: dict[pd.Timestamp, float] = {}
    for date in common_dates:
        f = forecasts.loc[date].astype(float)
        r = realized.loc[date].astype(float)
        common = list(f.dropna().index.intersection(r.dropna().index))
        correct = 0
        total = 0
        for i, left in enumerate(common):
            for right in common[i + 1 :]:
                f_order = np.sign(f[left] - f[right])
                r_order = np.sign(r[left] - r[right])
                if f_order == 0 or r_order == 0:
                    continue
                correct += int(f_order == r_order)
                total += 1
        output[date] = correct / total if total else np.nan
    return pd.Series(output, name="ranking_accuracy")


def forecast_error_metrics(forecasts: pd.DataFrame, realized: pd.DataFrame) -> dict[str, float]:
    common_dates = forecasts.index.intersection(realized.index)
    f = forecasts.loc[common_dates]
    r = realized.loc[common_dates]
    aligned_f, aligned_r = f.align(r, join="inner", axis=1)
    error = aligned_f - aligned_r
    return {
        "forecast_mae": float(error.abs().stack().mean()),
        "forecast_rmse": float(np.sqrt((error.pow(2)).stack().mean())),
        "rank_ic": mean_rank_ic(aligned_f, aligned_r),
        "ranking_accuracy": float(ranking_accuracy_by_date(aligned_f, aligned_r).dropna().mean()),
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0 or not np.isfinite(denominator):
        return float("nan")
    return float(numerator / denominator)


def to_jsonable(mapping: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, (np.floating, np.integer)):
            out[key] = float(value)
        elif isinstance(value, dict):
            out[key] = to_jsonable(value)
        elif isinstance(value, list):
            out[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in value]
        else:
            out[key] = value
    return out
