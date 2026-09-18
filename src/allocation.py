"""Portfolio allocation rules and transaction-cost accounting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import get_cash_sleeve, get_sleeves


def allocate_portfolio(
    mode: str,
    history: pd.DataFrame,
    forecasts: pd.Series,
    uncertainty: pd.Series,
    previous_weights: pd.Series,
    config: dict[str, Any],
    use_uncertainty: bool = True,
) -> pd.Series:
    sleeves = get_sleeves(config)
    if mode == "equal_weight":
        return allocate_equal(sleeves)
    if mode == "static_allocation":
        return strategic_weights(config)
    if mode == "inverse_volatility":
        return allocate_inverse_volatility(history, config)
    if mode == "forecast_unconstrained":
        return allocate_forecast_unconstrained(history, forecasts, config)
    if mode == "forecast":
        return allocate_forecast_constrained(history, forecasts, uncertainty, previous_weights, config, use_uncertainty)
    raise ValueError(f"Unknown allocation mode: {mode}")


def allocate_equal(sleeves: list[str]) -> pd.Series:
    return pd.Series(1.0 / len(sleeves), index=sleeves, dtype=float)


def strategic_weights(config: dict[str, Any]) -> pd.Series:
    sleeves = get_sleeves(config)
    weights = pd.Series(config["universe"]["strategic_weights"], dtype=float).reindex(sleeves).fillna(0.0)
    return weights / weights.sum()


def allocate_inverse_volatility(history: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    sleeves = get_sleeves(config)
    cash = get_cash_sleeve(config)
    window = int(config["features"].get("covariance_window", 36))
    sample = history.reindex(columns=sleeves).tail(window)
    vol = sample.std(ddof=1).replace(0.0, np.nan)
    inv_vol = 1.0 / vol
    if cash in inv_vol.index:
        inv_vol.loc[cash] = max(inv_vol.drop(cash, errors="ignore").median(), 1.0)
    raw = inv_vol.replace([np.inf, -np.inf], np.nan).fillna(inv_vol.median())
    return normalize_with_caps(raw, config)


def allocate_forecast_constrained(
    history: pd.DataFrame,
    forecasts: pd.Series,
    uncertainty: pd.Series,
    previous_weights: pd.Series,
    config: dict[str, Any],
    use_uncertainty: bool = True,
) -> pd.Series:
    sleeves = get_sleeves(config)
    cash = get_cash_sleeve(config)
    alloc_cfg = config["allocation"]
    window = int(config["features"].get("covariance_window", 36))
    sample = history.reindex(columns=sleeves).tail(window)
    vol = sample.std(ddof=1).reindex(sleeves).fillna(sample.stack().std(ddof=1))
    vol = vol.clip(lower=0.002)

    mu = forecasts.reindex(sleeves).fillna(0.0).astype(float)
    if use_uncertainty:
        unc = uncertainty.reindex(sleeves).fillna(float(uncertainty.mean())).clip(
            lower=float(alloc_cfg.get("uncertainty_floor", 0.01))
        )
        confidence = (mu.abs() / (float(alloc_cfg.get("confidence_scale", 1.5)) * unc)).clip(0.0, 1.0)
        neutral_mu = pd.Series(mu.mean(), index=sleeves)
        mu = confidence * mu + (1.0 - confidence) * neutral_mu

    risk_aversion = float(alloc_cfg.get("risk_aversion", 4.0))
    score = mu / np.power(vol, max(risk_aversion / 4.0, 0.25))
    score = score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    zscore = robust_zscore(score)
    temperature = max(float(alloc_cfg.get("signal_temperature", 0.65)), 0.05)
    raw = pd.Series(np.exp(np.clip(zscore / temperature, -4.0, 4.0)), index=sleeves)

    if cash in raw.index:
        negative_pressure = float((-mu.drop(cash, errors="ignore")).clip(lower=0.0).sum())
        stress = min(3.0, negative_pressure / max(mu.abs().sum(), 1e-6))
        raw.loc[cash] = max(raw.loc[cash], 1.0 + 4.0 * stress)

    signal_weights = normalize_with_caps(raw, config)
    neutral = strategic_weights(config)
    neutral_blend = float(alloc_cfg.get("neutral_blend", 0.25))
    blended = neutral_blend * neutral + (1.0 - neutral_blend) * signal_weights

    # Damp abrupt optimizer jumps even when the signal itself is noisy.
    previous = previous_weights.reindex(sleeves).fillna(0.0)
    turnover_preview = 0.5 * float((blended - previous).abs().sum())
    max_monthly_turnover = 0.65
    if turnover_preview > max_monthly_turnover:
        step = max_monthly_turnover / turnover_preview
        blended = previous + step * (blended - previous)
    return normalize_with_caps(blended, config)


def allocate_forecast_unconstrained(history: pd.DataFrame, forecasts: pd.Series, config: dict[str, Any]) -> pd.Series:
    sleeves = get_sleeves(config)
    window = int(config["features"].get("covariance_window", 36))
    sample = history.reindex(columns=sleeves).tail(window)
    cov = sample.cov().reindex(index=sleeves, columns=sleeves).fillna(0.0)
    ridge = np.eye(len(sleeves)) * 1e-4
    mu = forecasts.reindex(sleeves).fillna(0.0).values
    try:
        raw = np.linalg.solve(cov.values + ridge, mu)
    except np.linalg.LinAlgError:
        raw = np.linalg.pinv(cov.values + ridge) @ mu
    gross = np.sum(np.abs(raw))
    if gross <= 0 or not np.isfinite(gross):
        return strategic_weights(config)
    weights = pd.Series(raw / gross, index=sleeves)
    return weights.clip(lower=-0.30, upper=0.60)


def normalize_with_caps(raw: pd.Series, config: dict[str, Any]) -> pd.Series:
    sleeves = get_sleeves(config)
    alloc_cfg = config["allocation"]
    min_weight = float(alloc_cfg.get("min_weight", 0.0))
    max_weight = float(alloc_cfg.get("max_weight", 1.0))
    weights = raw.reindex(sleeves).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weights = weights.clip(lower=0.0)
    if weights.sum() <= 0:
        weights = pd.Series(1.0, index=sleeves)
    weights = weights / weights.sum()
    weights = weights.clip(lower=min_weight)

    for _ in range(100):
        over = weights > max_weight
        under = ~over
        if not over.any():
            break
        weights.loc[over] = max_weight
        remainder = 1.0 - float(weights.loc[over].sum())
        if remainder <= 0 or not under.any():
            break
        base = raw.reindex(weights.loc[under].index).clip(lower=0.0)
        if base.sum() <= 0:
            weights.loc[under] = remainder / int(under.sum())
        else:
            weights.loc[under] = remainder * base / base.sum()

    residual = 1.0 - float(weights.sum())
    if abs(residual) > 1e-10:
        free = weights[weights < max_weight - 1e-12]
        if free.empty:
            weights += residual / len(weights)
        else:
            weights.loc[free.index] += residual * free / free.sum()
    return weights.reindex(sleeves).fillna(0.0)


def compute_turnover(new_weights: pd.Series, previous_weights: pd.Series) -> float:
    common = new_weights.index.union(previous_weights.index)
    new = new_weights.reindex(common).fillna(0.0)
    old = previous_weights.reindex(common).fillna(0.0)
    return 0.5 * float((new - old).abs().sum())


def transaction_cost(turnover: float, config: dict[str, Any], enabled: bool = True) -> float:
    if not enabled:
        return 0.0
    return turnover * float(config["allocation"].get("transaction_cost_bps", 0.0)) / 10000.0


def gross_exposure(weights: pd.Series) -> float:
    return float(weights.abs().sum())


def robust_zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    median = float(values.median())
    mad = float((values - median).abs().median())
    if mad <= 1e-12:
        std = float(values.std(ddof=0))
        if std <= 1e-12:
            return pd.Series(0.0, index=series.index)
        return (values - float(values.mean())) / std
    return 0.6745 * (values - median) / mad
