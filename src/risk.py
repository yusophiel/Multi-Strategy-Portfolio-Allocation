"""Portfolio risk-state monitoring and overlay policies."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import get_cash_sleeve
from .data import rolling_average_correlation


def classify_risk_state(
    strategy_history: pd.DataFrame,
    portfolio_history: pd.Series,
    config: dict[str, Any],
    recent_forecast_error: float | None = None,
) -> dict[str, float | str | bool]:
    overlay_cfg = config["risk_overlay"]
    window = int(config["features"].get("risk_window", 12))
    min_obs = max(6, window)
    if len(strategy_history) < min_obs:
        return {
            "risk_regime": "normal",
            "risk_score": 0.0,
            "portfolio_vol": 0.0,
            "portfolio_drawdown": 0.0,
            "avg_corr": 0.0,
            "vol_flag": False,
            "drawdown_flag": False,
            "corr_flag": False,
            "forecast_error_flag": False,
        }

    if len(portfolio_history.dropna()) >= min_obs:
        port = portfolio_history.dropna().astype(float)
    else:
        port = strategy_history.mean(axis=1).astype(float)

    rolling_vol = port.rolling(window, min_periods=max(4, window // 2)).std()
    current_vol = float(rolling_vol.iloc[-1]) if np.isfinite(rolling_vol.iloc[-1]) else 0.0
    vol_threshold = rolling_vol.expanding(min_periods=min_obs).quantile(
        float(overlay_cfg.get("vol_quantile", 0.75))
    ).shift(1)
    vol_limit = float(vol_threshold.iloc[-1]) if len(vol_threshold.dropna()) else float(rolling_vol.dropna().median())
    vol_flag = bool(current_vol > vol_limit) if np.isfinite(vol_limit) else False

    recent_dd = current_drawdown(port.tail(window))
    drawdown_flag = bool(recent_dd < float(overlay_cfg.get("drawdown_limit", -0.08)))

    avg_corr_series = rolling_average_correlation(strategy_history, window)
    current_corr = float(avg_corr_series.iloc[-1]) if np.isfinite(avg_corr_series.iloc[-1]) else 0.0
    corr_threshold = avg_corr_series.expanding(min_periods=min_obs).quantile(
        float(overlay_cfg.get("corr_quantile", 0.75))
    ).shift(1)
    corr_limit = float(corr_threshold.iloc[-1]) if len(corr_threshold.dropna()) else float(avg_corr_series.dropna().median())
    corr_flag = bool(current_corr > corr_limit) if np.isfinite(corr_limit) else False

    forecast_error_flag = bool(recent_forecast_error is not None and recent_forecast_error > 0.05)
    raw_score = (
        0.35 * float(vol_flag)
        + 0.30 * float(drawdown_flag)
        + 0.25 * float(corr_flag)
        + 0.10 * float(forecast_error_flag)
    )
    soft_vol = soft_excess(current_vol, vol_limit)
    soft_corr = soft_excess(current_corr, corr_limit)
    soft_dd = min(1.0, abs(min(recent_dd, 0.0)) / max(abs(float(overlay_cfg.get("drawdown_limit", -0.08))), 1e-6))
    risk_score = float(np.clip(0.55 * raw_score + 0.20 * soft_vol + 0.15 * soft_corr + 0.10 * soft_dd, 0.0, 1.0))

    if risk_score >= 0.55:
        regime = "stress"
    elif risk_score >= float(overlay_cfg.get("risk_on_threshold", 0.25)):
        regime = "watch"
    else:
        regime = "normal"

    return {
        "risk_regime": regime,
        "risk_score": risk_score,
        "portfolio_vol": current_vol,
        "portfolio_drawdown": recent_dd,
        "avg_corr": current_corr,
        "vol_flag": vol_flag,
        "drawdown_flag": drawdown_flag,
        "corr_flag": corr_flag,
        "forecast_error_flag": forecast_error_flag,
    }


def apply_risk_overlay(
    weights: pd.Series,
    risk_state: dict[str, float | str | bool],
    config: dict[str, Any],
    policy: str = "none",
) -> tuple[pd.Series, dict[str, float | str]]:
    if policy == "none":
        return weights.copy(), {"policy": "none", "reduction": 0.0, "hedge_ratio": 0.0}

    cash = get_cash_sleeve(config)
    overlay_cfg = config["risk_overlay"]
    risk_score = float(risk_state.get("risk_score", 0.0))

    if policy == "static":
        reduction = float(overlay_cfg.get("static_reduction", 0.35)) if risk_score >= 0.55 else 0.0
        hedge_ratio = 0.0
    elif policy == "dynamic":
        threshold = float(overlay_cfg.get("risk_on_threshold", 0.25))
        if risk_score < threshold:
            reduction = 0.0
            hedge_ratio = 0.0
        else:
            scaled = (risk_score - threshold) / max(1.0 - threshold, 1e-6)
            reduction = float(overlay_cfg.get("dynamic_max_reduction", 0.65)) * scaled
            hedge_ratio = float(overlay_cfg.get("max_hedge_ratio", 0.35)) * scaled
    else:
        raise ValueError(f"Unknown risk overlay policy: {policy}")

    overlaid = weights.copy().astype(float)
    risky = [col for col in overlaid.index if col != cash]
    moved_to_cash = float(overlaid.loc[risky].sum() * reduction)
    overlaid.loc[risky] = overlaid.loc[risky] * (1.0 - reduction)
    if cash in overlaid.index:
        overlaid.loc[cash] = overlaid.loc[cash] + moved_to_cash
    else:
        overlaid = pd.concat([overlaid, pd.Series({cash: moved_to_cash})])
    overlaid = overlaid / overlaid.sum()
    action = {
        "policy": policy,
        "reduction": float(reduction),
        "hedge_ratio": float(hedge_ratio),
    }
    return overlaid, action


def hedge_pnl(
    hedge_ratio: float,
    next_returns: pd.Series,
    previous_hedge_ratio: float,
    config: dict[str, Any],
    costs_enabled: bool = True,
) -> tuple[float, float]:
    equity_leg = "equity_beta" if "equity_beta" in next_returns.index else next_returns.index[0]
    hedge_return = -float(hedge_ratio) * float(next_returns[equity_leg])
    if costs_enabled:
        hedge_cost = abs(float(hedge_ratio) - float(previous_hedge_ratio)) * float(
            config["risk_overlay"].get("hedge_cost_bps", 0.0)
        ) / 10000.0
    else:
        hedge_cost = 0.0
    return hedge_return, hedge_cost


def current_drawdown(returns: pd.Series) -> float:
    series = pd.Series(returns).dropna().astype(float)
    if series.empty:
        return 0.0
    wealth = (1.0 + series).cumprod()
    peak = wealth.cummax()
    return float((wealth / peak - 1.0).iloc[-1])


def soft_excess(value: float, threshold: float) -> float:
    if not np.isfinite(value) or not np.isfinite(threshold) or threshold <= 0:
        return 0.0
    return float(np.clip((value - threshold) / threshold, 0.0, 1.0))
