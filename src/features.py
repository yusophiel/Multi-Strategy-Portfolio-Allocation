"""Point-in-time feature engineering for strategy sleeves."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import get_cash_sleeve, get_sleeves
from .data import StrategyDataset, rolling_average_correlation


NON_FEATURE_COLUMNS = {"date", "next_date", "sleeve", "target_next_return", "risk_regime"}


def build_feature_frame(dataset: StrategyDataset, config: dict[str, Any]) -> pd.DataFrame:
    returns = dataset.returns.copy()
    macro = dataset.macro.reindex(returns.index).ffill().fillna(0.0)
    regimes = make_point_in_time_regime_features(returns, macro, config)

    feature_cfg = config["features"]
    windows = list(feature_cfg.get("windows", [3, 6, 12]))
    risk_window = int(feature_cfg.get("risk_window", 12))
    target_lag = int(config.get("experiment", {}).get("execution_lag_periods", 1))
    sleeves = get_sleeves(config)
    cash_sleeve = get_cash_sleeve(config)
    targets = returns.shift(-target_lag)
    next_dates = pd.Series(returns.index, index=returns.index).shift(-target_lag)

    market = make_market_features(returns, macro, config)
    rows: list[pd.DataFrame] = []

    for sleeve in sleeves:
        strategy = returns[sleeve].astype(float)
        frame = pd.DataFrame(index=returns.index)
        frame["date"] = returns.index
        frame["next_date"] = next_dates.values
        frame["sleeve"] = sleeve
        frame["target_next_return"] = targets[sleeve]
        frame["sleeve_is_cash"] = 1.0 if sleeve == cash_sleeve else 0.0
        frame["strategy_ret_1m"] = strategy

        for window in windows:
            min_periods = max(3, min(window, window // 2))
            frame[f"strategy_ret_{window}m"] = rolling_compound_return(strategy, window, min_periods)
            frame[f"strategy_vol_{window}m"] = strategy.rolling(window, min_periods=min_periods).std()
            frame[f"strategy_sharpe_{window}m"] = safe_divide(
                frame[f"strategy_ret_{window}m"],
                frame[f"strategy_vol_{window}m"] * np.sqrt(window),
            )
            frame[f"strategy_drawdown_{window}m"] = rolling_window_drawdown(strategy, window, min_periods)

        frame[f"strategy_corr_{risk_window}m"] = rolling_corr_with_others(returns, sleeve, risk_window)
        frame = frame.join(market)
        frame = frame.join(regimes)
        rows.append(frame)

    feature_frame = pd.concat(rows, axis=0, ignore_index=True)
    feature_frame = add_sleeve_cross_sectional_ranks(feature_frame, windows)
    return feature_frame


def make_market_features(
    returns: pd.DataFrame, macro: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    windows = list(config["features"].get("windows", [3, 6, 12]))
    risk_window = int(config["features"].get("risk_window", 12))
    equal_return = returns.mean(axis=1)
    market = pd.DataFrame(index=returns.index)
    market["market_ret_1m"] = equal_return

    for window in windows:
        min_periods = max(3, min(window, window // 2))
        market[f"market_ret_{window}m"] = rolling_compound_return(equal_return, window, min_periods)
        market[f"market_vol_{window}m"] = equal_return.rolling(window, min_periods=min_periods).std()
        market[f"market_drawdown_{window}m"] = rolling_window_drawdown(equal_return, window, min_periods)

    market[f"market_corr_{risk_window}m"] = rolling_average_correlation(returns, risk_window)
    for col in macro.columns:
        series = pd.to_numeric(macro[col], errors="coerce")
        market[f"macro_{col}"] = series
        market[f"macro_{col}_chg_3m"] = series.diff(3)
    return market


def make_point_in_time_regime_features(
    returns: pd.DataFrame, macro: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    feature_cfg = config["features"]
    overlay_cfg = config["risk_overlay"]
    window = int(feature_cfg.get("risk_window", 12))
    min_periods = int(feature_cfg.get("regime_min_periods", 36))
    equal_return = returns.mean(axis=1)

    vol = equal_return.rolling(window, min_periods=max(3, window // 2)).std()
    corr = rolling_average_correlation(returns, window)
    dd = rolling_window_drawdown(equal_return, window, max(3, window // 2))

    vol_threshold = vol.expanding(min_periods=min_periods).quantile(float(overlay_cfg.get("vol_quantile", 0.75))).shift(1)
    corr_threshold = corr.expanding(min_periods=min_periods).quantile(float(overlay_cfg.get("corr_quantile", 0.75))).shift(1)
    dd_limit = float(overlay_cfg.get("drawdown_limit", -0.08))

    regime = pd.DataFrame(index=returns.index)
    regime["regime_high_vol"] = (vol > vol_threshold).astype(float).fillna(0.0)
    regime["regime_high_corr"] = (corr > corr_threshold).astype(float).fillna(0.0)
    regime["regime_drawdown"] = (dd < dd_limit).astype(float).fillna(0.0)

    vix = macro.get("vix_proxy", pd.Series(0.0, index=returns.index)).reindex(returns.index).ffill().fillna(0.0)
    vix_threshold = vix.expanding(min_periods=min_periods).quantile(0.75).shift(1)
    regime["regime_high_vix"] = (vix > vix_threshold).astype(float).fillna(0.0)

    regime["regime_stress_score"] = regime[
        ["regime_high_vol", "regime_high_corr", "regime_drawdown", "regime_high_vix"]
    ].mean(axis=1)
    regime["risk_regime"] = np.where(
        regime["regime_stress_score"] >= 0.50,
        "stress",
        np.where(regime["regime_stress_score"] >= 0.25, "watch", "normal"),
    )
    return regime


def add_sleeve_cross_sectional_ranks(frame: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    out = frame.copy()
    for window in windows:
        col = f"strategy_ret_{window}m"
        if col in out.columns:
            out[f"{col}_rank"] = out.groupby("date")[col].rank(pct=True)
    return out


def feature_columns(feature_frame: pd.DataFrame, include_regime: bool = True) -> list[str]:
    columns = []
    for col in feature_frame.columns:
        if col in NON_FEATURE_COLUMNS:
            continue
        if not pd.api.types.is_numeric_dtype(feature_frame[col]):
            continue
        if not include_regime and col.startswith("regime_"):
            continue
        columns.append(col)
    return columns


def rolling_compound_return(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    return series.rolling(window, min_periods=min_periods).apply(
        lambda values: float(np.prod(1.0 + values) - 1.0),
        raw=True,
    )


def rolling_window_drawdown(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    def drawdown(values: np.ndarray) -> float:
        wealth = np.cumprod(1.0 + values)
        peak = np.maximum.accumulate(wealth)
        return float(np.min(wealth / peak - 1.0))

    return series.rolling(window, min_periods=min_periods).apply(drawdown, raw=True)


def rolling_corr_with_others(returns: pd.DataFrame, sleeve: str, window: int) -> pd.Series:
    values: list[float] = []
    min_rows = max(3, min(window, 6))
    for end in range(len(returns)):
        sample = returns.iloc[max(0, end + 1 - window) : end + 1]
        if len(sample) < min_rows:
            values.append(np.nan)
            continue
        valid_sample = sample.dropna(axis=1, thresh=min_rows)
        if sleeve not in valid_sample.columns or valid_sample.shape[1] < 2:
            values.append(np.nan)
            continue
        corr = valid_sample.corr()[sleeve].drop(labels=[sleeve], errors="ignore")
        corr = corr.replace([np.inf, -np.inf], np.nan).dropna()
        values.append(float(corr.mean()) if not corr.empty else np.nan)
    return pd.Series(values, index=returns.index)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0.0, np.nan)
