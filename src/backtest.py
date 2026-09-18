"""Integrated walk-forward backtest and ablation suite."""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .allocation import (
    allocate_equal,
    allocate_portfolio,
    compute_turnover,
    gross_exposure,
    strategic_weights,
    transaction_cost,
)
from .config import get_cash_sleeve, get_sleeves
from .data import StrategyDataset, load_dataset
from .features import build_feature_frame, feature_columns
from .forecasting import fit_select_predict
from .metrics import compute_performance_metrics, forecast_error_metrics
from .risk import apply_risk_overlay, classify_risk_state, hedge_pnl


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    label: str
    allocation_mode: str
    requires_forecast: bool
    use_uncertainty: bool
    overlay_policy: str
    costs_enabled: bool = True
    include_regime_features: bool = True


@dataclass
class ExperimentResult:
    spec: ExperimentSpec
    returns: pd.DataFrame
    weights: pd.DataFrame
    forecasts: pd.DataFrame
    risk_states: pd.DataFrame
    attribution: pd.DataFrame
    model_selection: pd.DataFrame
    metrics: dict[str, float]


@dataclass
class SuiteResult:
    config: dict[str, Any]
    dataset: StrategyDataset
    features: pd.DataFrame
    experiments: dict[str, ExperimentResult]
    summary: pd.DataFrame
    forecast_metrics: pd.DataFrame


ForecastCache = dict[tuple[pd.Timestamp, bool, tuple[str, ...]], Any]
ActiveSleeveCache = dict[tuple[pd.Timestamp, pd.Timestamp], list[str]]
ActiveConfigCache = dict[tuple[str, ...], dict[str, Any]]
CHECKPOINT_VERSION = 1


def run_experiment_suite(
    config: dict[str, Any],
    progress: bool = False,
    checkpoint_dir: str | Path | None = None,
) -> SuiteResult:
    dataset = load_dataset(config)
    features = build_feature_frame(dataset, config)
    specs = build_experiment_specs(config)
    forecast_cache: ForecastCache = {}
    active_sleeves_cache: ActiveSleeveCache = {}
    active_config_cache: ActiveConfigCache = {}
    checkpoint_root = Path(checkpoint_dir).expanduser().resolve() if checkpoint_dir else None
    config_hash = config_fingerprint(config)
    if checkpoint_root is not None:
        checkpoint_root.mkdir(parents=True, exist_ok=True)

    experiments = {}
    for idx, spec in enumerate(specs, start=1):
        cached = load_experiment_checkpoint(checkpoint_root, spec, config_hash)
        if cached is not None:
            if progress:
                print(f"[{idx}/{len(specs)}] Skipping completed experiment: {spec.label} ({spec.name})", flush=True)
            experiments[spec.name] = cached
            continue

        if progress:
            print(f"[{idx}/{len(specs)}] Running experiment: {spec.label} ({spec.name})", flush=True)
        result = run_single_experiment(
            dataset,
            features,
            config,
            spec,
            forecast_cache=forecast_cache,
            active_sleeves_cache=active_sleeves_cache,
            active_config_cache=active_config_cache,
        )
        experiments[spec.name] = result
        save_experiment_checkpoint(checkpoint_root, spec, config_hash, result)
    summary = pd.DataFrame({name: result.metrics for name, result in experiments.items()}).T
    fmetrics = build_forecast_metric_table(experiments)
    return SuiteResult(
        config=config,
        dataset=dataset,
        features=features,
        experiments=experiments,
        summary=summary,
        forecast_metrics=fmetrics,
    )


def config_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def experiment_checkpoint_path(checkpoint_root: Path, spec: ExperimentSpec) -> Path:
    return checkpoint_root / f"{spec.name}.pkl"


def load_experiment_checkpoint(
    checkpoint_root: Path | None,
    spec: ExperimentSpec,
    config_hash: str,
) -> ExperimentResult | None:
    if checkpoint_root is None:
        return None
    path = experiment_checkpoint_path(checkpoint_root, spec)
    if not path.exists():
        return None
    try:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ValueError, TypeError):
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("version") != CHECKPOINT_VERSION:
        return None
    if payload.get("config_hash") != config_hash:
        return None
    result = payload.get("result")
    if not isinstance(result, ExperimentResult):
        return None
    if result.spec.name != spec.name:
        return None
    return result


def save_experiment_checkpoint(
    checkpoint_root: Path | None,
    spec: ExperimentSpec,
    config_hash: str,
    result: ExperimentResult,
) -> None:
    if checkpoint_root is None:
        return
    path = experiment_checkpoint_path(checkpoint_root, spec)
    tmp_path = path.with_name(f"{path.name}.tmp")
    payload = {
        "version": CHECKPOINT_VERSION,
        "config_hash": config_hash,
        "experiment": spec.name,
        "result": result,
    }
    with tmp_path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def build_experiment_specs(config: dict[str, Any]) -> list[ExperimentSpec]:
    registry = {
        "equal_weight": ExperimentSpec(
            name="equal_weight",
            label="Equal Weight",
            allocation_mode="equal_weight",
            requires_forecast=False,
            use_uncertainty=False,
            overlay_policy="none",
        ),
        "static_allocation": ExperimentSpec(
            name="static_allocation",
            label="Static Allocation",
            allocation_mode="static_allocation",
            requires_forecast=False,
            use_uncertainty=False,
            overlay_policy="none",
        ),
        "inverse_volatility": ExperimentSpec(
            name="inverse_volatility",
            label="Inverse Volatility",
            allocation_mode="inverse_volatility",
            requires_forecast=False,
            use_uncertainty=False,
            overlay_policy="none",
        ),
        "equal_weight_overlay": ExperimentSpec(
            name="equal_weight_overlay",
            label="Equal Weight + Dynamic Overlay",
            allocation_mode="equal_weight",
            requires_forecast=False,
            use_uncertainty=False,
            overlay_policy="dynamic",
        ),
        "inverse_volatility_overlay": ExperimentSpec(
            name="inverse_volatility_overlay",
            label="Inverse Volatility + Dynamic Overlay",
            allocation_mode="inverse_volatility",
            requires_forecast=False,
            use_uncertainty=False,
            overlay_policy="dynamic",
        ),
        "forecast_allocation": ExperimentSpec(
            name="forecast_allocation",
            label="Forecast Allocation",
            allocation_mode="forecast",
            requires_forecast=True,
            use_uncertainty=False,
            overlay_policy="none",
        ),
        "forecast_shrinkage": ExperimentSpec(
            name="forecast_shrinkage",
            label="Forecast + Shrinkage",
            allocation_mode="forecast",
            requires_forecast=True,
            use_uncertainty=True,
            overlay_policy="none",
        ),
        "static_risk_limit": ExperimentSpec(
            name="static_risk_limit",
            label="Static Risk Limit",
            allocation_mode="forecast",
            requires_forecast=True,
            use_uncertainty=True,
            overlay_policy="static",
        ),
        "full_workflow": ExperimentSpec(
            name="full_workflow",
            label="Full Workflow",
            allocation_mode="forecast",
            requires_forecast=True,
            use_uncertainty=True,
            overlay_policy="dynamic",
        ),
        "no_regime_features": ExperimentSpec(
            name="no_regime_features",
            label="No Regime Features",
            allocation_mode="forecast",
            requires_forecast=True,
            use_uncertainty=True,
            overlay_policy="dynamic",
            include_regime_features=False,
        ),
        "no_costs": ExperimentSpec(
            name="no_costs",
            label="No Costs",
            allocation_mode="forecast",
            requires_forecast=True,
            use_uncertainty=True,
            overlay_policy="dynamic",
            costs_enabled=False,
        ),
    }
    enabled = config["experiment"].get("enabled") or list(registry)
    return [registry[name] for name in enabled if name in registry]


def run_single_experiment(
    dataset: StrategyDataset,
    features: pd.DataFrame,
    config: dict[str, Any],
    spec: ExperimentSpec,
    forecast_cache: ForecastCache | None = None,
    active_sleeves_cache: ActiveSleeveCache | None = None,
    active_config_cache: ActiveConfigCache | None = None,
) -> ExperimentResult:
    returns = dataset.returns
    sleeves = get_sleeves(config)
    all_sleeves = list(sleeves)
    execution_lag = int(config["experiment"].get("execution_lag_periods", 1))
    if execution_lag < 1:
        raise ValueError("execution_lag_periods must be at least 1 to preserve t+1 execution")
    decision_dates = list(returns.index[:-execution_lag])
    warmup = int(config["experiment"].get("train_window", 84)) + int(
        config["experiment"].get("validation_window", 24)
    )
    rebalance_every = int(config["experiment"].get("rebalance_every", 1))
    decision_dates = decision_dates[warmup::rebalance_every]

    feature_cols = feature_columns(features, include_regime=spec.include_regime_features)
    model_names = list(config["forecasting"]["models"])
    if not spec.include_regime_features:
        model_names = [name for name in model_names if name != "regime_conditional_mean"]

    previous_weights = starting_weights(spec, config)
    previous_hedge_ratio = 0.0
    realized_portfolio_returns: list[float] = []
    realized_dates: list[pd.Timestamp] = []
    recent_forecast_errors: list[float] = []

    return_records: list[dict[str, Any]] = []
    weight_records: list[dict[str, Any]] = []
    forecast_records: list[dict[str, Any]] = []
    risk_records: list[dict[str, Any]] = []
    attribution_records: list[dict[str, Any]] = []
    model_records: list[dict[str, Any]] = []

    for decision_date in decision_dates:
        next_idx = returns.index.get_loc(decision_date) + execution_lag
        if next_idx >= len(returns):
            break
        return_date = returns.index[next_idx]
        active_key = (pd.Timestamp(decision_date), pd.Timestamp(return_date))
        if active_sleeves_cache is not None and active_key in active_sleeves_cache:
            active_sleeves = active_sleeves_cache[active_key]
        else:
            active_sleeves = active_sleeves_for_period(returns, decision_date, return_date, config)
            if active_sleeves_cache is not None:
                active_sleeves_cache[active_key] = active_sleeves
        if len(active_sleeves) < 2:
            continue
        config_key = tuple(active_sleeves)
        if active_config_cache is not None and config_key in active_config_cache:
            local_config = active_config_cache[config_key]
        else:
            local_config = config_for_active_sleeves(config, active_sleeves)
            if active_config_cache is not None:
                active_config_cache[config_key] = local_config
        history = returns.loc[:decision_date, active_sleeves]
        next_returns = returns.loc[return_date, active_sleeves].astype(float)

        forecasts, uncertainty, selected_model, validation_score, model_scores = baseline_forecast(history, local_config)
        if spec.requires_forecast:
            cache_key = (pd.Timestamp(decision_date), spec.include_regime_features, tuple(model_names))
            if forecast_cache is not None and cache_key in forecast_cache:
                selection = forecast_cache[cache_key]
            else:
                selection = fit_select_predict(
                    feature_frame=features,
                    decision_date=decision_date,
                    config=config,
                    feature_cols=feature_cols,
                    model_names=model_names,
                )
                if forecast_cache is not None:
                    forecast_cache[cache_key] = selection
            forecasts = selection.forecasts.reindex(active_sleeves)
            uncertainty = selection.uncertainty.reindex(active_sleeves)
            selected_model = selection.selected_model
            validation_score = selection.validation_score
            model_scores = selection.model_scores

        pre_overlay_weights = allocate_portfolio(
            mode=spec.allocation_mode,
            history=history,
            forecasts=forecasts,
            uncertainty=uncertainty,
            previous_weights=previous_weights,
            config=local_config,
            use_uncertainty=spec.use_uncertainty,
        ).reindex(active_sleeves).fillna(0.0)

        recent_error = float(np.mean(recent_forecast_errors[-6:])) if recent_forecast_errors else None
        portfolio_history = pd.Series(realized_portfolio_returns, index=realized_dates, dtype=float)
        risk_state = classify_risk_state(history, portfolio_history, local_config, recent_error)
        final_weights, overlay_action = apply_risk_overlay(
            pre_overlay_weights,
            risk_state,
            local_config,
            policy=spec.overlay_policy,
        )
        final_weights = final_weights.reindex(active_sleeves).fillna(0.0)
        full_final_weights = final_weights.reindex(all_sleeves).fillna(0.0)

        turnover = compute_turnover(full_final_weights, previous_weights)
        cost = transaction_cost(turnover, local_config, enabled=spec.costs_enabled)
        gross_return = float(final_weights @ next_returns)
        hedge_return, hedge_cost = hedge_pnl(
            float(overlay_action["hedge_ratio"]),
            next_returns,
            previous_hedge_ratio,
            local_config,
            costs_enabled=spec.costs_enabled,
        )
        net_return = gross_return + hedge_return - cost - hedge_cost
        exposure = gross_exposure(final_weights) + abs(float(overlay_action["hedge_ratio"]))

        error = float((forecasts.reindex(active_sleeves) - next_returns).abs().mean())
        if np.isfinite(error):
            recent_forecast_errors.append(error)

        return_records.append(
            {
                "experiment": spec.name,
                "information_date": decision_date,
                "decision_date": decision_date,
                "holding_period_start": decision_date,
                "holding_period_end": return_date,
                "return_date": return_date,
                "execution_lag_periods": execution_lag,
                "active_sleeve_count": len(active_sleeves),
                "gross_return": gross_return,
                "hedge_return": hedge_return,
                "transaction_cost": cost,
                "hedge_cost": hedge_cost,
                "net_return": net_return,
                "turnover": turnover,
                "gross_exposure": exposure,
                "selected_model": selected_model,
                "validation_score": validation_score,
            }
        )

        weight_row = {
            "experiment": spec.name,
            "information_date": decision_date,
            "decision_date": decision_date,
            "holding_period_start": decision_date,
            "holding_period_end": return_date,
            "return_date": return_date,
            "execution_lag_periods": execution_lag,
            "active_sleeve_count": len(active_sleeves),
            "overlay_policy": spec.overlay_policy,
            "overlay_reduction": overlay_action["reduction"],
            "hedge_ratio": overlay_action["hedge_ratio"],
        }
        weight_row.update({f"weight_{sleeve}": float(full_final_weights[sleeve]) for sleeve in all_sleeves})
        weight_records.append(weight_row)

        for sleeve in active_sleeves:
            forecast_records.append(
                {
                    "experiment": spec.name,
                    "information_date": decision_date,
                    "decision_date": decision_date,
                    "holding_period_start": decision_date,
                    "holding_period_end": return_date,
                    "return_date": return_date,
                    "execution_lag_periods": execution_lag,
                    "active_sleeve_count": len(active_sleeves),
                    "sleeve": sleeve,
                    "forecast": float(forecasts.get(sleeve, np.nan)),
                    "uncertainty": float(uncertainty.get(sleeve, np.nan)),
                    "realized_return": float(next_returns[sleeve]),
                    "selected_model": selected_model,
                }
            )

        risk_row = {
            "experiment": spec.name,
            "information_date": decision_date,
            "decision_date": decision_date,
            "holding_period_start": decision_date,
            "holding_period_end": return_date,
            "return_date": return_date,
            "execution_lag_periods": execution_lag,
            "active_sleeve_count": len(active_sleeves),
            "overlay_policy": spec.overlay_policy,
            "overlay_reduction": overlay_action["reduction"],
            "hedge_ratio": overlay_action["hedge_ratio"],
        }
        risk_row.update(risk_state)
        risk_records.append(risk_row)

        attribution_records.append(
            attribution_row(
                spec.name,
                decision_date,
                return_date,
                final_weights,
                forecasts,
                next_returns,
                gross_return,
                hedge_return,
                cost,
                hedge_cost,
                local_config,
            )
        )

        model_row = {
            "experiment": spec.name,
            "information_date": decision_date,
            "decision_date": decision_date,
            "selected_model": selected_model,
            "validation_score": validation_score,
        }
        model_row.update({f"score_{name}": float(score) for name, score in model_scores.items()})
        model_records.append(model_row)

        realized_portfolio_returns.append(net_return)
        realized_dates.append(return_date)
        previous_weights = full_final_weights
        previous_hedge_ratio = float(overlay_action["hedge_ratio"])

    returns_frame = pd.DataFrame(return_records)
    weights_frame = pd.DataFrame(weight_records)
    forecasts_frame = pd.DataFrame(forecast_records)
    risk_frame = pd.DataFrame(risk_records)
    attribution_frame = pd.DataFrame(attribution_records)
    model_frame = pd.DataFrame(model_records)

    metrics = compute_performance_metrics(
        returns_frame["net_return"],
        periods_per_year=int(config["experiment"].get("periods_per_year", 12)),
        cost=returns_frame["transaction_cost"] + returns_frame["hedge_cost"],
        turnover=returns_frame["turnover"],
        exposure=returns_frame["gross_exposure"],
    )
    metrics["avg_validation_score"] = float(returns_frame["validation_score"].replace([np.inf, -np.inf], np.nan).mean())
    metrics["avg_hedge_ratio"] = float(weights_frame.get("hedge_ratio", pd.Series(dtype=float)).mean())
    metrics["stress_period_share"] = float((risk_frame.get("risk_regime", pd.Series(dtype=str)) == "stress").mean())
    metrics["avg_active_sleeves"] = float(returns_frame.get("active_sleeve_count", pd.Series(dtype=float)).mean())

    return ExperimentResult(
        spec=spec,
        returns=returns_frame,
        weights=weights_frame,
        forecasts=forecasts_frame,
        risk_states=risk_frame,
        attribution=attribution_frame,
        model_selection=model_frame,
        metrics=metrics,
    )


def active_sleeves_for_period(
    returns: pd.DataFrame,
    decision_date: pd.Timestamp,
    return_date: pd.Timestamp,
    config: dict[str, Any],
) -> list[str]:
    sleeves = get_sleeves(config)
    if not bool(config["experiment"].get("adaptive_universe", False)):
        return sleeves

    cash = get_cash_sleeve(config)
    min_history = int(config["experiment"].get("min_sleeve_history", 1))
    history = returns.loc[:decision_date, sleeves]
    active = []
    for sleeve in sleeves:
        if sleeve == cash:
            if pd.notna(returns.loc[return_date, sleeve]):
                active.append(sleeve)
            continue
        valid_history = history[sleeve].dropna()
        has_current_nav_return = pd.notna(returns.loc[decision_date, sleeve])
        has_next_return = pd.notna(returns.loc[return_date, sleeve])
        if len(valid_history) >= min_history and has_current_nav_return and has_next_return:
            active.append(sleeve)

    if cash in sleeves and cash not in active and pd.notna(returns.loc[return_date, cash]):
        active.append(cash)
    return active


def config_for_active_sleeves(config: dict[str, Any], active_sleeves: list[str]) -> dict[str, Any]:
    local = copy.deepcopy(config)
    local["universe"]["sleeves"] = list(active_sleeves)
    strategic = pd.Series(config["universe"].get("strategic_weights", {}), dtype=float).reindex(active_sleeves)
    if strategic.isna().all() or float(strategic.fillna(0.0).sum()) <= 0.0:
        weights = {sleeve: 1.0 / len(active_sleeves) for sleeve in active_sleeves}
    else:
        strategic = strategic.fillna(0.0)
        strategic = strategic / strategic.sum()
        weights = strategic.to_dict()
    local["universe"]["strategic_weights"] = weights
    return local


def starting_weights(spec: ExperimentSpec, config: dict[str, Any]) -> pd.Series:
    if spec.allocation_mode == "equal_weight":
        return allocate_equal(get_sleeves(config))
    return strategic_weights(config)


def baseline_forecast(
    history: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.Series, pd.Series, str, float, dict[str, float]]:
    sleeves = get_sleeves(config)
    window = min(12, len(history))
    forecasts = history.tail(window).mean().reindex(sleeves).fillna(0.0)
    uncertainty = history.tail(max(12, min(len(history), 36))).std(ddof=1).reindex(sleeves)
    floor = float(config["allocation"].get("uncertainty_floor", 0.01))
    uncertainty = uncertainty.fillna(uncertainty.mean()).clip(lower=floor)
    return forecasts, uncertainty, "historical_mean", 0.0, {"historical_mean": 0.0}


def attribution_row(
    experiment: str,
    decision_date: pd.Timestamp,
    return_date: pd.Timestamp,
    weights: pd.Series,
    forecasts: pd.Series,
    realized: pd.Series,
    gross_return: float,
    hedge_return: float,
    transaction_cost_value: float,
    hedge_cost: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    sleeves = get_sleeves(config)
    neutral = strategic_weights(config).reindex(sleeves).fillna(0.0)
    equal = allocate_equal(sleeves)
    realized = realized.reindex(sleeves).fillna(0.0)
    weights = weights.reindex(sleeves).fillna(0.0)
    forecasts = forecasts.reindex(sleeves).fillna(0.0)
    forecast_error = forecasts - realized
    return {
        "experiment": experiment,
        "information_date": decision_date,
        "decision_date": decision_date,
        "holding_period_start": decision_date,
        "holding_period_end": return_date,
        "return_date": return_date,
        "strategy_selection_contribution": float(((weights - neutral) * realized).sum()),
        "allocation_contribution_vs_equal": float(gross_return - (equal @ realized)),
        "market_beta_contribution": float(weights.get("equity_beta", 0.0) * realized.get("equity_beta", 0.0)),
        "cash_contribution": float(weights.get("cash", 0.0) * realized.get("cash", 0.0)),
        "hedge_contribution": float(hedge_return - hedge_cost),
        "transaction_cost_drag": float(-transaction_cost_value),
        "mean_abs_forecast_error": float(forecast_error.abs().mean()),
        "forecast_error_directional_bias": float(forecast_error.mean()),
    }


def build_forecast_metric_table(experiments: dict[str, ExperimentResult]) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    for name, result in experiments.items():
        frame = result.forecasts
        if frame.empty:
            continue
        forecasts = frame.pivot_table(index="return_date", columns="sleeve", values="forecast")
        realized = frame.pivot_table(index="return_date", columns="sleeve", values="realized_return")
        rows[name] = forecast_error_metrics(forecasts, realized)
    return pd.DataFrame(rows).T
