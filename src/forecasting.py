"""Rolling forecast models and validation-based model selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .metrics import mean_rank_ic


class ForecastModel(Protocol):
    name: str

    def fit(self, frame: pd.DataFrame, feature_cols: list[str]) -> "ForecastModel":
        ...

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        ...


@dataclass
class ForecastSelectionResult:
    decision_date: pd.Timestamp
    selected_model: str
    validation_score: float
    forecasts: pd.Series
    uncertainty: pd.Series
    model_scores: dict[str, float]


class HistoricalMeanModel:
    name = "historical_mean"

    def fit(self, frame: pd.DataFrame, feature_cols: list[str]) -> "HistoricalMeanModel":
        clean = frame.dropna(subset=["target_next_return"])
        self.global_mean = float(clean["target_next_return"].mean()) if not clean.empty else 0.0
        self.means = clean.groupby("sleeve")["target_next_return"].mean().to_dict()
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        return frame["sleeve"].map(self.means).fillna(self.global_mean).astype(float)


class MomentumRuleModel:
    name = "momentum_rule"

    def __init__(self, window: int = 6) -> None:
        self.window = window

    def fit(self, frame: pd.DataFrame, feature_cols: list[str]) -> "MomentumRuleModel":
        clean = frame.dropna(subset=["target_next_return"])
        self.mean_by_sleeve = clean.groupby("sleeve")["target_next_return"].mean().to_dict()
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        col = f"strategy_ret_{self.window}m"
        if col not in frame.columns:
            return frame["sleeve"].map(self.mean_by_sleeve).fillna(0.0).astype(float)
        raw = pd.to_numeric(frame[col], errors="coerce").fillna(0.0) / max(self.window, 1)
        sleeve_mean = frame["sleeve"].map(self.mean_by_sleeve).fillna(0.0)
        return (0.35 * raw + 0.65 * sleeve_mean).astype(float)


class ReversalRuleModel:
    name = "reversal_rule"

    def __init__(self, window: int = 3) -> None:
        self.window = window

    def fit(self, frame: pd.DataFrame, feature_cols: list[str]) -> "ReversalRuleModel":
        clean = frame.dropna(subset=["target_next_return"])
        self.mean_by_sleeve = clean.groupby("sleeve")["target_next_return"].mean().to_dict()
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        col = f"strategy_ret_{self.window}m"
        if col not in frame.columns:
            return frame["sleeve"].map(self.mean_by_sleeve).fillna(0.0).astype(float)
        raw = -pd.to_numeric(frame[col], errors="coerce").fillna(0.0) / max(self.window, 1)
        sleeve_mean = frame["sleeve"].map(self.mean_by_sleeve).fillna(0.0)
        return (0.25 * raw + 0.75 * sleeve_mean).astype(float)


class LinearRegressionModel:
    name = "linear_regression"

    def __init__(self, ridge_alpha: float = 0.0) -> None:
        self.ridge_alpha = ridge_alpha

    def fit(self, frame: pd.DataFrame, feature_cols: list[str]) -> "LinearRegressionModel":
        clean = frame.dropna(subset=["target_next_return"]).copy()
        self.feature_cols = list(feature_cols)
        self.sleeves = sorted(clean["sleeve"].dropna().unique().tolist())
        x = self._design(clean, fit=True)
        y = clean["target_next_return"].astype(float).values
        penalty = np.eye(x.shape[1]) * float(self.ridge_alpha)
        penalty[0, 0] = 0.0
        try:
            self.coef = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        except np.linalg.LinAlgError:
            self.coef = np.linalg.pinv(x.T @ x + penalty) @ x.T @ y
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        x = self._design(frame, fit=False)
        return pd.Series(x @ self.coef, index=frame.index, dtype=float)

    def _design(self, frame: pd.DataFrame, fit: bool) -> np.ndarray:
        numeric = frame.reindex(columns=self.feature_cols).apply(pd.to_numeric, errors="coerce")
        if fit:
            self.means = numeric.mean().replace([np.inf, -np.inf], np.nan).fillna(0.0)
            self.stds = numeric.std(ddof=0).replace([0.0, np.inf, -np.inf], np.nan).fillna(1.0)
        numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(self.means)
        numeric = (numeric - self.means) / self.stds
        sleeve_dummies = pd.get_dummies(frame["sleeve"], prefix="sleeve").reindex(
            columns=[f"sleeve_{name}" for name in self.sleeves],
            fill_value=0.0,
        )
        intercept = np.ones((len(frame), 1))
        return np.hstack([intercept, numeric.values.astype(float), sleeve_dummies.values.astype(float)])


class RidgeRegressionModel(LinearRegressionModel):
    name = "ridge_regression"


class SparseLinearModel(LinearRegressionModel):
    name = "sparse_linear"

    def __init__(self, ridge_alpha: float = 6.0, sparse_alpha: float = 0.01) -> None:
        super().__init__(ridge_alpha=ridge_alpha)
        self.sparse_alpha = sparse_alpha

    def fit(self, frame: pd.DataFrame, feature_cols: list[str]) -> "SparseLinearModel":
        super().fit(frame, feature_cols)
        threshold = float(self.sparse_alpha)
        shrunk = np.sign(self.coef) * np.maximum(np.abs(self.coef) - threshold, 0.0)
        shrunk[0] = self.coef[0]
        self.coef = shrunk
        return self


class RegimeConditionalMeanModel:
    name = "regime_conditional_mean"

    def fit(self, frame: pd.DataFrame, feature_cols: list[str]) -> "RegimeConditionalMeanModel":
        clean = frame.dropna(subset=["target_next_return"]).copy()
        self.global_mean = float(clean["target_next_return"].mean()) if not clean.empty else 0.0
        self.sleeve_mean = clean.groupby("sleeve")["target_next_return"].mean().to_dict()
        self.group_mean = clean.groupby(["sleeve", "risk_regime"])["target_next_return"].mean().to_dict()
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        values = []
        for _, row in frame.iterrows():
            key = (row["sleeve"], row.get("risk_regime", "normal"))
            values.append(self.group_mean.get(key, self.sleeve_mean.get(row["sleeve"], self.global_mean)))
        return pd.Series(values, index=frame.index, dtype=float)


class TreeStumpModel:
    name = "tree_stump"

    def fit(self, frame: pd.DataFrame, feature_cols: list[str]) -> "TreeStumpModel":
        clean = frame.dropna(subset=["target_next_return"]).copy()
        self.global_mean = float(clean["target_next_return"].mean()) if not clean.empty else 0.0
        self.sleeve_mean = clean.groupby("sleeve")["target_next_return"].mean().to_dict()
        self.feature = None
        self.threshold = 0.0
        self.left = {}
        self.right = {}

        best_mse = float("inf")
        candidate_cols = feature_cols[:40]
        for col in candidate_cols:
            values = pd.to_numeric(clean[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if values.notna().sum() < 20 or values.nunique(dropna=True) < 3:
                continue
            threshold = float(values.median())
            branch = pd.Series(np.where(values <= threshold, "left", "right"), index=clean.index, name="_branch")
            means = clean.assign(_branch=branch).groupby(["sleeve", "_branch"])["target_next_return"].mean()
            keys = pd.MultiIndex.from_arrays([clean["sleeve"].values, branch.values])
            pred = means.reindex(keys).to_numpy(dtype=float)
            fallback = clean["sleeve"].map(self.sleeve_mean).fillna(self.global_mean).to_numpy(dtype=float)
            pred = np.where(np.isnan(pred), fallback, pred)
            mse = float(np.mean((clean["target_next_return"].to_numpy(dtype=float) - pred) ** 2))
            if mse < best_mse:
                best_mse = mse
                self.feature = col
                self.threshold = threshold
                self.left = means.xs("left", level="_branch", drop_level=True).to_dict() if "left" in means.index.get_level_values("_branch") else {}
                self.right = means.xs("right", level="_branch", drop_level=True).to_dict() if "right" in means.index.get_level_values("_branch") else {}
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        if self.feature is None or self.feature not in frame.columns:
            return frame["sleeve"].map(self.sleeve_mean).fillna(self.global_mean).astype(float)
        values = pd.to_numeric(frame[self.feature], errors="coerce").fillna(self.threshold)
        out = []
        for sleeve, value in zip(frame["sleeve"], values):
            table = self.left if value <= self.threshold else self.right
            out.append(table.get(sleeve, self.sleeve_mean.get(sleeve, self.global_mean)))
        return pd.Series(out, index=frame.index, dtype=float)


def make_model(name: str, config: dict[str, Any]) -> ForecastModel:
    fcfg = config["forecasting"]
    if name == "historical_mean":
        return HistoricalMeanModel()
    if name == "momentum_rule":
        return MomentumRuleModel(window=int(fcfg.get("momentum_window", 6)))
    if name == "reversal_rule":
        return ReversalRuleModel(window=int(fcfg.get("reversal_window", 3)))
    if name == "linear_regression":
        return LinearRegressionModel(ridge_alpha=1e-8)
    if name == "ridge_regression":
        return RidgeRegressionModel(ridge_alpha=float(fcfg.get("ridge_alpha", 8.0)))
    if name == "sparse_linear":
        return SparseLinearModel(
            ridge_alpha=float(fcfg.get("ridge_alpha", 8.0)),
            sparse_alpha=float(fcfg.get("sparse_alpha", 0.015)),
        )
    if name == "regime_conditional_mean":
        return RegimeConditionalMeanModel()
    if name == "tree_stump":
        return TreeStumpModel()
    raise ValueError(f"Unknown forecast model: {name}")


def fit_select_predict(
    feature_frame: pd.DataFrame,
    decision_date: pd.Timestamp,
    config: dict[str, Any],
    feature_cols: list[str],
    model_names: list[str] | None = None,
) -> ForecastSelectionResult:
    exp_cfg = config["experiment"]
    model_names = model_names or list(config["forecasting"]["models"])
    decision_date = pd.Timestamp(decision_date)

    next_dates = pd.to_datetime(feature_frame["next_date"], errors="coerce")
    history_mask = next_dates.notna() & (next_dates <= decision_date)
    available_dates = sorted(
        pd.Timestamp(date) for date in feature_frame.loc[history_mask, "date"].unique()
    )
    validation_window = int(exp_cfg.get("validation_window", 24))
    train_window = int(exp_cfg.get("train_window", 84))
    if len(available_dates) < validation_window + 6:
        raise ValueError(f"Not enough history before {decision_date.date()} for walk-forward model selection")

    val_dates = available_dates[-validation_window:]
    train_dates = available_dates[:-validation_window]
    if exp_cfg.get("walk_forward_mode", "rolling") == "rolling":
        train_dates = train_dates[-train_window:]

    safe_frame = feature_frame[history_mask]
    train_df = safe_frame[safe_frame["date"].isin(train_dates)].dropna(subset=["target_next_return"])
    val_df = safe_frame[safe_frame["date"].isin(val_dates)].dropna(subset=["target_next_return"])
    min_rows = int(exp_cfg.get("min_training_rows", 60))
    if len(train_df) < min_rows:
        raise ValueError(
            f"Only {len(train_df)} training rows before {decision_date.date()}, need at least {min_rows}"
        )

    model_scores: dict[str, float] = {}
    fitted_models: dict[str, ForecastModel] = {}
    for name in model_names:
        model = make_model(name, config).fit(train_df, feature_cols)
        fitted_models[name] = model
        val_pred = prediction_frame(val_df, model.predict(val_df))
        val_realized = val_df.pivot(index="date", columns="sleeve", values="target_next_return")
        score = mean_rank_ic(val_pred, val_realized)
        if not np.isfinite(score):
            err = val_pred.align(val_realized, join="inner", axis=1)
            score = -float((err[0] - err[1]).abs().stack().mean())
        model_scores[name] = score

    selected_name = max(model_scores, key=lambda key: (model_scores[key], -model_names.index(key)))
    final_df = pd.concat([train_df, val_df], axis=0)
    final_model = make_model(selected_name, config).fit(final_df, feature_cols)
    pred_rows = feature_frame[feature_frame["date"] == decision_date].copy()
    predictions = final_model.predict(pred_rows)
    forecasts = pd.Series(predictions.values, index=pred_rows["sleeve"].values, dtype=float).sort_index()

    insample_pred = final_model.predict(final_df)
    uncertainty = residual_uncertainty(final_df, insample_pred, config).reindex(forecasts.index)
    floor = float(config["allocation"].get("uncertainty_floor", 0.01))
    uncertainty = uncertainty.fillna(max(floor, uncertainty.mean(skipna=True))).clip(lower=floor)
    return ForecastSelectionResult(
        decision_date=decision_date,
        selected_model=selected_name,
        validation_score=float(model_scores[selected_name]),
        forecasts=forecasts,
        uncertainty=uncertainty,
        model_scores=model_scores,
    )


def prediction_frame(rows: pd.DataFrame, predictions: pd.Series) -> pd.DataFrame:
    temp = rows.loc[:, ["date", "sleeve"]].copy()
    temp["prediction"] = predictions.values
    return temp.pivot(index="date", columns="sleeve", values="prediction")


def residual_uncertainty(frame: pd.DataFrame, predictions: pd.Series, config: dict[str, Any]) -> pd.Series:
    temp = frame.loc[:, ["sleeve", "target_next_return"]].copy()
    temp["prediction"] = predictions.values
    temp["residual"] = temp["target_next_return"] - temp["prediction"]
    by_sleeve = temp.groupby("sleeve")["residual"].std(ddof=1)
    if by_sleeve.isna().all():
        by_sleeve = pd.Series(float(temp["residual"].std(ddof=1)), index=by_sleeve.index)
    floor = float(config["allocation"].get("uncertainty_floor", 0.01))
    return by_sleeve.fillna(float(temp["residual"].std(ddof=1))).clip(lower=floor)
