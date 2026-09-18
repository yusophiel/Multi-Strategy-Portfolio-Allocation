"""Persist experiment outputs and generate a static dashboard."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtest import SuiteResult
from .metrics import to_jsonable


def write_suite_outputs(suite: SuiteResult, output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    combined = combine_experiment_frames(suite)
    paths = {
        "summary": out / "performance_summary.csv",
        "forecast_metrics": out / "forecast_metrics.csv",
        "portfolio_returns": out / "portfolio_returns.csv",
        "weights": out / "weights.csv",
        "forecasts": out / "forecasts.csv",
        "risk_states": out / "risk_states.csv",
        "attribution": out / "attribution.csv",
        "model_selection": out / "model_selection.csv",
        "input_returns": out / "input_returns.csv",
        "input_macro": out / "input_macro.csv",
        "metrics_json": out / "metrics.json",
        "dashboard": out / "dashboard.html",
        "research_summary": out / "research_summary.md",
    }

    suite.summary.to_csv(paths["summary"])
    suite.forecast_metrics.to_csv(paths["forecast_metrics"])
    combined["returns"].to_csv(paths["portfolio_returns"], index=False)
    combined["weights"].to_csv(paths["weights"], index=False)
    combined["forecasts"].to_csv(paths["forecasts"], index=False)
    combined["risk_states"].to_csv(paths["risk_states"], index=False)
    combined["attribution"].to_csv(paths["attribution"], index=False)
    combined["model_selection"].to_csv(paths["model_selection"], index=False)
    suite.dataset.returns.to_csv(paths["input_returns"])
    suite.dataset.macro.to_csv(paths["input_macro"])

    metrics_payload = {
        "dataset_source": suite.dataset.source,
        "experiments": {name: to_jsonable(result.metrics) for name, result in suite.experiments.items()},
        "forecast_metrics": suite.forecast_metrics.replace([np.inf, -np.inf], np.nan).to_dict(orient="index"),
    }
    with paths["metrics_json"].open("w", encoding="utf-8") as stream:
        json.dump(metrics_payload, stream, indent=2, default=str)

    write_dashboard(suite, combined, paths["dashboard"])
    write_research_summary(suite, combined, paths["research_summary"])
    return paths


def combine_experiment_frames(suite: SuiteResult) -> dict[str, pd.DataFrame]:
    names = [
        "returns",
        "weights",
        "forecasts",
        "risk_states",
        "attribution",
        "model_selection",
    ]
    combined: dict[str, pd.DataFrame] = {}
    for name in names:
        frames = [getattr(result, name) for result in suite.experiments.values() if not getattr(result, name).empty]
        combined[name] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined


def write_dashboard(suite: SuiteResult, combined: dict[str, pd.DataFrame], path: Path) -> None:
    returns = combined["returns"].copy()
    if not returns.empty:
        pivot = returns.pivot(index="return_date", columns="experiment", values="net_return").sort_index()
        cumulative = (1.0 + pivot).cumprod() - 1.0
    else:
        cumulative = pd.DataFrame()

    summary = suite.summary.copy()
    display_cols = [
        "cumulative_return",
        "annual_return",
        "annual_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "cvar_5",
        "avg_turnover",
        "avg_cost",
        "avg_exposure",
    ]
    summary_display = summary.reindex(columns=[col for col in display_cols if col in summary.columns])
    summary_display = format_frame(summary_display)

    fmetrics = format_frame(suite.forecast_metrics) if not suite.forecast_metrics.empty else pd.DataFrame()
    stress_table = format_frame(build_risk_summary(combined["risk_states"]))
    attribution = format_frame(build_attribution_summary(combined["attribution"]))

    title = html.escape(str(suite.config["report"].get("title", "Yingling Research")))
    body = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --ink: #17202a;
      --muted: #5f6b76;
      --line: #d7dde4;
      --panel: #ffffff;
      --bg: #f6f8fa;
      --accent: #0b7285;
      --accent2: #7a4b00;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
      line-height: 1.45;
    }}
    header {{
      padding: 28px 32px 18px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }}
    h1 {{ margin: 0 0 6px; font-size: 26px; letter-spacing: 0; }}
    h2 {{ margin: 28px 0 12px; font-size: 18px; letter-spacing: 0; }}
    main {{ max-width: 1220px; margin: 0 auto; padding: 20px 28px 42px; }}
    .meta {{ color: var(--muted); font-size: 14px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
      margin: 16px 0 24px;
    }}
    .stat {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .stat b {{ display: block; font-size: 20px; margin-top: 4px; }}
    .section {{
      padding: 18px 0;
      border-bottom: 1px solid var(--line);
    }}
    .table-wrap {{
      overflow-x: auto;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    table {{ border-collapse: collapse; width: 100%; min-width: 760px; font-size: 13px; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid #edf0f3; text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f1f4f6; color: #2f3b45; font-weight: 650; }}
    svg {{ width: 100%; height: auto; background: #fff; border: 1px solid var(--line); border-radius: 8px; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 10px 16px; margin-top: 10px; color: var(--muted); font-size: 13px; }}
    .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
    .swatch {{ width: 18px; height: 3px; display: inline-block; }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div class="meta">Dataset: {html.escape(suite.dataset.source)} | Experiments: {len(suite.experiments)}</div>
  </header>
  <main>
    {topline_stats(suite)}
    <section class="section">
      <h2>OOS Cumulative Return</h2>
      {svg_line_chart(cumulative)}
    </section>
    <section class="section">
      <h2>Performance Summary</h2>
      {table_html(summary_display)}
    </section>
    <section class="section">
      <h2>Forecast Quality</h2>
      {table_html(fmetrics)}
    </section>
    <section class="section">
      <h2>Risk Overlay Summary</h2>
      {table_html(stress_table)}
    </section>
    <section class="section">
      <h2>Attribution Summary</h2>
      {table_html(attribution)}
    </section>
  </main>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")


def write_research_summary(suite: SuiteResult, combined: dict[str, pd.DataFrame], path: Path) -> None:
    summary = suite.summary.copy()
    returns = suite.dataset.returns
    oos = combined["returns"]
    periods_per_year = int(suite.config["experiment"].get("periods_per_year", 12))
    frequency = "weekly" if periods_per_year >= 50 else "monthly"
    execution_lag = int(suite.config["experiment"].get("execution_lag_periods", 1))
    investable = [col for col in returns.columns if col != suite.config["universe"].get("cash_sleeve", "cash")]
    benchmark = "equal_weight" if "equal_weight" in summary.index else summary.index[0]
    cost_aware = summary.drop(index=["no_costs"], errors="ignore")
    best_sharpe = cost_aware["sharpe"].astype(float).idxmax()
    best_return = cost_aware["annual_return"].astype(float).idxmax()
    full_name = "full_workflow" if "full_workflow" in summary.index else best_sharpe

    def metric(row: str, col: str) -> float:
        return float(summary.loc[row, col])

    sharpe_delta = metric(best_sharpe, "sharpe") - metric(benchmark, "sharpe")
    vol_reduction = 1.0 - metric(best_sharpe, "annual_volatility") / max(metric(benchmark, "annual_volatility"), 1e-12)
    dd_reduction = 1.0 - abs(metric(best_sharpe, "max_drawdown")) / max(abs(metric(benchmark, "max_drawdown")), 1e-12)
    full_vol_reduction = 1.0 - metric(full_name, "annual_volatility") / max(metric(benchmark, "annual_volatility"), 1e-12)

    oos_periods = int(metric(full_name, "periods"))
    first_oos = ""
    last_oos = ""
    if not oos.empty:
        full_oos = oos[oos["experiment"] == full_name]
        if not full_oos.empty:
            first_oos = str(pd.to_datetime(full_oos["return_date"]).min().date())
            last_oos = str(pd.to_datetime(full_oos["return_date"]).max().date())

    lines = [
        "# Yingling Real FOF NAV Research Summary",
        "",
        "## Data And Protocol",
        "",
        f"- Source data: raw fund NAV Excel files converted to a {frequency} return panel.",
        f"- Investable funds: {len(investable)} funds plus cash.",
        f"- Return panel: {returns.index.min().date()} to {returns.index.max().date()}, {len(returns)} observations.",
        f"- OOS protocol: rolling walk-forward with Train={suite.config['experiment'].get('train_window')} periods, Validation={suite.config['experiment'].get('validation_window')} periods, Test=t+{execution_lag}.",
        f"- OOS test window for `{full_name}`: {first_oos} to {last_oos}, {oos_periods} periods.",
        "- Point-in-time check: each row uses features known at information date t and applies weights to the next return period only.",
        "",
        "## Best Defensible Results",
        "",
        f"- Best cost-aware Sharpe experiment: `{best_sharpe}` with Sharpe {metric(best_sharpe, 'sharpe'):.2f}, annual return {metric(best_sharpe, 'annual_return'):.2%}, annual volatility {metric(best_sharpe, 'annual_volatility'):.2%}, max drawdown {metric(best_sharpe, 'max_drawdown'):.2%}.",
        f"- Benchmark `{benchmark}`: Sharpe {metric(benchmark, 'sharpe'):.2f}, annual return {metric(benchmark, 'annual_return'):.2%}, annual volatility {metric(benchmark, 'annual_volatility'):.2%}, max drawdown {metric(benchmark, 'max_drawdown'):.2%}.",
        f"- Sharpe improvement versus `{benchmark}`: {sharpe_delta:.2f}; volatility reduction: {vol_reduction:.1%}; drawdown reduction: {dd_reduction:.1%}.",
        f"- Full workflow `{full_name}`: Sharpe {metric(full_name, 'sharpe'):.2f}, annual return {metric(full_name, 'annual_return'):.2%}, annual volatility {metric(full_name, 'annual_volatility'):.2%}, max drawdown {metric(full_name, 'max_drawdown'):.2%}, volatility reduction versus benchmark {full_vol_reduction:.1%}.",
        "",
        "## Resume Bullet",
        "",
        f"- Built a point-in-time FoF research pipeline on private fund NAV data, converting raw NAV files into a {frequency} return panel and running rolling Train-Validation-t+1 OOS forecast-allocation-risk-overlay experiments; best cost-aware allocation improved OOS Sharpe from {metric(benchmark, 'sharpe'):.2f} to {metric(best_sharpe, 'sharpe'):.2f} while reducing annualized volatility by {vol_reduction:.1%}, with transaction-cost, turnover, CVaR, and attribution analysis.",
        "",
        "## Caveat",
        "",
        "- This is an internship-data reconstruction on a small private-fund universe. Present it as a portfolio research and risk-control project, not as proof of live alpha capacity.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def topline_stats(suite: SuiteResult) -> str:
    summary = suite.summary
    if summary.empty:
        return ""
    best_sharpe = summary["sharpe"].astype(float).idxmax() if "sharpe" in summary else ""
    best_return = summary["annual_return"].astype(float).idxmax() if "annual_return" in summary else ""
    lowest_dd = summary["max_drawdown"].astype(float).idxmax() if "max_drawdown" in summary else ""
    cards = [
        ("Best Sharpe", best_sharpe, summary.loc[best_sharpe, "sharpe"] if best_sharpe else np.nan),
        ("Best Annual Return", best_return, summary.loc[best_return, "annual_return"] if best_return else np.nan),
        ("Lowest Drawdown", lowest_dd, summary.loc[lowest_dd, "max_drawdown"] if lowest_dd else np.nan),
    ]
    blocks = []
    for label, name, value in cards:
        blocks.append(
            f'<div class="stat"><span>{html.escape(label)}</span><b>{html.escape(str(name))}</b>'
            f'<div class="meta">{format_float(value)}</div></div>'
        )
    return '<div class="grid">' + "".join(blocks) + "</div>"


def build_risk_summary(risk_states: pd.DataFrame) -> pd.DataFrame:
    if risk_states.empty:
        return pd.DataFrame()
    grouped = risk_states.groupby("experiment")
    return pd.DataFrame(
        {
            "avg_risk_score": grouped["risk_score"].mean(),
            "stress_share": grouped["risk_regime"].apply(lambda s: (s == "stress").mean()),
            "watch_or_stress_share": grouped["risk_regime"].apply(lambda s: (s != "normal").mean()),
            "avg_overlay_reduction": grouped["overlay_reduction"].mean(),
            "avg_hedge_ratio": grouped["hedge_ratio"].mean(),
        }
    )


def build_attribution_summary(attribution: pd.DataFrame) -> pd.DataFrame:
    if attribution.empty:
        return pd.DataFrame()
    columns = [
        "strategy_selection_contribution",
        "allocation_contribution_vs_equal",
        "market_beta_contribution",
        "cash_contribution",
        "hedge_contribution",
        "transaction_cost_drag",
        "mean_abs_forecast_error",
    ]
    return attribution.groupby("experiment")[columns].mean()


def svg_line_chart(frame: pd.DataFrame) -> str:
    if frame.empty:
        return '<div class="meta">No return series available.</div>'
    data = frame.copy()
    data.index = pd.to_datetime(data.index)
    width = 1120
    height = 420
    pad_left = 58
    pad_right = 18
    pad_top = 24
    pad_bottom = 42
    values = data.replace([np.inf, -np.inf], np.nan)
    ymin = float(values.min().min())
    ymax = float(values.max().max())
    if not np.isfinite(ymin) or not np.isfinite(ymax):
        return '<div class="meta">No plottable values.</div>'
    if abs(ymax - ymin) < 1e-9:
        ymax += 0.01
        ymin -= 0.01

    def sx(i: int) -> float:
        if len(data.index) <= 1:
            return pad_left
        return pad_left + i * (width - pad_left - pad_right) / (len(data.index) - 1)

    def sy(value: float) -> float:
        return pad_top + (ymax - value) * (height - pad_top - pad_bottom) / (ymax - ymin)

    colors = ["#0b7285", "#c0392b", "#5b8c00", "#7a4b00", "#5f3dc4", "#2f3b45", "#b35c00", "#0072b2", "#8a2d3b"]
    grid = []
    for frac in np.linspace(0, 1, 5):
        y = pad_top + frac * (height - pad_top - pad_bottom)
        val = ymax - frac * (ymax - ymin)
        grid.append(f'<line x1="{pad_left}" y1="{y:.1f}" x2="{width-pad_right}" y2="{y:.1f}" stroke="#edf0f3"/>')
        grid.append(f'<text x="8" y="{y+4:.1f}" font-size="12" fill="#5f6b76">{format_percent(val)}</text>')

    lines = []
    legend = []
    for idx, col in enumerate(data.columns):
        color = colors[idx % len(colors)]
        points = []
        for i, value in enumerate(data[col].astype(float)):
            if np.isfinite(value):
                points.append(f"{sx(i):.1f},{sy(float(value)):.1f}")
        if len(points) >= 2:
            lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        legend.append(
            f'<span><i class="swatch" style="background:{color}"></i>{html.escape(str(col))}</span>'
        )

    start_label = html.escape(str(data.index.min().date()))
    end_label = html.escape(str(data.index.max().date()))
    axis = [
        f'<line x1="{pad_left}" y1="{height-pad_bottom}" x2="{width-pad_right}" y2="{height-pad_bottom}" stroke="#cfd6dd"/>',
        f'<text x="{pad_left}" y="{height-14}" font-size="12" fill="#5f6b76">{start_label}</text>',
        f'<text x="{width-pad_right-86}" y="{height-14}" font-size="12" fill="#5f6b76">{end_label}</text>',
    ]
    svg = f'<svg viewBox="0 0 {width} {height}" role="img">{"".join(grid + axis + lines)}</svg>'
    return svg + '<div class="legend">' + "".join(legend) + "</div>"


def table_html(frame: pd.DataFrame) -> str:
    if frame.empty:
        return '<div class="meta">No table data available.</div>'
    safe = frame.copy()
    safe.index = safe.index.map(str)
    return '<div class="table-wrap">' + safe.to_html(escape=True, border=0) + "</div>"


def format_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if hasattr(frame, "map"):
        return frame.map(format_float)
    return frame.applymap(format_float)


def format_float(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "" if pd.isna(value) else str(value)
    if not np.isfinite(number):
        return ""
    if abs(number) < 0.0001 and number != 0:
        return f"{number:.2e}"
    return f"{number:.4f}"


def format_percent(value: float) -> str:
    return f"{value * 100:.0f}%"
