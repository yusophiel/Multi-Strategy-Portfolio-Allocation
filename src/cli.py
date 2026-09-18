"""Command line interface for the research pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .backtest import run_experiment_suite
from .config import deep_update, load_config
from .data import generate_synthetic_dataset
from .nav_ingest import prepare_nav_return_panel
from .report import write_suite_outputs


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="yingling-research")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full walk-forward research suite.")
    run_parser.add_argument("--config", type=str, default=None, help="Path to a JSON config override.")
    run_parser.add_argument("--output", type=str, default="outputs/default_run", help="Output directory.")
    run_parser.add_argument("--no-resume", action="store_true", help="Ignore saved experiment checkpoints.")

    data_parser = subparsers.add_parser("make-sample-data", help="Write deterministic sample input CSV files.")
    data_parser.add_argument("--config", type=str, default=None, help="Path to a JSON config override.")
    data_parser.add_argument("--output", type=str, default="data/raw", help="Output directory.")

    nav_parser = subparsers.add_parser("prepare-nav-data", help="Convert fund NAV Excel files to a return panel.")
    nav_parser.add_argument("--input", type=str, required=True, help="Folder containing fund NAV Excel files.")
    nav_parser.add_argument("--output", type=str, default="data/processed/fof_nav_monthly_returns.csv")
    nav_parser.add_argument("--metadata-output", type=str, default="data/processed/fof_nav_metadata.csv")
    nav_parser.add_argument("--skipped-output", type=str, default="data/processed/fof_nav_skipped.csv")
    nav_parser.add_argument("--config-output", type=str, default="data/processed/nav_config.json")
    nav_parser.add_argument("--frequency", choices=["monthly", "weekly"], default="monthly")
    nav_parser.add_argument("--min-observations", type=int, default=36)
    nav_parser.add_argument("--min-sleeve-history", type=int, default=None, help="Minimum observations before a fund is investable in adaptive mode.")
    nav_parser.add_argument("--cash-return", type=float, default=None, help="Periodic cash return; defaults by frequency.")
    nav_parser.add_argument("--keep-missing", action="store_true", help="Keep staggered histories instead of common dates.")

    args = parser.parse_args(argv)
    if args.command == "run":
        cmd_run(args.config, args.output, resume=not args.no_resume)
    elif args.command == "make-sample-data":
        cmd_make_sample_data(args.config, args.output)
    elif args.command == "prepare-nav-data":
        cmd_prepare_nav_data(args)


def cmd_run(config_path: str | None, output_dir: str, resume: bool = True) -> None:
    config = load_config(config_path)
    checkpoint_dir = Path(output_dir).expanduser() / "_checkpoints" if resume else None
    suite = run_experiment_suite(config, progress=True, checkpoint_dir=checkpoint_dir)
    paths = write_suite_outputs(suite, output_dir)
    print(f"Completed {len(suite.experiments)} experiments.")
    print(f"Dashboard: {paths['dashboard']}")
    print(f"Summary: {paths['summary']}")


def cmd_make_sample_data(config_path: str | None, output_dir: str) -> None:
    config = load_config(config_path)
    dataset = generate_synthetic_dataset(config)
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    returns_path = out / "sample_strategy_returns.csv"
    macro_path = out / "sample_macro.csv"
    dataset.returns.to_csv(returns_path)
    dataset.macro.to_csv(macro_path)
    print(f"Sample returns: {returns_path}")
    print(f"Sample macro: {macro_path}")


def cmd_prepare_nav_data(args: argparse.Namespace) -> None:
    result = prepare_nav_return_panel(
        input_dir=args.input,
        frequency=args.frequency,
        min_observations=args.min_observations,
        drop_missing=not args.keep_missing,
        cash_return=default_cash_return(args.frequency) if args.cash_return is None else args.cash_return,
    )
    returns_path = Path(args.output).expanduser()
    metadata_path = Path(args.metadata_output).expanduser()
    skipped_path = Path(args.skipped_output).expanduser()
    config_path = Path(args.config_output).expanduser()
    for path in [returns_path, metadata_path, skipped_path, config_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    result.returns.to_csv(returns_path)
    result.metadata.to_csv(metadata_path, index=False)
    result.skipped.to_csv(skipped_path, index=False)

    periods_per_year = 12 if args.frequency == "monthly" else 52
    observation_count = max(len(result.returns), 1)
    train_window = min(24 if args.frequency == "monthly" else 104, max(6, observation_count // 2))
    validation_window = min(6 if args.frequency == "monthly" else 26, max(3, observation_count // 5))
    adaptive_universe = bool(args.keep_missing)
    if adaptive_universe:
        min_training_rows = max(80, int(train_window * 2.5))
    else:
        min_training_rows = max(30, int(train_window * len(result.returns.columns) * 0.55))
    min_sleeve_history = args.min_sleeve_history
    if min_sleeve_history is None:
        min_sleeve_history = 12 if args.frequency == "weekly" else 6
    sleeves = list(result.returns.columns)
    equal_weights = {col: 1.0 / len(sleeves) for col in sleeves}
    config = deep_update(
        load_config(None),
        {
            "data": {
                "returns_csv": str(returns_path),
                "macro_csv": None,
                "frequency": "M" if args.frequency == "monthly" else "W-FRI",
            },
            "universe": {
                "sleeves": sleeves,
                "cash_sleeve": "cash",
                "strategic_weights": equal_weights,
            },
            "experiment": {
                "periods_per_year": periods_per_year,
                "train_window": train_window,
                "validation_window": validation_window,
                "execution_lag_periods": 1,
                "adaptive_universe": adaptive_universe,
                "min_sleeve_history": min_sleeve_history,
                "min_training_rows": min_training_rows,
                "enabled": real_data_experiments(),
            },
            "features": {
                "risk_window": min(12 if args.frequency == "monthly" else 26, max(4, train_window // 2)),
                "covariance_window": min(24 if args.frequency == "monthly" else 52, max(6, train_window)),
                "regime_min_periods": min(24 if args.frequency == "monthly" else 52, max(6, train_window)),
            },
        },
    )
    config["universe"] = {
        "sleeves": sleeves,
        "cash_sleeve": "cash",
        "strategic_weights": equal_weights,
    }
    config["report"]["title"] = "Yingling Real FOF NAV Research"
    with config_path.open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, ensure_ascii=False)

    print(f"Return panel: {returns_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Skipped files: {skipped_path}")
    print(f"Run config: {config_path}")
    print(f"Selected funds: {len(result.returns.columns)}")
    print(f"Return periods: {len(result.returns)}")


def default_cash_return(frequency: str) -> float:
    return 0.001 if frequency == "monthly" else 0.00023


def real_data_experiments() -> list[str]:
    return [
        "equal_weight",
        "static_allocation",
        "inverse_volatility",
        "equal_weight_overlay",
        "inverse_volatility_overlay",
        "forecast_allocation",
        "forecast_shrinkage",
        "static_risk_limit",
        "full_workflow",
        "no_regime_features",
        "no_costs",
    ]
