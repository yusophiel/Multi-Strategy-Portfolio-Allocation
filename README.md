# Yingling Systematic Investment Research

A Python research pipeline for multi-strategy portfolio forecasting, allocation,
and risk management.

**Public demo: synthetic data only.** Company datasets, fund identifiers, and
historical research outputs are not included. Demo results are not actual fund
performance.

## Workflow

```text
Returns → Features → Walk-forward forecasts → Allocation → Risk overlay → Report
```

- Compare equal-weight, static, and inverse-volatility baselines with forecast-based portfolios.
- Evaluate uncertainty shrinkage, transaction costs, and static/dynamic risk overlays.
- Export performance metrics, attribution, and a standalone HTML dashboard.

## Quick start

Python 3.10+ is required. Run from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
make run
```

Open `outputs/default_run/dashboard.html` to view the report. The default run
uses reproducible synthetic data generated in memory; no data download is needed.
Completed experiments are cached locally. Use `--no-resume` with the `run` command
to recompute after changing settings.

```bash
make test
```

## Project structure

```text
configs/default.json       Demo and research parameters
src/
  data.py, nav_ingest.py    Synthetic data, CSV loading, Excel NAV conversion
  features.py              Rolling strategy and market-state features
  forecasting.py           Forecast models and validation
  allocation.py, risk.py   Portfolio weights and risk overlays
  backtest.py, metrics.py  Walk-forward evaluation and performance metrics
  report.py, cli.py        Reports and command-line entry points
tests/                     Allocation, metrics, ingestion, and pipeline tests
```

## Use your own data locally

Keep private files under `data/`, which is excluded from Git. The NAV converter
accepts Excel files with columns such as `日期`, `单位净值`, and `累计净值`:

```bash
python3 -m src prepare-nav-data --input data/private
python3 -m src run --config data/processed/nav_config.json --output outputs/local_run
```

For CSV inputs, copy `configs/default.json` to `data/local_config.json`, set
`data.returns_csv`, and update `universe.sleeves` and `universe.strategic_weights`
to match your columns. Returns must be decimal values, with a `date` column;
`data.macro_csv` is optional. Set `experiment.periods_per_year` to match the data
frequency. See `prepare-nav-data --help` for weekly and staggered-history options.

Generated reports include copies of input data. Keep `data/`, `outputs/`, and
private configurations local; do not force-add them to Git or upload them manually.
