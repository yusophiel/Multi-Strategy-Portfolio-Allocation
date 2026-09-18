"""Convert fund NAV Excel files into a return panel usable by the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import pandas as pd


DATE_NAMES = {"date", "nav_date", "datetime", "valuation_date"}
NAV_PRIORITY = ["复权累计净值", "累计净值", "单位净值", "adjusted_nav", "nav"]


@dataclass(frozen=True)
class FundNavSeries:
    fund_name: str
    file_name: str
    sheet_name: str
    nav_column: str
    nav: pd.Series
    quality_score: int


@dataclass(frozen=True)
class NavIngestResult:
    returns: pd.DataFrame
    metadata: pd.DataFrame
    skipped: pd.DataFrame


def prepare_nav_return_panel(
    input_dir: str | Path,
    frequency: str = "monthly",
    min_observations: int = 36,
    drop_missing: bool = True,
    include_cash: bool = True,
    cash_return: float = 0.001,
) -> NavIngestResult:
    """Read fund NAV files and convert selected funds to periodic returns."""

    input_path = Path(input_dir).expanduser()
    funds: list[FundNavSeries] = []
    skipped: list[dict[str, Any]] = []

    for path in list_nav_files(input_path):
        try:
            funds.append(read_best_nav_series(path))
        except Exception as exc:
            skipped.append({"file_name": path.name, "reason": f"{type(exc).__name__}: {exc}"})

    if not funds:
        raise ValueError(f"No usable NAV files found in {input_path}")

    rule = frequency_to_rule(frequency)
    returns = {}
    meta_rows: list[dict[str, Any]] = []
    for fund in funds:
        periodic_nav = fund.nav.resample(rule).last().dropna()
        periodic_return = periodic_nav.pct_change().dropna()
        if len(periodic_return) < min_observations:
            skipped.append(
                {
                    "file_name": fund.file_name,
                    "reason": f"Only {len(periodic_return)} {frequency} returns, below min_observations={min_observations}",
                }
            )
            continue
        returns[fund.fund_name] = periodic_return
        meta_rows.append(
            {
                "fund_name": fund.fund_name,
                "file_name": fund.file_name,
                "sheet_name": fund.sheet_name,
                "nav_column": fund.nav_column,
                "raw_nav_rows": len(fund.nav),
                "return_observations": len(periodic_return),
                "first_return_date": periodic_return.index.min(),
                "last_return_date": periodic_return.index.max(),
                "quality_score": fund.quality_score,
            }
        )

    if not returns:
        raise ValueError("NAV files were readable, but no fund met the minimum observation threshold")

    panel = pd.DataFrame(returns).sort_index()
    if drop_missing:
        panel = panel.dropna(how="any")
    if panel.empty:
        raise ValueError("After aligning selected funds, no common return dates remain")

    if include_cash and "cash" not in panel.columns:
        panel["cash"] = float(cash_return)

    panel.index.name = "date"
    metadata = pd.DataFrame(meta_rows)
    skipped_frame = pd.DataFrame(skipped, columns=["file_name", "reason"])
    return NavIngestResult(returns=panel, metadata=metadata, skipped=skipped_frame)


def list_nav_files(input_path: Path) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    files = []
    for path in input_path.iterdir():
        if path.name.startswith("._") or path.name.startswith("."):
            continue
        if path.suffix.lower() in {".xlsx", ".xls"}:
            files.append(path)
    return sorted(files)


def read_best_nav_series(path: Path) -> FundNavSeries:
    engine = excel_engine_for_path(path)
    try:
        xls = pd.ExcelFile(path, engine=engine)
    except ImportError as exc:
        raise ImportError(missing_excel_dependency_message(path, engine)) from exc
    candidates: list[FundNavSeries] = []
    errors: list[str] = []

    for sheet in xls.sheet_names:
        for header in (0, None):
            try:
                raw = pd.read_excel(path, sheet_name=sheet, header=header, engine=engine)
            except ImportError as exc:
                raise ImportError(missing_excel_dependency_message(path, engine)) from exc
            except Exception as exc:
                errors.append(f"{sheet}: {type(exc).__name__}: {exc}")
                continue
            candidate = nav_series_from_frame(raw, path, sheet)
            if candidate is not None:
                candidates.append(candidate)

    if not candidates:
        joined = "; ".join(errors[:3])
        raise ValueError(f"No date/NAV columns found. {joined}".strip())
    return max(candidates, key=lambda fund: (fund.quality_score, len(fund.nav)))


def excel_engine_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return "openpyxl"
    if suffix == ".xls":
        return "xlrd"
    raise ValueError(f"Unsupported NAV file extension: {path.suffix}")


def missing_excel_dependency_message(path: Path, engine: str) -> str:
    if engine == "xlrd":
        return f"Install xlrd>=2.0 to read legacy .xls NAV file: {path.name}"
    if engine == "openpyxl":
        return f"Install openpyxl>=3.1 to read .xlsx NAV file: {path.name}"
    return f"Install the pandas Excel dependency for {path.name}"


def nav_series_from_frame(raw: pd.DataFrame, path: Path, sheet: str) -> FundNavSeries | None:
    if raw.empty:
        return None
    frame = raw.copy()
    frame.columns = [str(col).strip() for col in frame.columns]
    date_col, date_labeled = find_date_column(frame)
    nav_col, nav_labeled = find_nav_column(frame, date_col)
    if date_col is None or nav_col is None:
        return None

    dates = parse_possible_dates(frame[date_col])
    nav = pd.to_numeric(frame[nav_col], errors="coerce")
    clean = pd.DataFrame({"date": dates, "nav": nav}).dropna()
    clean = clean[clean["nav"] > 0.0].sort_values("date").drop_duplicates("date", keep="last")
    if len(clean) < 3:
        return None

    fund_name, name_from_column = infer_fund_name(frame, path)
    quality_score = len(clean)
    quality_score += 10000 if name_from_column and names_match(fund_name, path.stem) else 0
    quality_score += 5000 if nav_labeled else 0
    quality_score += 2500 if date_labeled else 0
    series = clean.set_index("date")["nav"].astype(float)
    series.index = pd.DatetimeIndex(series.index, name="date")
    return FundNavSeries(
        fund_name=fund_name,
        file_name=path.name,
        sheet_name=sheet,
        nav_column=str(nav_col),
        nav=series,
        quality_score=quality_score,
    )


def find_date_column(frame: pd.DataFrame) -> tuple[str | None, bool]:
    for col in frame.columns:
        name = str(col).strip().lower()
        if "日期" in str(col) or name in DATE_NAMES:
            return str(col), True
    scored = []
    for col in frame.columns:
        parsed = parse_possible_dates(frame[col])
        scored.append((int(parsed.notna().sum()), str(col)))
    best_count, best_col = max(scored, default=(0, ""))
    return (best_col, False) if best_count >= max(3, len(frame) // 3) else (None, False)


def find_nav_column(frame: pd.DataFrame, date_col: str | None) -> tuple[str | None, bool]:
    for target in NAV_PRIORITY:
        for col in frame.columns:
            if target in str(col).strip():
                return str(col), True
    numeric_scores = []
    for col in frame.columns:
        if str(col) == str(date_col):
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        positive = values[(values > 0.0) & (values < 1000.0)]
        numeric_scores.append((int(positive.notna().sum()), str(col)))
    best_count, best_col = max(numeric_scores, default=(0, ""))
    return (best_col, False) if best_count >= max(3, len(frame) // 3) else (None, False)


def infer_fund_name(frame: pd.DataFrame, path: Path) -> tuple[str, bool]:
    for col in frame.columns:
        if "产品名称" in str(col) or "基金简称" in str(col):
            values = frame[col].dropna().astype(str)
            if not values.empty:
                return values.iloc[0].strip(), True
    return path.stem, False


def parse_possible_dates(values: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(values, errors="coerce")


def names_match(left: str, right: str) -> bool:
    compact_left = normalize_name(left)
    compact_right = normalize_name(right)
    return compact_left in compact_right or compact_right in compact_left


def normalize_name(value: str) -> str:
    text = str(value)
    for token in ["私募证券投资基金", "证券投资基金", "私募基金", "基金", " "]:
        text = text.replace(token, "")
    return text.strip()


def frequency_to_rule(frequency: str) -> str:
    normalized = frequency.lower().strip()
    if normalized in {"m", "month", "monthly"}:
        return "ME"
    if normalized in {"w", "week", "weekly"}:
        return "W-FRI"
    raise ValueError("frequency must be monthly or weekly")
