"""Safety validators to guard against timeframe drift and data leakage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import load_app_config

DATE_CANDIDATES = ("date", "timestamp", "datetime", "as_of")
PRICE_CANDIDATES = ("close", "adj_close", "price")


@dataclass
class ValidationResult:
    name: str
    status: str
    details: str


def _read_table(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - defensive logging
        return pd.DataFrame({"__error__": [str(exc)]})
    return None


def _infer_datetime_column(frame: pd.DataFrame) -> Optional[str]:
    """Attempt to locate an existing datetime column or index.

    The helper returns the column name containing datetime-like data.  If the
    frame is indexed by :class:`~pandas.DatetimeIndex` we return the sentinel
    string ``"__index__"`` so callers can fall back to the index without
    mutating the original object.
    """

    # 1) Direct dtype inspection avoids re-parsing already-normalised columns.
    for col in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[col]):
            return col

    # 2) Fuzzy name matching for common datetime column aliases.
    for candidate in DATE_CANDIDATES:
        for col in frame.columns:
            if candidate in col.lower():
                series = frame[col]
                try:
                    pd.to_datetime(series, errors="raise")
                    return col
                except Exception:  # pragma: no cover - ignore mis-typed column
                    continue

    # 3) Use the index if it is already datetime based.
    if isinstance(frame.index, pd.DatetimeIndex):
        return "__index__"

    return None


def _infer_price_column(frame: pd.DataFrame) -> Optional[str]:
    for candidate in PRICE_CANDIDATES:
        for col in frame.columns:
            lowered = col.lower()
            if candidate == lowered or lowered.endswith(candidate):
                return col
    for col in frame.columns:
        if any(cand in col.lower() for cand in PRICE_CANDIDATES):
            return col
    return None


def _get_datetime_series(frame: pd.DataFrame, column: Optional[str]) -> Optional[pd.Series]:
    if column is None:
        return None
    if column == "__index__":
        if isinstance(frame.index, pd.DatetimeIndex):
            return pd.to_datetime(pd.Series(frame.index, index=frame.index, name="index"), errors="coerce")
        return None
    series = frame[column]
    return pd.to_datetime(series, errors="coerce")


def _write_report(report_path: Path, header: str, summary: List[ValidationResult], extras: str = "") -> None:
    lines = [header, "", "| Check | Status | Details |", "| --- | --- | --- |"]
    for result in summary:
        lines.append(f"| {result.name} | {result.status} | {result.details} |")
    if extras:
        lines.extend(["", extras])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_timeframe_validator(
    database_root: Path,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    report_path: Path,
) -> bool:
    """Validate that company silver/gold rows fall within the configured window."""

    summary: List[ValidationResult] = []
    offenders: List[str] = []
    start = pd.to_datetime(start) if start else None
    end = pd.to_datetime(end) if end else None
    total_files = 0
    total_rows = 0
    total_outside = 0

    companies_dir = database_root / "companies"
    if not companies_dir.exists():
        summary.append(ValidationResult("directories", "SKIP", "No companies/ directory found"))
        _write_report(report_path, "# Timeframe Validation", summary)
        return True

    for company in sorted(companies_dir.iterdir()):
        if not company.is_dir():
            continue
        for layer in ("silver", "gold"):
            layer_dir = company / layer
            if not layer_dir.exists():
                continue
            for path in layer_dir.rglob("*"):
                if path.suffix.lower() not in {".csv", ".parquet", ".pq"}:
                    continue
                total_files += 1
                frame = _read_table(path)
                if frame is None or frame.empty:
                    continue
                if "__error__" in frame.columns:
                    offenders.append(f"{path}: unable to read file - {frame['__error__'].iloc[0]}")
                    continue
                column = _infer_datetime_column(frame)
                dates = _get_datetime_series(frame, column)
                if dates is None:
                    offenders.append(f"{path}: unable to infer datetime column")
                    continue
                valid = dates.notna()
                mask = valid.copy()
                if start is not None:
                    mask &= dates >= start
                if end is not None:
                    mask &= dates <= end
                total_rows += len(frame)
                outside = (~mask).sum()
                total_outside += outside
                if outside:
                    offenders.append(
                        f"{path}: {outside} rows outside [{start.date() if start else '-'}, {end.date() if end else '-'}]"
                    )
    status = "PASS" if total_outside == 0 else "FAIL"
    details = f"checked {total_files} files ({total_rows} rows), outside={total_outside}"
    summary.append(ValidationResult("timeframe", status, details))
    extras = "\n".join(["## Offending files", "", *offenders]) if offenders else ""
    _write_report(report_path, "# Timeframe Validation", summary, extras)
    return total_outside == 0


def _sample_indices(frame: pd.DataFrame, sample_size: int = 1000) -> pd.Index:
    if len(frame) <= sample_size:
        return frame.index
    return frame.sample(sample_size, random_state=0).index.sort_values()


def run_leakage_validator(
    gold_root: Path,
    horizons: Iterable[int],
    report_path: Path,
    sample_size: int = 1000,
) -> bool:
    summary: List[ValidationResult] = []
    offenders: List[str] = []
    gold_files = list(gold_root.rglob("panel.parquet")) + list(gold_root.rglob("panel.csv"))
    if not gold_files:
        summary.append(ValidationResult("datasets", "SKIP", "No gold panel files found"))
        _write_report(report_path, "# Leakage Validation", summary)
        return True

    for path in gold_files:
        frame = _read_table(path)
        if frame is None or frame.empty:
            continue
        if "__error__" in frame.columns:
            offenders.append(f"{path}: unable to read file - {frame['__error__'].iloc[0]}")
            continue
        column = _infer_datetime_column(frame)
        dates = _get_datetime_series(frame, column)
        if dates is not None:
            frame = frame.assign(__validator_datetime=dates).dropna(subset=["__validator_datetime"]).sort_values(
                "__validator_datetime"
            )
        price_col = _infer_price_column(frame)
        numeric_cols = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
        sample_idx = _sample_indices(frame, sample_size)
        for horizon in horizons:
            label_col = f"ret_h{horizon}"
            if price_col and label_col in frame.columns:
                future_price = frame[price_col].shift(-horizon)
                expected_return = (future_price - frame[price_col]) / frame[price_col]
                label_slice = frame[label_col].loc[sample_idx]
                expected_slice = expected_return.loc[sample_idx]
                diff = (label_slice - expected_slice).abs()
                mismatch = diff > 1e-6
                if mismatch.any():
                    offenders.append(
                        f"{path} horizon {horizon}: {int(mismatch.sum())} label rows deviate from recomputed returns"
                    )
            feature_cols = [c for c in numeric_cols if not c.startswith("ret_h") and not c.startswith("vol_h")]
            leakage_hits: List[str] = []
            for feature in feature_cols:
                col_values = frame[feature].loc[sample_idx]
                if price_col:
                    future_price = frame[price_col].shift(-horizon).loc[sample_idx]
                    if future_price.notna().any() and col_values.notna().any():
                        price_match = np.isclose(col_values, future_price, atol=1e-6, rtol=1e-4)
                        if price_match.mean() > 0.8:
                            leakage_hits.append(feature)
                            continue
                    expected_return = (future_price - frame[price_col].loc[sample_idx]) / frame[price_col].loc[
                        sample_idx
                    ]
                    return_match = np.isclose(col_values, expected_return, atol=1e-6, rtol=1e-4)
                    if return_match.mean() > 0.8:
                        leakage_hits.append(feature)
                        continue
                if label_col in frame.columns:
                    label_match = np.isclose(col_values, frame[label_col].loc[sample_idx], atol=1e-6, rtol=1e-4)
                    if label_match.mean() > 0.8:
                        leakage_hits.append(feature)
                        continue
                if column and column != "__index__":
                    shifted = frame[feature].shift(-horizon).loc[sample_idx]
                    if shifted.notna().any() and col_values.notna().any():
                        shifted_match = np.isclose(col_values, shifted, atol=1e-6, rtol=1e-4)
                        if shifted_match.mean() > 0.8:
                            leakage_hits.append(feature)
            status = "PASS" if not leakage_hits else "FAIL"
            details = f"{path.name} horizon {horizon}"
            summary.append(ValidationResult(details, status, ", ".join(leakage_hits) or "no issues"))
            for hit in leakage_hits:
                offenders.append(f"{path} horizon {horizon}: feature '{hit}' mirrors future data")
    extras = "\n".join(["## Potential leakage", "", *offenders]) if offenders else ""
    all_pass = not offenders
    _write_report(report_path, "# Leakage Validation", summary, extras)
    return all_pass


def run_validators(config_path: str = "config/app.yaml", database_root: str = "database") -> Tuple[bool, bool]:
    cfg = load_app_config(config_path)
    start = cfg.get("data", {}).get("start_date")
    end = cfg.get("data", {}).get("end_date")
    horizons = cfg.get("labels", {}).get("horizons", [5, 20, 50])
    timeframe_ok = run_timeframe_validator(Path(database_root), start, end, Path("reports/validation_timeframe.md"))
    leakage_ok = run_leakage_validator(
        Path(database_root) / "companies",
        horizons,
        Path("reports/validation_leakage.md"),
    )
    return timeframe_ok, leakage_ok


__all__ = [
    "run_timeframe_validator",
    "run_leakage_validator",
    "run_validators",
]
