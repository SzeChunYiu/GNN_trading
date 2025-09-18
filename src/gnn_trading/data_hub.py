"""Local data management utilities for the trading platform.

The project now supports a persistent *data hub* that stores market, news,
fundamental, and document artefacts on disk.  This module provides a thin
wrapper around a directory hierarchy so that other components can rely on a
consistent layout without worrying about file management details.

The hub focuses on three goals:

* Normalising where artefacts live (``market/``, ``fundamentals/``, ``news/``,
  ``reports/``) below a configurable root folder.
* Recording simple metadata (timestamp, source, schema hints) in a lightweight
  JSON catalogue to speed up discovery and auditing.
* Exposing convenience loaders/savers that integrate seamlessly with the
  quantamental research stack via the :class:`DictDataProvider` helper.

The implementation deliberately avoids heavyweight dependencies; pandas is the
only requirement for serialising tabular data.  Callers are free to store data
as CSV or Parquet as long as they provide the appropriate suffix.  The helper
functions default to CSV for maximum portability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

CATALOG_FILENAME = "catalog.json"


@dataclass
class ArtifactRecord:
    """Metadata snapshot for a stored asset."""

    kind: str
    ticker: str
    path: Path
    updated_at: datetime
    source: str | None = None
    notes: str | None = None

    def to_dict(self) -> Dict[str, str]:
        return {
            "kind": self.kind,
            "ticker": self.ticker,
            "path": str(self.path),
            "updated_at": self.updated_at.isoformat(),
            "source": self.source or "",
            "notes": self.notes or "",
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, str]) -> "ArtifactRecord":
        return cls(
            kind=payload["kind"],
            ticker=payload["ticker"],
            path=Path(payload["path"]),
            updated_at=datetime.fromisoformat(payload["updated_at"]),
            source=payload.get("source") or None,
            notes=payload.get("notes") or None,
        )


class LocalDataHub:
    """Manage on-disk artefacts for market, news, and document data."""

    SUBDIRS = {
        "market": "market",
        "fundamentals": "fundamentals",
        "news": "news",
        "reports": "reports",
        "metadata": "metadata",
    }

    def __init__(self, root: str | Path = "local_data") -> None:
        self.root = Path(root)
        for folder in self.SUBDIRS.values():
            (self.root / folder).mkdir(parents=True, exist_ok=True)
        self._catalog_path = self.root / self.SUBDIRS["metadata"] / CATALOG_FILENAME
        self._catalog: Dict[str, Dict[str, str]] = {}
        self._load_catalog()

    # ------------------------------------------------------------------
    # Catalog helpers

    def _load_catalog(self) -> None:
        if self._catalog_path.exists():
            with self._catalog_path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self._catalog = raw
        else:
            self._catalog = {}

    def _save_catalog(self) -> None:
        payload = {key: value for key, value in self._catalog.items()}
        with self._catalog_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def _record(self, record: ArtifactRecord) -> None:
        key = f"{record.kind}:{record.ticker}"
        self._catalog[key] = record.to_dict()
        self._save_catalog()

    def _remove(self, kind: str, ticker: str) -> None:
        key = f"{kind}:{ticker}"
        if key in self._catalog:
            del self._catalog[key]
            self._save_catalog()

    def list_records(self, kind: Optional[str] = None) -> Dict[str, ArtifactRecord]:
        """Return catalogued artefacts, optionally filtered by kind."""

        out: Dict[str, ArtifactRecord] = {}
        for key, raw in self._catalog.items():
            rec = ArtifactRecord.from_dict(raw)
            if kind and rec.kind != kind:
                continue
            out[key] = rec
        return out

    # ------------------------------------------------------------------
    # Directory helpers

    def _resolve_path(self, kind: str, filename: str) -> Path:
        subdir = self.SUBDIRS.get(kind)
        if subdir is None:
            raise ValueError(f"Unsupported artefact kind: {kind}")
        return self.root / subdir / filename

    # ------------------------------------------------------------------
    # Market data utilities

    def market_path(self, ticker: str, suffix: str = ".csv") -> Path:
        return self._resolve_path("market", f"{ticker.upper()}{suffix}")

    def store_market_data(
        self,
        ticker: str,
        frame: pd.DataFrame,
        source: str | None = None,
        notes: str | None = None,
        suffix: str = ".csv",
    ) -> Path:
        """Persist OHLCV style data for *ticker* and update the catalog."""

        path = self.market_path(ticker, suffix)
        frame.to_csv(path, index=True)
        record = ArtifactRecord(
            kind="market",
            ticker=ticker.upper(),
            path=path,
            updated_at=datetime.utcnow(),
            source=source,
            notes=notes,
        )
        self._record(record)
        return path

    def load_market_data(self, ticker: str) -> pd.DataFrame:
        path = self.market_path(ticker)
        if path.exists():
            return pd.read_csv(path, parse_dates=[0], index_col=0)
        raise FileNotFoundError(f"Market data for {ticker} not found at {path}")

    # ------------------------------------------------------------------
    # Fundamentals / news / reports

    def _store_generic(
        self,
        kind: str,
        ticker: str,
        payload: pd.DataFrame | Dict[str, object],
        source: str | None = None,
        notes: str | None = None,
        suffix: str = ".csv",
    ) -> Path:
        if isinstance(payload, pd.DataFrame):
            path = self._resolve_path(kind, f"{ticker.upper()}{suffix}")
            payload.to_csv(path, index=True)
        else:
            path = self._resolve_path(kind, f"{ticker.upper()}.json")
            with path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        record = ArtifactRecord(
            kind=kind,
            ticker=ticker.upper(),
            path=path,
            updated_at=datetime.utcnow(),
            source=source,
            notes=notes,
        )
        self._record(record)
        return path

    def store_fundamentals(
        self,
        ticker: str,
        payload: pd.DataFrame | Dict[str, object],
        source: str | None = None,
        notes: str | None = None,
    ) -> Path:
        return self._store_generic("fundamentals", ticker, payload, source, notes)

    def store_news(
        self,
        ticker: str,
        payload: pd.DataFrame | Dict[str, object],
        source: str | None = None,
        notes: str | None = None,
    ) -> Path:
        return self._store_generic("news", ticker, payload, source, notes)

    def store_report(
        self,
        ticker: str,
        payload: pd.DataFrame | Dict[str, object],
        source: str | None = None,
        notes: str | None = None,
    ) -> Path:
        return self._store_generic("reports", ticker, payload, source, notes)

    def remove(self, kind: str, ticker: str) -> None:
        path = self._resolve_path(kind, f"{ticker.upper()}.csv")
        if not path.exists():
            json_path = path.with_suffix(".json")
            if json_path.exists():
                json_path.unlink()
        else:
            path.unlink()
        self._remove(kind, ticker)

    # ------------------------------------------------------------------
    # Discovery helpers

    def describe(self, ticker: str) -> Dict[str, ArtifactRecord]:
        """Return all artefacts recorded for *ticker*."""

        out = {}
        for key, rec in self.list_records().items():
            if rec.ticker == ticker.upper():
                out[key] = rec
        return out

    def ensure_market_csv(self, ticker: str, csv_path: str) -> Path:
        """Ingest an external CSV file into the hub if it is newer."""

        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(csv_path)
        frame = pd.read_csv(path, parse_dates=[0])
        return self.store_market_data(ticker, frame, source=str(path))

    # ------------------------------------------------------------------
    # Quantamental integration

    def as_data_provider(self, tickers: Iterable[str]):
        """Return a :class:`DictDataProvider` backed by the stored artefacts."""

        from .quantamental import DictDataProvider  # Local import to avoid cycle

        market: Dict[str, pd.DataFrame] = {}
        fundamentals: Dict[str, pd.DataFrame] = {}
        alternative: Dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            try:
                market[ticker.upper()] = self.load_market_data(ticker)
            except FileNotFoundError:
                continue
            fund_path = self._resolve_path("fundamentals", f"{ticker.upper()}.csv")
            if fund_path.exists():
                fundamentals[ticker.upper()] = pd.read_csv(fund_path, index_col=0, parse_dates=True)
            alt_path = self._resolve_path("news", f"{ticker.upper()}.csv")
            if alt_path.exists():
                alternative[ticker.upper()] = pd.read_csv(alt_path, index_col=0, parse_dates=True)
        return DictDataProvider(market_data=market, fundamentals=fundamentals, alternative=alternative)

    # ------------------------------------------------------------------
    # Convenience views

    def summary_table(self) -> pd.DataFrame:
        """Return the catalog as a pandas DataFrame for quick inspection."""

        if not self._catalog:
            return pd.DataFrame(columns=["kind", "ticker", "path", "updated_at", "source", "notes"])
        rows = []
        for raw in self._catalog.values():
            rows.append(raw)
        frame = pd.DataFrame(rows)
        if "updated_at" in frame.columns:
            frame["updated_at"] = pd.to_datetime(frame["updated_at"], errors="coerce")
        return frame.sort_values(["ticker", "kind"])


__all__ = ["ArtifactRecord", "LocalDataHub"]
