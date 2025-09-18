"""LLM-powered research extractor with timeframe and provenance safeguards.

This module implements a light-weight pipeline that discovers research sources
for a company, validates them against the configured timeframe, and extracts
structured summaries via an LLM provider.  The implementation is guarded behind
``config/app.yaml``'s ``llm.enabled`` flag so that environments without API
access can continue to operate normally.

The design emphasises: JSON schema validation, provenance tracking
(checksum/url/timestamps), retries with exponential backoff, and persistence to
the silver layer of the on-disk data lake.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence

import pandas as pd

from .config import load_app_config

try:  # pragma: no cover - optional dependency in offline envs
    from openai import OpenAI
except Exception:  # pragma: no cover - the extractor falls back to an offline client
    OpenAI = None  # type: ignore

try:  # pragma: no cover - optional sentiment dependency
    from transformers import pipeline
except Exception:  # pragma: no cover - avoid import failure in constrained envs
    pipeline = None  # type: ignore


ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass
class SourceCandidate:
    """Describes a potential research artefact discovered during crawling."""

    url: str
    published_at: datetime
    source_type: str  # "news" or "filing"
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class CompanyResearchRecord:
    """Structured JSON payload persisted after LLM extraction."""

    symbol: str
    company_name: str
    title: str
    summary: str
    published_at: str
    fetched_at: str
    url: str
    checksum: str
    source_type: str
    confidence: float
    sentiment: float
    rationale: str
    citations: Sequence[str]


RECORD_SCHEMA = {
    "type": "object",
    "required": [
        "symbol",
        "company_name",
        "title",
        "summary",
        "published_at",
        "fetched_at",
        "url",
        "checksum",
        "source_type",
        "confidence",
        "sentiment",
        "rationale",
        "citations",
    ],
    "properties": {
        "symbol": {"type": "string"},
        "company_name": {"type": "string"},
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "published_at": {"type": "string"},
        "fetched_at": {"type": "string"},
        "url": {"type": "string"},
        "checksum": {"type": "string"},
        "source_type": {"type": "string", "enum": ["news", "filing"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "sentiment": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "rationale": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}


class SourceConnector(Protocol):
    """Protocol implemented by discovery connectors."""

    def discover(
        self,
        symbol: str,
        company_name: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Sequence[SourceCandidate]:
        ...


class TextFetcher(Protocol):
    """Abstraction so tests can inject deterministic fetchers."""

    def __call__(self, candidate: SourceCandidate) -> Optional[str]:
        ...


class LLMClient(Protocol):
    """Interface implemented by OpenAI (or offline) clients."""

    def extract(self, prompt: str, schema: Dict[str, object]) -> Dict[str, object]:
        ...


def _default_sentiment_scorer() -> Callable[[str], float]:
    """Returns a callable that scores sentiment in [-1, 1]."""

    def _heuristic(text: str) -> float:
        lower = text.lower()
        positive = sum(lower.count(tok) for tok in ["growth", "beat", "strong"])
        negative = sum(lower.count(tok) for tok in ["weak", "miss", "risk"])
        total = positive + negative
        if total == 0:
            return 0.0
        return (positive - negative) / max(total, 1)

    if pipeline is None:  # pragma: no cover - fallback executed in tests
        return _heuristic

    try:  # pragma: no cover - huggingface download may fail offline
        sentiment_pipeline = pipeline("text-classification", model="ProsusAI/finbert", top_k=1)
    except Exception:
        return _heuristic

    def _score(text: str) -> float:  # pragma: no cover - heavy dependency
        result = sentiment_pipeline(text[:2000])[0]
        label = result["label"].lower()
        score = float(result["score"])
        if label == "positive":
            return score
        if label == "negative":
            return -score
        return 0.0

    return _score


class OfflineLLMClient:
    """Fallback client that produces deterministic summaries without API calls."""

    def extract(self, prompt: str, schema: Dict[str, object]) -> Dict[str, object]:  # pragma: no cover - trivial
        summary = prompt.splitlines()[0][:200]
        return {
            "title": "Offline synopsis",
            "summary": summary,
            "confidence": 0.4,
            "rationale": "Offline mode – review source manually.",
            "citations": [],
        }


class OpenAIResponsesClient:
    """Thin wrapper over the OpenAI Responses API enforcing JSON schema."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        if OpenAI is None:
            raise RuntimeError("openai package not available")
        self._client = OpenAI()
        self._model = model

    def extract(self, prompt: str, schema: Dict[str, object]) -> Dict[str, object]:  # pragma: no cover - network call
        response = self._client.responses.create(
            model=self._model,
            input=[{"role": "user", "content": prompt}],
            response_format={"type": "json_schema", "json_schema": {"name": "CompanyResearchRecord", "schema": schema}},
        )
        output = response.output[0].content[0].text  # type: ignore[attr-defined]
        return json.loads(output)


class LocalBronzeConnector:
    """Discovers files within the bronze layer that match the timeframe."""

    def __init__(self, database_root: Path, subdir: str, source_type: str) -> None:
        self.database_root = database_root
        self.subdir = subdir
        self.source_type = source_type

    def discover(
        self,
        symbol: str,
        company_name: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Sequence[SourceCandidate]:
        base = self.database_root / "companies" / symbol / "bronze" / self.subdir
        if not base.exists():
            return []
        candidates: List[SourceCandidate] = []
        for path in base.rglob("*.json"):
            published = datetime.fromtimestamp(path.stat().st_mtime)
            if not (start_date <= published <= end_date):
                continue
            candidates.append(
                SourceCandidate(
                    url=path.as_uri(),
                    published_at=published,
                    source_type=self.source_type,
                    metadata={"path": str(path)},
                )
            )
        return sorted(candidates, key=lambda c: c.published_at)


def _read_local_path(candidate: SourceCandidate) -> Optional[str]:
    path_str = candidate.metadata.get("path")
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _default_fetcher(candidate: SourceCandidate) -> Optional[str]:
    """Fetches remote content with a best-effort strategy."""

    if candidate.url.startswith("file://") or candidate.metadata.get("path"):
        return _read_local_path(candidate)
    try:  # pragma: no cover - network disabled in tests
        import requests

        response = requests.get(candidate.url, timeout=15)
        response.raise_for_status()
        return response.text
    except Exception:
        return None


def _validate_record(payload: Dict[str, object]) -> CompanyResearchRecord:
    for field in RECORD_SCHEMA["required"]:
        if field not in payload:
            raise ValueError(f"Missing required field '{field}' in LLM response")
    for key, meta in RECORD_SCHEMA["properties"].items():
        if key not in payload:
            continue
        value = payload[key]
        typ = meta.get("type")
        if typ == "number" and not isinstance(value, (int, float)):
            raise ValueError(f"Field '{key}' must be numeric")
        if typ == "string" and not isinstance(value, str):
            raise ValueError(f"Field '{key}' must be a string")
        if typ == "array" and not isinstance(value, (list, tuple)):
            raise ValueError(f"Field '{key}' must be an array")
    return CompanyResearchRecord(**payload)  # type: ignore[arg-type]


class LLMResearchExtractor:
    """Coordinates discovery, extraction, and persistence for research artefacts."""

    def __init__(
        self,
        database_root: Path,
        connectors: Sequence[SourceConnector],
        llm_client: Optional[LLMClient] = None,
        fetcher: TextFetcher = _default_fetcher,
        rate_limit_per_min: int = 20,
        max_retries: int = 3,
        backoff_seconds: float = 1.5,
    ) -> None:
        self.database_root = database_root
        self.connectors = connectors
        self.llm = llm_client or OfflineLLMClient()
        self.fetcher = fetcher
        self.rate_limit = max(rate_limit_per_min, 1)
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.sentiment_fn = _default_sentiment_scorer()
        self._last_call: float = 0.0

    # ------------------------------------------------------------------
    # Public API

    def run(
        self,
        symbols: Sequence[str],
        company_names: Optional[Dict[str, str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        start = pd.to_datetime(start_date) if start_date else pd.Timestamp("1900-01-01")
        end = pd.to_datetime(end_date) if end_date else pd.Timestamp.utcnow()
        results: Dict[str, Dict[str, pd.DataFrame]] = {}
        for symbol in symbols:
            name = company_names.get(symbol, symbol) if company_names else symbol
            candidates = self._collect_candidates(symbol, name, start.to_pydatetime(), end.to_pydatetime())
            if dry_run:
                self._print_dry_run(symbol, name, candidates)
                continue
            if not candidates:
                continue
            records = self._process_candidates(symbol, name, candidates, start, end)
            if not records:
                continue
            grouped = self._persist(symbol, records)
            results[symbol] = grouped
        return results

    # ------------------------------------------------------------------
    # Internal helpers

    def _collect_candidates(
        self,
        symbol: str,
        company_name: str,
        start: datetime,
        end: datetime,
    ) -> List[SourceCandidate]:
        discoveries: List[SourceCandidate] = []
        for connector in self.connectors:
            try:
                findings = connector.discover(symbol, company_name, start, end)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"Connector {connector!r} failed for {symbol}: {exc}")
                continue
            for candidate in findings:
                if start <= candidate.published_at <= end:
                    discoveries.append(candidate)
        unique: Dict[str, SourceCandidate] = {}
        for candidate in discoveries:
            unique.setdefault(candidate.url, candidate)
        return list(sorted(unique.values(), key=lambda c: c.published_at))

    def _print_dry_run(
        self,
        symbol: str,
        company_name: str,
        candidates: Sequence[SourceCandidate],
    ) -> None:
        print(f"=== Dry run for {symbol} ({company_name}) ===")
        print("CompanyResearchRecord schema:")
        print(json.dumps(RECORD_SCHEMA, indent=2))
        if not candidates:
            print("No candidates discovered within timeframe.")
            return
        for candidate in candidates:
            print(f"- {candidate.source_type.upper()} | {candidate.published_at.strftime(ISO_FORMAT)} | {candidate.url}")

    def _process_candidates(
        self,
        symbol: str,
        company_name: str,
        candidates: Sequence[SourceCandidate],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> List[CompanyResearchRecord]:
        processed: List[CompanyResearchRecord] = []
        for candidate in candidates:
            text = self.fetcher(candidate)
            if not text:
                continue
            checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
            published_ts = pd.Timestamp(candidate.published_at)
            if not (start <= published_ts <= end):
                continue
            prompt = self._build_prompt(symbol, company_name, text, candidate)
            payload = self._call_llm(prompt)
            payload.update(
                {
                    "symbol": symbol,
                    "company_name": company_name,
                    "published_at": candidate.published_at.strftime(ISO_FORMAT),
                    "fetched_at": datetime.utcnow().strftime(ISO_FORMAT),
                    "url": candidate.url,
                    "checksum": checksum,
                    "source_type": candidate.source_type,
                }
            )
            record = _validate_record(payload)
            record.sentiment = float(self.sentiment_fn(text))
            processed.append(record)
        return processed

    def _call_llm(self, prompt: str) -> Dict[str, object]:
        attempt = 0
        while True:
            attempt += 1
            now = time.time()
            elapsed = now - self._last_call
            min_interval = 60.0 / self.rate_limit
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            try:
                result = self.llm.extract(prompt, RECORD_SCHEMA)
                self._last_call = time.time()
                return result
            except Exception as exc:  # pragma: no cover - network/backoff logic
                if attempt >= self.max_retries:
                    raise RuntimeError(f"LLM extraction failed after {attempt} attempts: {exc}") from exc
                time.sleep(self.backoff_seconds * attempt)

    @staticmethod
    def _build_prompt(symbol: str, company_name: str, text: str, candidate: SourceCandidate) -> str:
        header = (
            f"Summarise the following document for {company_name} ({symbol}). "
            "Only use facts from the text. Cite source snippets where relevant."
        )
        return header + "\n" + text[:4000]

    def _write_dataframe(self, df: pd.DataFrame, out_path: Path) -> None:
        try:
            df.to_parquet(out_path, index=False)
        except Exception:  # pragma: no cover - fallback when pyarrow unavailable
            df.to_pickle(out_path)

    def _persist(self, symbol: str, records: Sequence[CompanyResearchRecord]) -> Dict[str, pd.DataFrame]:
        silver_dir = self.database_root / "companies" / symbol / "silver"
        silver_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = silver_dir / "provenance_ledger.jsonl"
        grouped: Dict[str, List[Dict[str, object]]] = {"news": [], "filing": []}
        for record in records:
            grouped.setdefault(record.source_type, []).append(asdict(record))
        outputs: Dict[str, pd.DataFrame] = {}
        for kind, rows in grouped.items():
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["published_at"] = pd.to_datetime(df["published_at"])
            df["fetched_at"] = pd.to_datetime(df["fetched_at"])
            out_path = silver_dir / ("news_clean.parquet" if kind == "news" else "filings_clean.parquet")
            self._write_dataframe(df, out_path)
            outputs[kind] = df
            with ledger_path.open("a", encoding="utf-8") as ledger:
                for row in rows:
                    ledger.write(json.dumps(row) + "\n")
        return outputs


def build_default_extractor(
    config_path: str = "config/app.yaml",
    database_root: str = "database",
    model: Optional[str] = None,
) -> Optional[LLMResearchExtractor]:
    cfg = load_app_config(config_path)
    llm_cfg = cfg.get("llm", {})
    if not llm_cfg.get("enabled", False):
        print("LLM extractor disabled via config.")
        return None
    connectors: List[SourceConnector] = [
        LocalBronzeConnector(Path(database_root), "news", "news"),
        LocalBronzeConnector(Path(database_root), "filings", "filing"),
    ]
    client: Optional[LLMClient] = None
    provider = llm_cfg.get("provider", "openai")
    if provider == "openai" and llm_cfg.get("schema_strict", True):
        try:
            client = OpenAIResponsesClient(model=model or "gpt-4o-mini")
        except Exception as exc:  # pragma: no cover - offline fallback
            print(f"Falling back to offline LLM client: {exc}")
            client = OfflineLLMClient()
    else:
        client = OfflineLLMClient()
    return LLMResearchExtractor(
        database_root=Path(database_root),
        connectors=connectors,
        llm_client=client,
        rate_limit_per_min=int(llm_cfg.get("rate_limit_per_min", 20)),
    )


__all__ = [
    "CompanyResearchRecord",
    "LLMResearchExtractor",
    "LocalBronzeConnector",
    "OpenAIResponsesClient",
    "OfflineLLMClient",
    "SourceCandidate",
    "build_default_extractor",
]
