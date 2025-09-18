import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from gnn_trading.llm_extractor import (
    CompanyResearchRecord,
    LLMResearchExtractor,
    LocalBronzeConnector,
    SourceCandidate,
)


class DummyConnector:
    def __init__(self, candidates):
        self._candidates = candidates

    def discover(self, symbol, company_name, start_date, end_date):
        return self._candidates


class DummyLLM:
    def extract(self, prompt, schema):
        return {
            "title": "Test",
            "summary": prompt.splitlines()[0][:50],
            "confidence": 0.8,
            "rationale": "LLM synthetic output",
            "citations": ["snippet"],
            "sentiment": 0.1,
        }


class DummyFetcher:
    def __init__(self, payloads):
        self.payloads = payloads

    def __call__(self, candidate):
        return self.payloads.get(candidate.url)


def test_timeframe_filtering(tmp_path):
    start = datetime.utcnow() - timedelta(days=5)
    end = datetime.utcnow()
    inside = start + timedelta(days=1)
    outside = start - timedelta(days=10)
    cands = [
        SourceCandidate(url="file://inside", published_at=inside, source_type="news"),
        SourceCandidate(url="file://outside", published_at=outside, source_type="news"),
    ]
    fetcher = DummyFetcher({"file://inside": "Great growth ahead", "file://outside": "Should be filtered"})
    extractor = LLMResearchExtractor(
        database_root=tmp_path,
        connectors=[DummyConnector(cands)],
        llm_client=DummyLLM(),
        fetcher=fetcher,
    )
    extractor.run(["AAPL"], company_names={"AAPL": "Apple"}, start_date=start.isoformat(), end_date=end.isoformat())
    out_path = tmp_path / "companies" / "AAPL" / "silver" / "news_clean.parquet"
    assert out_path.exists(), "News parquet should be written"
    try:
        df = pd.read_parquet(out_path)
    except Exception:
        df = pd.read_pickle(out_path)
    assert len(df) == 1
    assert df.iloc[0]["url"] == "file://inside"


def test_dry_run_prints_schema(capsys, tmp_path):
    start = datetime.utcnow() - timedelta(days=1)
    end = datetime.utcnow()
    connector = DummyConnector(
        [SourceCandidate(url="file://example", published_at=end, source_type="filing")]
    )
    extractor = LLMResearchExtractor(
        database_root=tmp_path,
        connectors=[connector],
        llm_client=DummyLLM(),
        fetcher=DummyFetcher({"file://example": "sample"}),
    )
    extractor.run(["MSFT"], company_names={"MSFT": "Microsoft"}, start_date=start.isoformat(), end_date=end.isoformat(), dry_run=True)
    captured = capsys.readouterr().out
    assert "Dry run" in captured
    assert "CompanyResearchRecord" in captured
