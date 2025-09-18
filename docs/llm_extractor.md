# LLM Research Extractor

The research extractor discovers filings and news artefacts for each ticker,
filters them to the configured timeframe, and summarises them with an LLM.  The
feature is disabled by default; enable it by setting `llm.enabled: true` inside
`config/app.yaml`.

## Safety guarantees

- **Timeframe compliance** – candidate sources are screened during discovery
  *and* after parsing so only items within `[data.start_date, data.end_date]`
  survive.
- **Schema enforcement** – every LLM response must satisfy the
  `CompanyResearchRecord` schema; missing fields raise an error.
- **Provenance** – persisted rows include `url`, `fetched_at`, and `checksum`
  values.  The extractor also appends each record to
  `silver/provenance_ledger.jsonl` for audit trails.
- **No fabricated numbers** – the prompt instructs the LLM to rely solely on
  source text.  If the model cannot find numeric disclosures it must state so.
  Violations should be treated as critical defects.

## Usage

```bash
# Dry run – prints discovered URLs and the schema without calling the API
env PYTHONPATH=src python -m gnn_trading.llm_cli \
  --symbols AAPL,MSFT \
  --start-date 2024-07-01 \
  --end-date 2024-07-31 \
  --dry-run

# Live run – requires llm.enabled=true and valid OpenAI credentials
env PYTHONPATH=src python -m gnn_trading.llm_cli \
  --symbols AAPL \
  --start-date 2024-07-01 \
  --end-date 2024-07-31
```

Outputs are written to `database/companies/<TICKER>/silver/`:

- `news_clean.parquet`
- `filings_clean.parquet`
- `provenance_ledger.jsonl`

The extractor respects rate limits (default 20 calls/minute) and retries with
exponential backoff.

## Extending connectors

The default implementation reads from the bronze layer.  Implement the
`SourceConnector` protocol to plug in alternative discovery strategies (SEC
APIs, GDELT, investor relations feeds, etc.).  Connectors should emit
`SourceCandidate` objects populated with publication timestamps so timeframe
filters can execute efficiently.

## Sentiment scoring

The extractor attempts to use the FinBERT sentiment model via
`transformers`.  If the model is unavailable the code falls back to a keyword
heuristic to remain functional in offline environments.

