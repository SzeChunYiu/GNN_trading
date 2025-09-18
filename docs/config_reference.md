# Configuration Reference

The application now reads a consolidated YAML file (`config/app.yaml`) to avoid
scattered CLI defaults.  All keys are optional; omitting the file reverts to the
legacy behaviour.

## Top-Level Keys

| Key | Description | Default |
| --- | --- | --- |
| `data.start_date` | Earliest timestamp kept by validators and ingestion. | `null` (no bound) |
| `data.end_date` | Latest timestamp kept by validators and ingestion. | `null` (no bound) |
| `data.providers.prices` | Ordered list of price data providers. | `['stooq', 'yfinance']` |
| `data.providers.fundamentals` | Fundamental data provider identifier. | `sec_xbrl` |
| `data.providers.news` | Ordered list of news/document providers. | `['gdelt', 'sec_filings']` |
| `data.reporting_delay_days` | Lag applied to fundamentals before use. | `3` |
| `features.lookback` | Rolling lookback (days) for leakage-safe features. | `120` |
| `labels.horizons` | Forecast horizons used for labels/validators. | `[5, 20, 50]` |
| `edges.topk` | Max correlation edges per node in cross-asset graphs. | `5` |
| `edges.use_sector` | Whether to include sector-based edges. | `true` |
| `registry.path` | Root directory for the disk-backed model registry. | `./models/gnn_crosssec` |
| `llm.enabled` | Enables LLM-backed assistants when true. | `false` |
| `llm.provider` | Default provider name when LLM is enabled. | `openai` |
| `llm.schema_strict` | Enforces schema validation in LLM extractors. | `true` |
| `llm.rate_limit_per_min` | Maximum extractor calls per minute. | `20` |
| `ui.enabled` | Toggles experimental UI surfaces beyond the CLI. | `false` |

## Usage Patterns

1. Copy `config_example.yaml` to `config/app.yaml` and adjust values.
2. CLI commands accept `--config PATH` to point at alternate files.
3. Validators and registry helpers respect the config while remaining disabled
   unless explicitly invoked.

## Backwards Compatibility

All keys default to the legacy behaviours when omitted.  Removing the config
file or toggling individual flags allows reverting without code changes.
