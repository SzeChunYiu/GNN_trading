"""Command-line entry point for the LLM research extractor."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from .config import load_app_config
from .llm_extractor import build_default_extractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LLM research extractor")
    parser.add_argument("--symbols", required=True, help="Comma separated ticker symbols")
    parser.add_argument("--start-date", required=True, help="Inclusive start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="Inclusive end date (YYYY-MM-DD)")
    parser.add_argument("--database-root", default="database", help="Root of the local data lake")
    parser.add_argument("--config", default="config/app.yaml", help="Path to application config")
    parser.add_argument("--model", default=None, help="Optional override for the OpenAI model")
    parser.add_argument("--dry-run", action="store_true", help="Do not call the LLM; print planned actions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_app_config(args.config)
    llm_cfg = cfg.get("llm", {})
    if not llm_cfg.get("enabled", False):
        print("LLM extractor is disabled (llm.enabled=false). Enable it in config/app.yaml to proceed.")
        return
    extractor = build_default_extractor(args.config, args.database_root, args.model)
    if extractor is None:
        return
    symbols = [tok.strip().upper() for tok in args.symbols.split(",") if tok.strip()]
    company_names: Dict[str, str] = {sym: sym for sym in symbols}
    results = extractor.run(
        symbols=symbols,
        company_names=company_names,
        start_date=args.start_date,
        end_date=args.end_date,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return
    for symbol, groups in results.items():
        news_rows = groups.get("news", 0)
        filing_rows = groups.get("filing", 0)
        news_count = len(news_rows) if hasattr(news_rows, "__len__") else 0
        filing_count = len(filing_rows) if hasattr(filing_rows, "__len__") else 0
        print(
            f"Persisted {news_count} news items and {filing_count} filings for {symbol} "
            f"under {Path(args.database_root) / 'companies' / symbol / 'silver'}"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
