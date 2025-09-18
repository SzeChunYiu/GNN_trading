"""Batch recommendation helper built on top of the fund manager backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .data_hub import LocalDataHub
from .fund_manager import AnalysisCoordinator, GNNBackend, Holding, PortfolioDB


def resolve_csv_path(ticker: str, template: str, hub: LocalDataHub) -> Optional[str]:
    candidate = Path(template.format(ticker=ticker))
    if candidate.exists():
        return str(candidate)
    hub_path = hub.market_path(ticker)
    if hub_path.exists():
        return str(hub_path)
    return None


def summarize_reports(reports: List[Dict[str, object]]) -> List[Dict[str, object]]:
    summary = []
    for report in reports:
        decision = report["decision"]
        model = report["model"]
        summary.append(
            {
                "ticker": report["ticker"],
                "decision": decision["decision"],
                "composite_score": decision["composite_score"],
                "probability": model.get("probability"),
                "expected_return": model.get("expected_return"),
                "volatility": model.get("volatility"),
                "expected_price": report.get("expected_price"),
            }
        )
    summary.sort(key=lambda row: row["composite_score"], reverse=True)
    return summary


def run_recommendations(
    backend: GNNBackend,
    coordinator: AnalysisCoordinator,
    tickers: List[str],
    template: str,
    hub: LocalDataHub,
    holdings: Dict[str, Holding],
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for ticker in tickers:
        csv_path = resolve_csv_path(ticker, template, hub)
        if csv_path is None:
            print(f"Skipping {ticker}: no market data found.")
            continue
        holding = holdings.get(ticker)
        analysis = coordinator.evaluate(ticker, csv_path, holding)
        if analysis is None:
            continue
        results.append(analysis)
    return summarize_reports(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate buy/sell recommendations in batch mode")
    parser.add_argument("--db", default="fund_manager.db", help="SQLite file with stored holdings")
    parser.add_argument("--tickers", nargs="*", help="Optional tickers to analyse; defaults to DB holdings")
    parser.add_argument("--data-template", default="data/{ticker}.csv", help="Path template for OHLCV CSV files")
    parser.add_argument("--checkpoint", default=None, help="Path to a trained supervised model state dict")
    parser.add_argument("--device", default="auto", help="Device to run inference on")
    parser.add_argument("--window", type=int, default=60, help="Sequence length window for the encoder")
    parser.add_argument("--pred-horizon", type=int, default=5, help="Prediction horizon used during training")
    parser.add_argument("--model-json", default=None, help="Optional JSON file with model hyperparameters")
    parser.add_argument("--labels-json", default=None, help="Optional JSON file with label generation params")
    parser.add_argument("--news-json", default=None, help="Optional JSON mapping YYYY-MM-DD to news embedding vectors")
    parser.add_argument("--date-col", default="date", help="Date column in the OHLCV CSV files")
    parser.add_argument("--data-root", default="local_data", help="Directory for the local data hub cache")
    parser.add_argument("--macro-json", default=None, help="Optional JSON file with macro indicator defaults")
    parser.add_argument("--output", default=None, help="Optional path to save the summary as JSON")
    return parser.parse_args()


def load_json(path: Optional[str]) -> Optional[Dict]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    args = parse_args()
    model_cfg = load_json(args.model_json)
    label_cfg = load_json(args.labels_json)
    macro_cfg = load_json(args.macro_json) or {}
    hub = LocalDataHub(args.data_root)
    backend = GNNBackend(
        model_cfg=model_cfg,
        label_cfg=label_cfg,
        window=args.window,
        pred_horizon=args.pred_horizon,
        device=args.device,
        checkpoint=args.checkpoint,
        news_json=args.news_json,
        date_col=args.date_col,
    )
    db = PortfolioDB(args.db)
    coordinator = AnalysisCoordinator(backend=backend, macro_view=macro_cfg)
    holdings = {holding.ticker: holding for holding in db.list_holdings()}
    tickers = [t.upper() for t in args.tickers] if args.tickers else list(holdings)
    if not tickers:
        print("No tickers supplied and no holdings found; nothing to score.")
        return
    summary = run_recommendations(backend, coordinator, tickers, args.data_template, hub, holdings)
    if not summary:
        print("No recommendations generated.")
        return
    print("Ticker  Decision  Score   Prob%  ExpRet%  ExpPx")
    print("-" * 54)
    for row in summary:
        prob = (row["probability"] or 0.0) * 100
        exp_ret = (row["expected_return"] or 0.0) * 100
        exp_price = row["expected_price"] or np.nan
        print(
            f"{row['ticker']:<6} {row['decision']:<9} {row['composite_score']:>6.3f} "
            f"{prob:>6.2f} {exp_ret:>8.2f} {exp_price:>7.2f}"
        )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
        print(f"Saved summary to {args.output}")


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
