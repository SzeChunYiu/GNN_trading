"""Interactive multi-agent fund manager console application.

The interface in this version behaves like a mini terminal application powered
by several specialised agents.  Users can maintain a portfolio, configure macro
views and request detailed buy/sell guidance.  Every class and method carries
explicit annotations so that extending the system is straightforward.
"""

from __future__ import annotations

import argparse
import cmd
import json
import os
import sqlite3
import textwrap
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch

from .agents import AgentContext, DecisionEngine, default_agents
from .dataset import OHLCVGraphDataset
from .features import compute_features
from .labels import make_breakout_labels
from .models import GAT_GRU_Encoder, SupervisedModel
from .patterns import add_pattern_scores
from .utils import pick_device


@dataclass
class Holding:
    """Simple snapshot of a portfolio position."""

    ticker: str
    shares: float
    cost_basis: float


class PortfolioDB:
    """SQLite-backed storage with helper utilities for CRUD operations."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path)
        self._create_table()

    def _create_table(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS holdings (
                ticker TEXT PRIMARY KEY,
                shares REAL NOT NULL,
                cost_basis REAL NOT NULL
            )
            """
        )
        self.conn.commit()

    def upsert(self, holding: Holding) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO holdings(ticker, shares, cost_basis)
            VALUES (?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                shares=excluded.shares,
                cost_basis=excluded.cost_basis
            """,
            (holding.ticker.upper(), holding.shares, holding.cost_basis),
        )
        self.conn.commit()

    def remove(self, ticker: str) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM holdings WHERE ticker=?", (ticker.upper(),))
        self.conn.commit()

    def list_holdings(self) -> List[Holding]:
        cur = self.conn.cursor()
        cur.execute("SELECT ticker, shares, cost_basis FROM holdings ORDER BY ticker")
        rows = cur.fetchall()
        return [Holding(ticker=row[0], shares=row[1], cost_basis=row[2]) for row in rows]


class GNNBackend:
    """Utility wrapper around the neural network encoder for inference."""

    def __init__(
        self,
        model_cfg: Optional[Dict] = None,
        label_cfg: Optional[Dict] = None,
        window: int = 60,
        pred_horizon: int = 5,
        device: str = "auto",
        checkpoint: Optional[str] = None,
        news_json: Optional[str] = None,
        date_col: str = "date",
    ):
        self.model_cfg = {
            "hid": 64,
            "heads": 2,
            "gru_hid": 64,
            "proj_dim": 64,
            "dropout": 0.2,
            "num_gru_layers": 2,
            "bidirectional": True,
            "num_gat_layers": 3,
            "use_temporal_attn": True,
            "attn_heads": 4,
            "attn_dropout": 0.1,
        }
        if model_cfg:
            self.model_cfg.update(model_cfg)

        self.label_cfg = {
            "breakout_lookback": 20,
            "breakout_pct": 0.01,
            "vol_z_threshold": 0.0,
            "confirm_days": 0,
        }
        if label_cfg:
            self.label_cfg.update(label_cfg)

        self.window = window
        self.pred_horizon = pred_horizon
        self.date_col = date_col
        self.device = pick_device(device)
        self.checkpoint = checkpoint
        self._pending_checkpoint = checkpoint
        self._news_vectors = self._load_news_vectors(news_json) if news_json else None
        self.scaler = None
        self.encoder: Optional[GAT_GRU_Encoder] = None
        self.model: Optional[SupervisedModel] = None
        self.input_dim: Optional[int] = None

    @staticmethod
    def _load_news_vectors(path: str) -> Dict[str, Iterable[float]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): v for k, v in data.items()}

    def _init_model(self, in_dim: int, news_dim: Optional[int] = None) -> None:
        cfg = self.model_cfg
        encoder = GAT_GRU_Encoder(
            in_dim=in_dim,
            hid=cfg.get("hid", 64),
            heads=cfg.get("heads", 2),
            gru_hid=cfg.get("gru_hid", 64),
            proj_dim=cfg.get("proj_dim", cfg.get("hid", 64)),
            dropout=cfg.get("dropout", 0.2),
            num_gru_layers=cfg.get("num_gru_layers", 2),
            bidirectional=cfg.get("bidirectional", True),
            num_gat_layers=cfg.get("num_gat_layers", 3),
            use_temporal_attn=cfg.get("use_temporal_attn", True),
            attn_heads=cfg.get("attn_heads", 4),
            attn_dropout=cfg.get("attn_dropout", 0.1),
        ).to(self.device)
        model = SupervisedModel(encoder, news_dim=news_dim, dropout=cfg.get("dropout", 0.2)).to(self.device)
        encoder.eval()
        model.eval()
        self.encoder = encoder
        self.model = model
        self.input_dim = in_dim
        if self._pending_checkpoint:
            self.load_checkpoint(self._pending_checkpoint)
            self._pending_checkpoint = None

    def load_checkpoint(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        if self.model is None:
            self._pending_checkpoint = path
            return
        state = torch.load(path, map_location=self.device)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"Warning: missing parameters during load: {missing}")
        if unexpected:
            print(f"Warning: unexpected parameters during load: {unexpected}")

    # ------------------------------------------------------------------
    # Data preparation helpers

    def prepare_features(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path, parse_dates=[self.date_col])
        if self.date_col not in df.columns:
            raise ValueError(f"Column '{self.date_col}' not found in {csv_path}")
        df = df.sort_values(self.date_col).set_index(self.date_col)
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns {missing} in {csv_path}")
        base = df[list(required)].copy()
        feat = compute_features(base)
        feat = add_pattern_scores(feat)
        feat["breakout_label"] = make_breakout_labels(
            feat,
            self.label_cfg["breakout_lookback"],
            self.label_cfg["vol_z_threshold"],
            self.label_cfg["breakout_pct"],
            self.label_cfg["confirm_days"],
        )
        feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
        return feat

    def build_dataset(self, df_feat: pd.DataFrame) -> Optional[OHLCVGraphDataset]:
        if len(df_feat) < self.window + self.pred_horizon:
            return None
        news_vectors = self._prepare_news(df_feat)
        dataset = OHLCVGraphDataset(
            df_feat,
            window=self.window,
            pred_horizon=self.pred_horizon,
            use_labels=False,
            scaler=self.scaler,
            news_feat=news_vectors,
        )
        if self.scaler is None:
            self.scaler = dataset.scaler
        return dataset

    def latest_sample(self, dataset: OHLCVGraphDataset):
        if dataset is None or len(dataset) == 0:
            return None
        return dataset[len(dataset) - 1]

    def infer(self, df_feat: pd.DataFrame) -> Optional[Dict[str, float]]:
        dataset = self.build_dataset(df_feat)
        sample = self.latest_sample(dataset)
        if sample is None:
            return None
        in_dim = sample.x.shape[-1]
        news_dim = sample.news.shape[-1] if hasattr(sample, "news") else None
        if self.model is None:
            self._init_model(in_dim, news_dim=news_dim)
        elif self.input_dim is not None and in_dim != self.input_dim:
            raise ValueError("Input dimension changed; restart application to rebuild the model.")
        device_data = sample.to(self.device)
        logits, ret, vol = self.model(device_data)
        prob = torch.softmax(logits, dim=-1)[0, 1].item()
        exp_ret = ret.item()
        exp_vol = vol.item()
        last_close = float(df_feat["close"].iloc[-1]) if "close" in df_feat.columns else float("nan")
        news_vec = sample.news.cpu().numpy() if hasattr(sample, "news") else None
        return {
            "probability": prob,
            "expected_return": exp_ret,
            "volatility": exp_vol,
            "last_close": last_close,
            "news_vector": news_vec,
            "dataset": dataset,
        }

    # Backwards compatibility helpers ---------------------------------

    def _prepare_news(self, df: pd.DataFrame) -> Optional[Dict[str, Iterable[float]]]:
        if not self._news_vectors:
            return None
        if isinstance(df.index, pd.DatetimeIndex):
            return {str(idx.date()): self._news_vectors.get(str(idx.date())) for idx in df.index}
        return None

    @torch.no_grad()
    def score_ticker(self, csv_path: str, ticker: str) -> Optional[Dict[str, float]]:
        df_feat = self.prepare_features(csv_path)
        result = self.infer(df_feat)
        if result is None:
            print(f"Not enough data to score {ticker}. Need at least {self.window} rows.")
            return None
        result = result.copy()
        result.pop("dataset", None)
        return result


class AnalysisCoordinator:
    """Runs the roster of agents and collates their reports."""

    def __init__(self, backend: GNNBackend, macro_view: Optional[Dict[str, float]] = None):
        self.backend = backend
        self.macro_view = macro_view or {}
        self.agents = default_agents()
        self.engine = DecisionEngine()

    def update_macro(self, **kwargs: float) -> None:
        self.macro_view.update(kwargs)

    def evaluate(self, ticker: str, csv_path: str, holding: Optional[Holding]) -> Optional[Dict[str, object]]:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found for {ticker}: {csv_path}")
        df_feat = self.backend.prepare_features(csv_path)
        inference = self.backend.infer(df_feat)
        if inference is None:
            return None
        dataset = inference.pop("dataset")
        latest_row = df_feat.iloc[-1]
        context = AgentContext(
            ticker=ticker,
            features=df_feat,
            latest_row=latest_row,
            holding=holding,
            model_score={k: v for k, v in inference.items() if k not in {"news_vector"}},
            news_vector=inference.get("news_vector"),
            macro_view=self.macro_view,
        )
        reports = [agent.analyze(context) for agent in self.agents]
        decision = self.engine.aggregate(reports)
        expected_price = inference["last_close"] * (1 + inference["expected_return"])
        return {
            "ticker": ticker,
            "holding": holding,
            "features": df_feat,
            "dataset": dataset,
            "model": inference,
            "reports": reports,
            "decision": decision,
            "expected_price": expected_price,
        }


class FundManagerShell(cmd.Cmd):
    """cmd-based user interface providing discoverable commands."""

    intro = textwrap.dedent(
        """
        Welcome to the multi-agent fund manager!  Type 'help' to see available
        commands.  Common tasks: 'add', 'holdings', 'score', 'summary', 'macro'.
        """
    )
    prompt = "fund-manager> "

    def __init__(self, db: PortfolioDB, coordinator: AnalysisCoordinator, data_template: str):
        super().__init__()
        self.db = db
        self.coordinator = coordinator
        self.data_template = data_template

    # ------------- Helper formatting routines -------------------------

    @staticmethod
    def _format_money(value: float) -> str:
        return "$" + format(value, ",.2f") if np.isfinite(value) else "N/A"

    @staticmethod
    def _format_pct(value: float) -> str:
        return f"{value*100:,.2f}%" if np.isfinite(value) else "N/A"

    def _print_holdings(self) -> None:
        holdings = self.db.list_holdings()
        if not holdings:
            print("No holdings recorded yet.")
            return
        print("\nTicker   Shares       Cost Basis")
        print("---------------------------------")
        for h in holdings:
            print(f"{h.ticker:<7} {h.shares:>10.2f}   {h.cost_basis:>10.2f}")

    def _print_analysis(self, analysis: Dict[str, object]) -> None:
        model = analysis["model"]
        reports = analysis["reports"]
        decision = analysis["decision"]
        holding = analysis["holding"]
        print("\n=== Final Decision ===")
        print(f"Ticker: {analysis['ticker']}")
        print(f"Recommended action: {decision['decision']}")
        print(f"Composite score: {decision['composite_score']:.3f}")
        print("Rationale:")
        for line in decision["rationale"]:
            print(f"  - {line}")
        print("\nModel snapshot:")
        print(f"  Last close: {self._format_money(model['last_close'])}")
        print(f"  Expected return: {self._format_pct(model['expected_return'])}")
        print(f"  Volatility: {self._format_pct(model['volatility'])}")
        print(f"  Breakout probability: {self._format_pct(model['probability'])}")
        print(f"  Expected price: {self._format_money(analysis['expected_price'])}")
        if holding:
            unrealized = (model['last_close'] - holding.cost_basis) / (holding.cost_basis + 1e-9)
            print(f"  Holding: {holding.shares} @ {holding.cost_basis} ({self._format_pct(unrealized)})")
        print("\nAgent breakdown:")
        for rep in reports:
            print(f"* {rep.name} -> {rep.action} | score={rep.score:.2f} | confidence={rep.confidence:.2f}")
            for insight in rep.insights:
                print(f"    - {insight}")

    # ---------------------- Commands ---------------------------------

    def do_add(self, arg: str) -> None:
        """add TICKER SHARES COST -- add or update a holding."""

        parts = arg.split()
        if len(parts) != 3:
            print("Usage: add TICKER SHARES COST")
            return
        ticker, shares, cost = parts
        try:
            holding = Holding(ticker=ticker.upper(), shares=float(shares), cost_basis=float(cost))
        except ValueError:
            print("Shares and cost must be numeric.")
            return
        self.db.upsert(holding)
        print(f"Saved holding for {holding.ticker}.")

    def do_remove(self, arg: str) -> None:
        """remove TICKER -- delete a holding."""

        ticker = arg.strip().upper()
        if not ticker:
            print("Usage: remove TICKER")
            return
        self.db.remove(ticker)
        print(f"Removed {ticker} from holdings.")

    def do_holdings(self, arg: str) -> None:  # pylint: disable=unused-argument
        """List all stored holdings."""

        self._print_holdings()

    do_list = do_holdings  # alias for convenience

    def do_macro(self, arg: str) -> None:
        """macro KEY=VALUE [KEY=VALUE ...] -- update macro view percentages."""

        if not arg:
            print("Current macro view:")
            for k, v in self.coordinator.macro_view.items():
                print(f"  {k}: {self._format_pct(v)}")
            return
        updates = {}
        for token in arg.split():
            if "=" not in token:
                print(f"Ignoring malformed token: {token}")
                continue
            key, val = token.split("=", 1)
            try:
                updates[key] = float(val)
            except ValueError:
                print(f"Could not parse value for {key}")
        if updates:
            self.coordinator.update_macro(**updates)
            print("Updated macro indicators.")

    def do_score(self, arg: str) -> None:
        """score [TICKER ...] -- run the agent ensemble for the given tickers."""

        tickers = [tok.strip().upper() for tok in arg.split()] if arg else []
        if not tickers:
            tickers = [h.ticker for h in self.db.list_holdings()]
        if not tickers:
            print("No tickers provided or stored in the database.")
            return
        holdings_map = {h.ticker: h for h in self.db.list_holdings()}
        for ticker in tickers:
            csv_path = self.data_template.format(ticker=ticker)
            holding = holdings_map.get(ticker)
            try:
                analysis = self.coordinator.evaluate(ticker, csv_path, holding)
            except Exception as exc:  # pragma: no cover - interactive feedback
                print(f"Failed to score {ticker}: {exc}")
                continue
            if analysis is None:
                print(f"Not enough data for {ticker}.")
                continue
            self._print_analysis(analysis)

    def do_summary(self, arg: str) -> None:  # pylint: disable=unused-argument
        """Show aggregated portfolio value and implied change."""

        holdings = self.db.list_holdings()
        if not holdings:
            print("No holdings recorded yet.")
            return
        total_value = 0.0
        expected_value = 0.0
        for holding in holdings:
            csv_path = self.data_template.format(ticker=holding.ticker)
            try:
                analysis = self.coordinator.evaluate(holding.ticker, csv_path, holding)
            except Exception as exc:  # pragma: no cover - interactive feedback
                print(f"Skipping {holding.ticker}: {exc}")
                continue
            if not analysis:
                continue
            price = analysis["model"]["last_close"]
            total_value += holding.shares * price
            expected_price = analysis["expected_price"]
            expected_value += holding.shares * expected_price
        if total_value == 0:
            print("Portfolio value is zero or data unavailable.")
            return
        change = ((expected_value - total_value) / total_value) * 100
        print(f"Current portfolio value: {self._format_money(total_value)}")
        print(f"Model-implied change over horizon: {change:,.2f}%")

    def do_export(self, arg: str) -> None:
        """export PATH.json -- dump the latest holdings and macro settings."""

        path = arg.strip()
        if not path:
            print("Usage: export PATH.json")
            return
        payload = {
            "holdings": [h.__dict__ for h in self.db.list_holdings()],
            "macro_view": self.coordinator.macro_view,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Exported data to {path}.")

    def do_agents(self, arg: str) -> None:  # pylint: disable=unused-argument
        """List the active agent roster."""

        print("Active agents:")
        for agent in self.coordinator.agents:
            print(f"  - {agent.name} (weight={agent.weight})")

    def do_quit(self, arg: str) -> bool:  # pylint: disable=unused-argument
        """Exit the application."""

        print("Goodbye!")
        return True

    do_exit = do_quit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive fund manager powered by multi-agent GNN analysis")
    parser.add_argument("--db", default="fund_manager.db", help="SQLite file to store holdings")
    parser.add_argument("--data-template", default="data/{ticker}.csv", help="Path template for OHLCV CSV files")
    parser.add_argument("--checkpoint", default=None, help="Path to a trained supervised model state dict")
    parser.add_argument("--device", default="auto", help="Device to run inference on")
    parser.add_argument("--window", type=int, default=60, help="Sequence length window for the encoder")
    parser.add_argument("--pred-horizon", type=int, default=5, help="Prediction horizon used during training")
    parser.add_argument("--model-json", default=None, help="Optional JSON file with model hyperparameters")
    parser.add_argument("--labels-json", default=None, help="Optional JSON file with label generation params")
    parser.add_argument("--news-json", default=None, help="Optional JSON mapping YYYY-MM-DD to news embedding vectors")
    parser.add_argument("--macro-json", default=None, help="Optional JSON file with macro indicator defaults")
    parser.add_argument("--date-col", default="date", help="Date column in the OHLCV CSV files")
    return parser.parse_args()


def load_json(path: Optional[str]) -> Optional[Dict]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    model_cfg = load_json(args.model_json)
    label_cfg = load_json(args.labels_json)
    macro_cfg = load_json(args.macro_json) or {}
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
    shell = FundManagerShell(db=db, coordinator=coordinator, data_template=args.data_template)
    shell.cmdloop()


if __name__ == "__main__":  # pragma: no cover
    main()
