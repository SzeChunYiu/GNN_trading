"""Graphical fund manager interface combining multi-agent analytics.

This module exposes a Tkinter-based GUI that mirrors the functionality of the
CLI fund manager while adding interactive tables, detail panes, and basic
visualisations.  It reuses the portfolio database, backend inference utilities,
and analysis coordinator from :mod:`gnn_trading.fund_manager` to avoid code
duplication.  The GUI is intentionally self-contained so that running
``python -m gnn_trading.fund_manager_gui`` launches a desktop-style interface.
"""

from __future__ import annotations

import argparse
import queue
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, simpledialog, ttk
from typing import Dict, Iterable, List, Optional

try:  # pragma: no cover - optional visual dependency
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except Exception:  # pragma: no cover - matplotlib optional
    FigureCanvasTkAgg = None  # type: ignore
    Figure = None  # type: ignore

from .fund_manager import AnalysisCoordinator, GNNBackend, Holding, PortfolioDB, load_json


@dataclass
class _AnalysisRow:
    """Container storing flattened metrics for the results table."""

    ticker: str
    action: str
    probability: float
    expected_return: float
    expected_price: float
    volatility: float
    risk_adjusted: float
    composite: float
    shares: float


class HoldingDialog(simpledialog.Dialog):
    """Simple modal dialog used for adding or editing holdings."""

    def __init__(self, parent: tk.Misc, title: str, holding: Optional[Holding] = None):
        self._holding = holding
        super().__init__(parent, title)

    def body(self, master: tk.Misc) -> Optional[tk.Widget]:  # pragma: no cover - GUI layout
        ttk.Label(master, text="Ticker").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Label(master, text="Shares").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Label(master, text="Cost Basis").grid(row=2, column=0, sticky="w", padx=4, pady=4)

        self.var_ticker = tk.StringVar(value=self._holding.ticker if self._holding else "")
        self.var_shares = tk.StringVar(value=str(self._holding.shares) if self._holding else "")
        self.var_cost = tk.StringVar(value=str(self._holding.cost_basis) if self._holding else "")

        ttk.Entry(master, textvariable=self.var_ticker).grid(row=0, column=1, padx=4, pady=4)
        ttk.Entry(master, textvariable=self.var_shares).grid(row=1, column=1, padx=4, pady=4)
        ttk.Entry(master, textvariable=self.var_cost).grid(row=2, column=1, padx=4, pady=4)
        return None

    def validate(self) -> bool:  # pragma: no cover - user interaction
        ticker = self.var_ticker.get().strip().upper()
        if not ticker:
            messagebox.showerror("Validation", "Ticker cannot be empty.")
            return False
        try:
            shares = float(self.var_shares.get())
            cost = float(self.var_cost.get())
        except ValueError:
            messagebox.showerror("Validation", "Shares and cost must be numeric values.")
            return False
        self.result = Holding(ticker=ticker, shares=shares, cost_basis=cost)
        return True


class FundManagerGUI:
    """Main window orchestrating holdings management and analytics display."""

    def __init__(
        self,
        root: tk.Tk,
        db: PortfolioDB,
        coordinator: AnalysisCoordinator,
        data_template: str,
    ) -> None:
        self.root = root
        self.db = db
        self.coordinator = coordinator
        self.data_template = data_template
        self.status_var = tk.StringVar(value="Ready")
        self.summary_var = tk.StringVar(value="Portfolio summary will appear here.")
        self._analysis_thread: Optional[threading.Thread] = None
        self._analysis_cache: Dict[str, Dict[str, object]] = {}
        self.figure = None
        self.chart_axes = None
        self.canvas = None

        self._build_layout()
        self.refresh_holdings()

    # ------------------------------------------------------------------
    # GUI layout helpers

    def _build_layout(self) -> None:  # pragma: no cover - GUI layout
        self.root.title("GNN Trading Fund Manager")
        self.root.geometry("1200x720")

        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # Holdings panel ------------------------------------------------
        holdings_frame = ttk.LabelFrame(main, text="Holdings")
        holdings_frame.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        holdings_frame.rowconfigure(0, weight=1)
        holdings_frame.columnconfigure(0, weight=1)

        self.holdings_tree = ttk.Treeview(
            holdings_frame,
            columns=("shares", "cost"),
            show="headings",
            height=12,
        )
        self.holdings_tree.heading("shares", text="Shares")
        self.holdings_tree.heading("cost", text="Cost Basis")
        self.holdings_tree.column("shares", width=80, anchor="e")
        self.holdings_tree.column("cost", width=90, anchor="e")
        self.holdings_tree.grid(row=0, column=0, sticky="nsew")

        holdings_scroll = ttk.Scrollbar(holdings_frame, orient=tk.VERTICAL, command=self.holdings_tree.yview)
        holdings_scroll.grid(row=0, column=1, sticky="ns")
        self.holdings_tree.configure(yscrollcommand=holdings_scroll.set)

        btn_frame = ttk.Frame(holdings_frame)
        btn_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=4)
        for idx, (label, handler) in enumerate(
            [
                ("Add", self._add_holding),
                ("Edit", self._edit_holding),
                ("Remove", self._remove_holding),
                ("Refresh", self.refresh_holdings),
            ]
        ):
            ttk.Button(btn_frame, text=label, command=handler).grid(row=0, column=idx, padx=2)

        ttk.Label(holdings_frame, textvariable=self.summary_var, wraplength=260, justify=tk.LEFT).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )

        # Results panel -------------------------------------------------
        results_frame = ttk.LabelFrame(main, text="Analysis Results")
        results_frame.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        results_frame.rowconfigure(1, weight=1)
        results_frame.columnconfigure(0, weight=1)

        control_frame = ttk.Frame(results_frame)
        control_frame.grid(row=0, column=0, sticky="ew")
        ttk.Button(control_frame, text="Analyze Holdings", command=self.analyze_holdings).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Analyze Selected", command=self.analyze_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Portfolio Summary", command=self.update_summary).pack(side=tk.LEFT, padx=2)
        ttk.Label(control_frame, textvariable=self.status_var).pack(side=tk.RIGHT)

        self.results_tree = ttk.Treeview(
            results_frame,
            columns=(
                "action",
                "prob",
                "exp_ret",
                "exp_price",
                "vol",
                "risk",
                "composite",
                "shares",
            ),
            show="headings",
            height=14,
        )
        headings = {
            "action": "Action",
            "prob": "Prob %",
            "exp_ret": "Exp Ret %",
            "exp_price": "Exp Price",
            "vol": "Vol %",
            "risk": "Risk Adj",
            "composite": "Composite",
            "shares": "Shares",
        }
        widths = {
            "action": 100,
            "prob": 80,
            "exp_ret": 90,
            "exp_price": 90,
            "vol": 80,
            "risk": 90,
            "composite": 90,
            "shares": 70,
        }
        for key, label in headings.items():
            self.results_tree.heading(key, text=label)
            self.results_tree.column(key, width=widths[key], anchor="e")
        self.results_tree.grid(row=1, column=0, sticky="nsew")

        res_scroll = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        res_scroll.grid(row=1, column=1, sticky="ns")
        self.results_tree.configure(yscrollcommand=res_scroll.set)
        self.results_tree.bind("<<TreeviewSelect>>", self._on_result_select)

        # Detail + visualisation panel ---------------------------------
        detail_frame = ttk.Frame(results_frame)
        detail_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=6)
        detail_frame.columnconfigure(1, weight=1)
        detail_frame.rowconfigure(0, weight=1)

        self.detail_text = tk.Text(detail_frame, height=12, width=50)
        self.detail_text.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.detail_text.configure(state=tk.DISABLED)

        chart_frame = ttk.Frame(detail_frame)
        chart_frame.grid(row=0, column=1, sticky="nsew")
        chart_frame.rowconfigure(0, weight=1)
        chart_frame.columnconfigure(0, weight=1)

        if Figure is not None and FigureCanvasTkAgg is not None:
            self.figure = Figure(figsize=(5, 3), dpi=100)
            self.chart_axes = self.figure.add_subplot(111)
            self.chart_axes.set_title("Expected Returns")
            self.chart_axes.set_xlabel("Ticker")
            self.chart_axes.set_ylabel("% Return")
            self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            ttk.Label(
                chart_frame,
                text="Install matplotlib for return visualisations.",
                justify=tk.CENTER,
            ).pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Holdings operations

    def refresh_holdings(self) -> None:
        for item in self.holdings_tree.get_children():
            self.holdings_tree.delete(item)
        for holding in self.db.list_holdings():
            self.holdings_tree.insert("", tk.END, iid=holding.ticker, values=(f"{holding.shares:.2f}", f"{holding.cost_basis:.2f}"))
        self.update_summary()

    def _add_holding(self) -> None:  # pragma: no cover - GUI event
        dialog = HoldingDialog(self.root, "Add Holding")
        if dialog.result:
            self.db.upsert(dialog.result)
            self.refresh_holdings()

    def _edit_holding(self) -> None:  # pragma: no cover - GUI event
        selection = self.holdings_tree.selection()
        if not selection:
            messagebox.showinfo("Edit", "Select a holding to edit.")
            return
        ticker = selection[0]
        rows = {h.ticker: h for h in self.db.list_holdings()}
        dialog = HoldingDialog(self.root, "Edit Holding", holding=rows[ticker])
        if dialog.result:
            self.db.upsert(dialog.result)
            self.refresh_holdings()

    def _remove_holding(self) -> None:  # pragma: no cover - GUI event
        selection = self.holdings_tree.selection()
        if not selection:
            messagebox.showinfo("Remove", "Select a holding to remove.")
            return
        ticker = selection[0]
        if messagebox.askyesno("Confirm", f"Remove {ticker} from holdings?"):
            self.db.remove(ticker)
            self.refresh_holdings()

    # ------------------------------------------------------------------
    # Analysis operations

    def analyze_holdings(self) -> None:  # pragma: no cover - GUI event
        tickers = [h.ticker for h in self.db.list_holdings()]
        if not tickers:
            messagebox.showinfo("Analyze", "No holdings available. Add a holding first.")
            return
        self._run_analysis(tickers)

    def analyze_selected(self) -> None:  # pragma: no cover - GUI event
        selection = self.holdings_tree.selection()
        if not selection:
            messagebox.showinfo("Analyze", "Select one or more holdings to analyze.")
            return
        tickers = [iid for iid in selection]
        self._run_analysis(tickers)

    def _run_analysis(self, tickers: Iterable[str]) -> None:
        if self._analysis_thread and self._analysis_thread.is_alive():
            messagebox.showinfo("Analyze", "Analysis already running.")
            return
        self.status_var.set("Analyzing...")
        q: queue.Queue[List[Dict[str, object]]] = queue.Queue()

        def worker() -> None:
            results: List[Dict[str, object]] = []
            for ticker in tickers:
                csv_path = self.data_template.format(ticker=ticker)
                holding_map = {h.ticker: h for h in self.db.list_holdings()}
                try:
                    analysis = self.coordinator.evaluate(ticker, csv_path, holding_map.get(ticker))
                    if analysis:
                        results.append(analysis)
                except Exception as exc:  # pragma: no cover - background feedback
                    results.append({"ticker": ticker, "error": str(exc)})
            q.put(results)

        self._analysis_thread = threading.Thread(target=worker, daemon=True)
        self._analysis_thread.start()
        self.root.after(200, self._poll_analysis_queue, q)

    def _poll_analysis_queue(self, q: queue.Queue[List[Dict[str, object]]]) -> None:
        if not q.empty():
            results = q.get()
            self._update_analysis(results)
            self.status_var.set("Completed")
        else:
            self.root.after(200, self._poll_analysis_queue, q)

    def _update_analysis(self, analyses: List[Dict[str, object]]) -> None:
        self._analysis_cache.clear()
        table_rows: List[_AnalysisRow] = []
        errors: List[str] = []
        for analysis in analyses:
            if "error" in analysis:
                errors.append(f"{analysis['ticker']}: {analysis['error']}")
                continue
            ticker = analysis["ticker"]
            self._analysis_cache[ticker] = analysis
            model = analysis["model"]
            decision = analysis["decision"]
            holding = analysis.get("holding")
            shares = holding.shares if holding else 0.0
            table_rows.append(
                _AnalysisRow(
                    ticker=ticker,
                    action=decision["decision"],
                    probability=float(model["probability"]),
                    expected_return=float(model["expected_return"]),
                    expected_price=float(analysis["expected_price"]),
                    volatility=float(model["volatility"]),
                    risk_adjusted=float(model["expected_return"] / (abs(model["volatility"]) + 1e-6)),
                    composite=float(decision["composite_score"]),
                    shares=shares,
                )
            )
        self._populate_results_table(table_rows)
        self._update_chart(table_rows)
        if errors:
            messagebox.showwarning("Analysis", "\n".join(errors))
        self.update_summary()

    def _populate_results_table(self, rows: List[_AnalysisRow]) -> None:
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        for row in sorted(rows, key=lambda r: r.risk_adjusted, reverse=True):
            self.results_tree.insert(
                "",
                tk.END,
                iid=row.ticker,
                values=(
                    row.action,
                    f"{row.probability*100:.2f}",
                    f"{row.expected_return*100:.2f}",
                    f"{row.expected_price:.2f}",
                    f"{row.volatility*100:.2f}",
                    f"{row.risk_adjusted:.3f}",
                    f"{row.composite:.3f}",
                    f"{row.shares:.2f}",
                ),
            )

    def _update_chart(self, rows: List[_AnalysisRow]) -> None:
        if not self.chart_axes or not self.canvas:
            return
        self.chart_axes.clear()
        self.chart_axes.set_title("Expected Returns")
        self.chart_axes.set_xlabel("Ticker")
        self.chart_axes.set_ylabel("% Return")
        if rows:
            tickers = [row.ticker for row in rows]
            returns = [row.expected_return * 100 for row in rows]
            colors = ["#2ca02c" if val >= 0 else "#d62728" for val in returns]
            self.chart_axes.bar(tickers, returns, color=colors)
            self.chart_axes.axhline(0, color="black", linewidth=0.8)
        self.chart_axes.figure.tight_layout()
        self.canvas.draw_idle()

    def _on_result_select(self, event: tk.Event) -> None:  # pragma: no cover - GUI event
        selection = self.results_tree.selection()
        if not selection:
            return
        ticker = selection[0]
        analysis = self._analysis_cache.get(ticker)
        if not analysis:
            return
        model = analysis["model"]
        decision = analysis["decision"]
        reports = analysis["reports"]
        self.detail_text.configure(state=tk.NORMAL)
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert(
            tk.END,
            f"Ticker: {ticker}\n"
            f"Action: {decision['decision']}\n"
            f"Composite score: {decision['composite_score']:.3f}\n"
            f"Probability: {model['probability']:.2%}\n"
            f"Expected return: {model['expected_return']:.2%}\n"
            f"Volatility: {model['volatility']:.2%}\n"
            f"Expected price: {analysis['expected_price']:.2f}\n\n"
            "Agent breakdown:\n",
        )
        for report in reports:
            self.detail_text.insert(
                tk.END,
                f"- {report.name}: {report.action} | score={report.score:.2f} | confidence={report.confidence:.2f}\n",
            )
            for insight in report.insights:
                self.detail_text.insert(tk.END, f"    • {insight}\n")
        self.detail_text.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Summary helpers

    def update_summary(self) -> None:
        holdings = self.db.list_holdings()
        if not holdings:
            self.summary_var.set("No holdings recorded. Add positions to begin analysis.")
            return
        total_value = 0.0
        expected_value = 0.0
        actions: List[str] = []
        for holding in holdings:
            analysis = self._analysis_cache.get(holding.ticker)
            if not analysis:
                continue
            price = analysis["model"]["last_close"]
            expected_price = analysis["expected_price"]
            total_value += holding.shares * price
            expected_value += holding.shares * expected_price
            actions.append(f"{holding.ticker}: {analysis['decision']['decision']}")
        if total_value <= 0 or expected_value <= 0:
            self.summary_var.set("Run analysis to populate the portfolio summary.")
            return
        change = (expected_value - total_value) / (total_value + 1e-9) * 100
        summary_lines = [
            f"Current value: ${total_value:,.2f}",
            f"Model-implied change: {change:,.2f}%",
        ]
        if actions:
            summary_lines.append("Actions: " + ", ".join(actions))
        self.summary_var.set("\n".join(summary_lines))


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the GUI entry point."""

    parser = argparse.ArgumentParser(description="Graphical fund manager powered by the multi-agent backend")
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


def main() -> None:  # pragma: no cover - manual execution
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

    root = tk.Tk()
    gui = FundManagerGUI(root=root, db=db, coordinator=coordinator, data_template=args.data_template)
    root.mainloop()
    return gui


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
