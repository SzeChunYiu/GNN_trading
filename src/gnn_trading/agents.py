"""Specialised analysis agents that collectively advise the fund manager.

Each agent inspects a different slice of information (price action, technicals,
news, macro back-drop) and returns a weighted report.  The fund manager then
combines the reports to suggest a final trade action.  The module is designed to
be lightweight so that new agents can be added without changing the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - hints only
    from .fund_manager import Holding

from .quantamental import ContextualQuantamentalAdapter, QuantamentalResearchAgent


@dataclass
class AgentReport:
    """Container for a single agent's opinion."""

    name: str
    score: float
    confidence: float
    weight: float
    action: str
    insights: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def weighted_score(self) -> float:
        """Combine score, confidence and weight for ranking."""

        return self.score * self.confidence * self.weight


@dataclass
class AgentContext:
    """Snapshot of the information passed to each agent."""

    ticker: str
    features: pd.DataFrame
    latest_row: pd.Series
    holding: Optional["Holding"]
    model_score: Optional[Dict[str, float]] = None
    news_vector: Optional[np.ndarray] = None
    macro_view: Optional[Dict[str, float]] = None


class BaseAnalysisAgent:
    """Base helper that gives every agent a consistent interface."""

    name: str = "Unnamed"
    weight: float = 1.0

    def analyze(self, context: AgentContext) -> AgentReport:
        raise NotImplementedError


class PriceActionAgent(BaseAnalysisAgent):
    """Wraps the GNN model output into a high-level recommendation."""

    name = "Price Action"
    weight = 1.2

    def analyze(self, context: AgentContext) -> AgentReport:
        score = context.model_score or {}
        prob = float(score.get("probability", 0.5))
        exp_ret = float(score.get("expected_return", 0.0))
        vol = float(score.get("volatility", 0.0))
        risk_adj = exp_ret / (abs(vol) + 1e-6)
        action = "WATCH"
        if prob > 0.65 and exp_ret > 0.03:
            action = "BUY"
        elif prob < 0.4 or exp_ret < 0.0:
            action = "AVOID"
        insights = [
            f"Model breakout probability {prob:.2%}",
            f"Expected return {exp_ret:.2%}",
            f"Volatility estimate {vol:.2%}",
        ]
        return AgentReport(
            name=self.name,
            score=risk_adj,
            confidence=min(max(prob, 0.1), 0.99),
            weight=self.weight,
            action=action,
            insights=insights,
            metrics={
                "probability": prob,
                "expected_return": exp_ret,
                "volatility": vol,
                "risk_adjusted": risk_adj,
            },
        )


class TechnicalAgent(BaseAnalysisAgent):
    """Uses handcrafted indicators such as RSI and MACD."""

    name = "Technical Analysis"
    weight = 1.0

    def analyze(self, context: AgentContext) -> AgentReport:
        latest = context.latest_row
        rsi = float(latest.get("rsi14", 50.0))
        macd_hist = float(latest.get("macd_hist", 0.0))
        slope = float(latest.get("slope_close_10", 0.0))
        compression = float(latest.get("compr_10", 0.0))
        bullish = int(rsi > 55) + int(macd_hist > 0) + int(slope > 0)
        bearish = int(rsi < 45) + int(macd_hist < 0) + int(slope < 0)
        score = (bullish - bearish) / 3.0
        confidence = 0.6 + min(abs(compression), 0.4)
        action = "NEUTRAL"
        if score > 0.3:
            action = "ACCUMULATE"
        elif score < -0.3:
            action = "REDUCE"
        insights = [
            f"RSI14 at {rsi:.1f}",
            f"MACD histogram {macd_hist:.3f}",
            f"10-day slope {slope:.4f}",
            f"Compression ratio {compression:.3f}",
        ]
        return AgentReport(
            name=self.name,
            score=score,
            confidence=confidence,
            weight=self.weight,
            action=action,
            insights=insights,
            metrics={
                "rsi14": rsi,
                "macd_hist": macd_hist,
                "slope_close_10": slope,
                "compression": compression,
            },
        )


class NewsSentimentAgent(BaseAnalysisAgent):
    """Interprets optional news vectors as a momentum cue."""

    name = "News Sentiment"
    weight = 0.8

    def analyze(self, context: AgentContext) -> AgentReport:
        if context.news_vector is None:
            return AgentReport(
                name=self.name,
                score=0.0,
                confidence=0.1,
                weight=self.weight,
                action="N/A",
                insights=["No news vector available; sentiment neutral."],
            )
        vec = np.asarray(context.news_vector)
        polarity = float(np.tanh(vec.mean()))
        volatility = float(vec.std())
        confidence = min(0.9, 0.5 + volatility)
        action = "POSITIVE" if polarity > 0.1 else "NEGATIVE" if polarity < -0.1 else "NEUTRAL"
        insights = [
            f"Average sentiment {polarity:.2f}",
            f"Dispersion {volatility:.2f}",
        ]
        return AgentReport(
            name=self.name,
            score=polarity,
            confidence=confidence,
            weight=self.weight,
            action=action,
            insights=insights,
            metrics={"sentiment": polarity, "dispersion": volatility},
        )


class MacroAgent(BaseAnalysisAgent):
    """Scores the macro back-drop based on user supplied indicators."""

    name = "Macro"
    weight = 0.9

    def analyze(self, context: AgentContext) -> AgentReport:
        macro = context.macro_view or {}
        if not macro:
            return AgentReport(
                name=self.name,
                score=0.0,
                confidence=0.2,
                weight=self.weight,
                action="NEUTRAL",
                insights=["No macro context configured."],
            )
        growth = float(macro.get("gdp_growth", 0.0))
        inflation = float(macro.get("inflation", 0.0))
        rates = float(macro.get("rate_trend", 0.0))
        macro_score = growth - max(inflation - 0.02, 0) - rates * 0.5
        action = "RISK-ON" if macro_score > 0 else "RISK-OFF"
        confidence = 0.5 + min(abs(macro_score), 0.5)
        insights = [
            f"GDP growth {growth:.2%}",
            f"Inflation {inflation:.2%}",
            f"Rate trend {rates:.2%}",
        ]
        return AgentReport(
            name=self.name,
            score=macro_score,
            confidence=confidence,
            weight=self.weight,
            action=action,
            insights=insights,
            metrics={
                "gdp_growth": growth,
                "inflation": inflation,
                "rate_trend": rates,
            },
        )


class QuantamentalAgent(BaseAnalysisAgent):
    """Autonomous quantamental agent combining quant + fundamental signals."""

    name = "Quantamental"
    weight = 1.1

    def __init__(
        self,
        research_agent: Optional[QuantamentalResearchAgent] = None,
        contextual_adapter: Optional[ContextualQuantamentalAdapter] = None,
    ) -> None:
        self.research_agent = research_agent
        self.contextual_adapter = contextual_adapter or ContextualQuantamentalAdapter()

    def analyze(self, context: AgentContext) -> AgentReport:
        composite_score = 0.0
        insights: List[str] = []
        metrics: Dict[str, float] = {}
        confidence = 0.5
        if self.research_agent is not None:
            try:
                reports = self.research_agent.analyze([context.ticker], as_of=context.features.index[-1])
            except Exception as exc:  # pragma: no cover - defensive fallback
                insights.append(f"Quantamental engine fallback: {exc}")
                reports = {}
            if reports:
                report = reports.get(context.ticker)
                if report:
                    composite_score = float(report["composite_score"])
                    confidence = 0.6 + min(abs(composite_score), 0.3)
                    metrics.update({f"quant_{k}": float(v) for k, v in report.get("fundamental_snapshot", {}).items()})
                    top_features = list(report.get("feature_importance", {}).items())[:5]
                    for feat, val in top_features:
                        insights.append(f"Importance {feat}: {val:.3f}")
                    diagnostics = report.get("diagnostics", {})
                    if diagnostics:
                        insights.append(
                            "IC {:.3f}".format(float(diagnostics.get("ic", 0.0)))
                        )
        if not insights:
            composite_score, contrib = self.contextual_adapter.score_from_context(context.features)
            metrics.update({f"feature_{k}": v for k, v in contrib.items()})
            insights.append("Contextual adapter used for quantamental view.")
        action = "ACCUMULATE"
        if composite_score > 0.5:
            action = "STRONG BUY"
        elif composite_score < -0.3:
            action = "UNDERWEIGHT"
        elif composite_score < 0.1:
            action = "NEUTRAL"
        return AgentReport(
            name=self.name,
            score=composite_score,
            confidence=confidence,
            weight=self.weight,
            action=action,
            insights=insights,
            metrics=metrics,
        )


class DecisionEngine:
    """Combine agent reports into a final recommendation."""

    thresholds = [
        ("BUY", 0.45),
        ("ADD", 0.35),
        ("HOLD", -0.05),
        ("TRIM", -0.20),
        ("SELL", -0.35),
    ]

    def aggregate(self, reports: Iterable[AgentReport]) -> Dict[str, str]:
        reports = list(reports)
        composite = sum(r.weighted_score for r in reports)
        decision = "SELL"
        for action, threshold in self.thresholds:
            if composite >= threshold:
                decision = action
                break
        rationale = [
            f"{r.name}: {r.action} (score={r.score:.2f}, conf={r.confidence:.2f})"
            for r in reports
        ]
        return {
            "decision": decision,
            "composite_score": composite,
            "rationale": rationale,
        }


def default_agents() -> List[BaseAnalysisAgent]:
    """Factory that yields the built-in multi-agent roster."""

    return [
        PriceActionAgent(),
        TechnicalAgent(),
        NewsSentimentAgent(),
        MacroAgent(),
        QuantamentalAgent(),
    ]
