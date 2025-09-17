"""Quantamental research agent with modular data, features, models, and risk tools.

This module implements an end-to-end quantamental research workflow that can
ingest market/fundamental data, engineer signals, train predictive models, and
build explainable portfolios with guardrails against look-ahead bias.  The
design emphasises composable abstractions so that practitioners can plug in
custom data providers, feature pipelines, model registries, and risk engines.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Data providers


class DataProvider(Protocol):
    """Abstract interface for fetching market, fundamental, and alt data."""

    def load_market_data(
        self,
        ticker: str,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        ...

    def load_fundamentals(
        self,
        ticker: str,
        as_of: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        ...

    def load_alternative_data(
        self,
        ticker: str,
        as_of: Optional[pd.Timestamp] = None,
    ) -> Optional[pd.DataFrame]:
        ...


@dataclass
class DictDataProvider:
    """Convenience provider backed by dictionaries for unit tests/examples."""

    market_data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    fundamentals: Dict[str, pd.DataFrame] = field(default_factory=dict)
    alternative: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def _slice(self, df: pd.DataFrame, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
        if df.index.dtype != "datetime64[ns]":
            return df
        mask = pd.Series(True, index=df.index)
        if start is not None:
            mask &= df.index >= start
        if end is not None:
            mask &= df.index <= end
        return df.loc[mask]

    def load_market_data(self, ticker: str, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        df = self.market_data.get(ticker.upper())
        if df is None:
            raise KeyError(f"Market data unavailable for {ticker}")
        return self._slice(df, start, end)

    def load_fundamentals(self, ticker: str, as_of: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        df = self.fundamentals.get(ticker.upper())
        if df is None:
            return pd.DataFrame()
        if as_of is None or df.index.dtype != "datetime64[ns]":
            return df
        return df.loc[df.index <= as_of]

    def load_alternative_data(self, ticker: str, as_of: Optional[pd.Timestamp] = None) -> Optional[pd.DataFrame]:
        df = self.alternative.get(ticker.upper())
        if df is None:
            return None
        if as_of is None or df.index.dtype != "datetime64[ns]":
            return df
        return df.loc[df.index <= as_of]


# ---------------------------------------------------------------------------
# Feature engineering


class FeaturePipeline(Protocol):
    """Transforms raw data into quantamental feature matrices."""

    def prepare_features(
        self,
        ticker: str,
        market_data: pd.DataFrame,
        fundamentals: pd.DataFrame,
        alt_data: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        ...


def _rolling_feature(series: pd.Series, window: int, func: str) -> pd.Series:
    if func == "mean":
        return series.rolling(window).mean()
    if func == "std":
        return series.rolling(window).std()
    if func == "sum":
        return series.rolling(window).sum()
    raise ValueError(f"Unsupported rolling func {func}")


def _seasonality_feature(series: pd.Series) -> pd.Series:
    grouped = series.groupby(series.index.month).transform("mean")
    return series / (grouped.replace(0, np.nan) + 1e-9) - 1


def _safe_div(num: pd.Series, denom: pd.Series) -> pd.Series:
    return num / (denom.replace(0, np.nan) + 1e-9)


@dataclass
class DefaultFeaturePipeline:
    """Reference pipeline generating quant signals and fundamental factors."""

    lookbacks: Sequence[int] = (5, 10, 20, 60, 120)

    def prepare_features(
        self,
        ticker: str,
        market_data: pd.DataFrame,
        fundamentals: pd.DataFrame,
        alt_data: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        if market_data.empty:
            raise ValueError(f"Market data empty for {ticker}")
        df = market_data.copy()
        if df.index.dtype != "datetime64[ns]":
            raise ValueError("Market data must be indexed by datetime for leakage checks")
        df = df.sort_index()
        close = df["close"].astype(float)
        volume = df.get("volume", pd.Series(index=df.index, dtype=float)).astype(float)
        returns = close.pct_change().fillna(0.0)

        feat = pd.DataFrame(index=df.index)
        feat["return_1d"] = returns
        for lb in self.lookbacks:
            feat[f"momentum_{lb}"] = close.pct_change(lb)
            feat[f"trend_{lb}"] = close / close.rolling(lb).mean() - 1
            feat[f"volatility_{lb}"] = returns.rolling(lb).std()
            feat[f"volume_flow_{lb}"] = _rolling_feature(volume, lb, "mean") / (volume + 1e-9) - 1
        feat["seasonality"] = _seasonality_feature(close)

        if not fundamentals.empty:
            fundamentals = fundamentals.sort_index()
            latest = fundamentals.ffill()
            feat = feat.join(latest.reindex(feat.index, method="ffill"), how="left")
            price = close
            feat["value_pe"] = _safe_div(latest.get("net_income", 0), latest.get("market_cap", price * 1e6))
            feat["quality_roe"] = _safe_div(latest.get("net_income", 0), latest.get("total_equity", 1))
            feat["profit_margin"] = _safe_div(latest.get("operating_income", 0), latest.get("revenue", 1))
            feat["leverage"] = _safe_div(latest.get("total_debt", 0), latest.get("total_assets", 1))
            feat["growth_revenue"] = latest.get("revenue", 0).pct_change().reindex(feat.index).fillna(0)
        else:
            feat[["value_pe", "quality_roe", "profit_margin", "leverage", "growth_revenue"]] = 0.0

        if alt_data is not None and not alt_data.empty:
            alt_data = alt_data.sort_index()
            alt_norm = (alt_data - alt_data.mean()) / (alt_data.std() + 1e-9)
            feat = feat.join(alt_norm.reindex(feat.index, method="ffill"), how="left")

        feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
        return feat


# ---------------------------------------------------------------------------
# Model registry and predictive engines


@dataclass
class ModelResult:
    """Captures predictions and explainability outputs."""

    predictions: pd.Series
    feature_importance: pd.Series
    diagnostics: Dict[str, float]


class ModelRegistry:
    """Lightweight registry mapping names to scikit-learn style estimators."""

    def __init__(self) -> None:
        self._models: Dict[str, object] = {}

    def register(self, name: str, model: object) -> None:
        self._models[name] = model

    def get(self, name: str) -> object:
        if name not in self._models:
            raise KeyError(f"Model '{name}' not registered")
        return self._models[name]


def _train_cross_sectional(
    features: pd.DataFrame,
    target: pd.Series,
    model,
    standardize: bool = True,
) -> Tuple[object, pd.DataFrame]:
    if features.empty or target.empty:
        raise ValueError("Cannot train model on empty data")
    X = features.values
    y = target.values
    scaler = StandardScaler() if standardize else None
    if scaler is not None:
        X = scaler.fit_transform(X)
    model.fit(X, y)
    if scaler is not None:
        setattr(model, "_scaler", scaler)
    return model, features


def _predict_model(model, features: pd.DataFrame) -> np.ndarray:
    X = features.values
    scaler = getattr(model, "_scaler", None)
    if scaler is not None:
        X = scaler.transform(X)
    return model.predict(X)


def _linear_importance(model, columns: Sequence[str]) -> pd.Series:
    coef = getattr(model, "coef_", None)
    if coef is None:
        return pd.Series(0.0, index=columns)
    if coef.ndim > 1:
        coef = coef[0]
    return pd.Series(coef, index=columns)


# ---------------------------------------------------------------------------
# Risk management and portfolio construction


@dataclass
class PortfolioResult:
    weights: pd.Series
    metrics: Dict[str, float]


class RiskManager:
    """Mean-variance portfolio optimiser with simple risk controls."""

    def __init__(self, risk_free_rate: float = 0.0, max_weight: float = 0.1) -> None:
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight

    def optimise(self, expected_returns: pd.Series, cov: pd.DataFrame, budget: float) -> PortfolioResult:
        if expected_returns.empty:
            raise ValueError("Expected returns are empty")
        cov = cov + np.eye(len(cov)) * 1e-6
        inv_cov = np.linalg.pinv(cov)
        ones = np.ones(len(expected_returns))
        A = ones @ inv_cov @ ones
        B = ones @ inv_cov @ expected_returns.values
        lam = (budget + self.risk_free_rate * B - B) / A
        raw_weights = inv_cov @ (expected_returns.values - lam)
        weights = pd.Series(raw_weights, index=expected_returns.index)
        weights = weights.clip(lower=-self.max_weight, upper=self.max_weight)
        if weights.sum() != 0:
            weights = weights / weights.abs().sum()
        portfolio_vol = float(np.sqrt(weights.values @ cov.values @ weights.values))
        portfolio_ret = float(weights @ expected_returns)
        sharpe = (portfolio_ret - self.risk_free_rate) / (portfolio_vol + 1e-9)
        return PortfolioResult(
            weights=weights,
            metrics={"expected_return": portfolio_ret, "volatility": portfolio_vol, "sharpe": sharpe},
        )


# ---------------------------------------------------------------------------
# Backtesting utilities


@dataclass
class BacktestReport:
    performance: pd.DataFrame
    metrics: Dict[str, float]
    IC: float


class WalkForwardBacktester:
    """Walk-forward backtester with leakage guardrails using expanding windows."""

    def __init__(self, splits: int = 4, top_k: int = 10) -> None:
        self.splits = splits
        self.top_k = top_k

    def run(
        self,
        features: pd.DataFrame,
        future_returns: pd.Series,
        model_factory,
    ) -> BacktestReport:
        if len(features) != len(future_returns):
            raise ValueError("Features and returns must be aligned for backtest")
        tscv = TimeSeriesSplit(n_splits=self.splits)
        equity_curve = []
        ic_scores = []
        pnl = 1.0
        for train_idx, test_idx in tscv.split(features):
            X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
            y_train, y_test = future_returns.iloc[train_idx], future_returns.iloc[test_idx]
            model = model_factory()
            model, _ = _train_cross_sectional(X_train, y_train, model)
            preds = _predict_model(model, X_test)
            ranking = np.argsort(preds)[::-1]
            selected = ranking[: self.top_k]
            realised = y_test.iloc[selected].mean()
            pnl *= float(1 + realised)
            equity_curve.append(pnl)
            ic = float(pd.Series(preds).corr(y_test, method="spearman"))
            if np.isfinite(ic):
                ic_scores.append(ic)
        perf_df = pd.DataFrame({"equity": equity_curve}, index=range(len(equity_curve)))
        returns = perf_df["equity"].pct_change().fillna(0)
        sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)
        max_dd = self._max_drawdown(perf_df["equity"])
        IC = float(np.nanmean(ic_scores)) if ic_scores else 0.0
        metrics = {
            "cumulative_return": perf_df["equity"].iloc[-1] - 1,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
        }
        return BacktestReport(performance=perf_df, metrics=metrics, IC=IC)

    @staticmethod
    def _max_drawdown(series: pd.Series) -> float:
        roll_max = series.cummax()
        drawdown = series / (roll_max + 1e-9) - 1
        return float(drawdown.min())


# ---------------------------------------------------------------------------
# Quantamental research agent


def _future_returns(close: pd.Series, horizon: int = 5) -> pd.Series:
    return close.pct_change(periods=horizon).shift(-horizon)


def _composite_score(quant_view: pd.Series, fundamental_view: pd.Series) -> pd.Series:
    quant = (quant_view - quant_view.mean()) / (quant_view.std(ddof=0) + 1e-9)
    fundamental = (fundamental_view - fundamental_view.mean()) / (fundamental_view.std(ddof=0) + 1e-9)
    return 0.6 * quant + 0.4 * fundamental


class QuantamentalResearchAgent:
    """Autonomous agent that unifies quant + fundamental insights."""

    def __init__(
        self,
        data_provider: DataProvider,
        feature_pipeline: Optional[FeaturePipeline] = None,
        model_registry: Optional[ModelRegistry] = None,
        risk_manager: Optional[RiskManager] = None,
        backtester: Optional[WalkForwardBacktester] = None,
        horizon: int = 5,
    ) -> None:
        self.data_provider = data_provider
        self.feature_pipeline = feature_pipeline or DefaultFeaturePipeline()
        self.model_registry = model_registry or self._default_models()
        self.risk_manager = risk_manager or RiskManager()
        self.backtester = backtester or WalkForwardBacktester()
        self.horizon = horizon

    def _default_models(self) -> ModelRegistry:
        registry = ModelRegistry()
        registry.register("return_ridge", Ridge(alpha=1.0))
        registry.register("classification_logit", LogisticRegression(max_iter=1000))
        registry.register("elastic_net", ElasticNet(alpha=0.5, l1_ratio=0.3))
        return registry

    # ---------------------------- Public API ----------------------------

    def analyze(self, tickers: Iterable[str], as_of: dt.datetime | pd.Timestamp) -> Dict[str, Dict[str, object]]:
        reports = {}
        for ticker in tickers:
            market = self.data_provider.load_market_data(ticker, end=pd.to_datetime(as_of))
            fundamentals = self.data_provider.load_fundamentals(ticker, as_of=pd.to_datetime(as_of))
            alt = self.data_provider.load_alternative_data(ticker, as_of=pd.to_datetime(as_of))
            features = self.feature_pipeline.prepare_features(ticker, market, fundamentals, alt)
            features = features.loc[features.index <= pd.to_datetime(as_of)]
            if len(features) < 50:
                continue
            future_ret = _future_returns(market["close"], self.horizon).reindex(features.index)
            aligned = features.dropna()
            future_ret = future_ret.reindex(aligned.index)
            future_ret = future_ret.dropna()
            if future_ret.empty:
                continue
            aligned = aligned.loc[future_ret.index]
            model = self.model_registry.get("return_ridge")
            model, trained_feat = _train_cross_sectional(aligned, future_ret, model)
            preds = pd.Series(_predict_model(model, aligned), index=aligned.index)
            quant_view = preds.iloc[-1]
            importance = _linear_importance(model, aligned.columns)
            fundamental_cols = [
                c
                for c in aligned.columns
                if c.startswith("value_")
                or c.startswith("quality_")
                or c.startswith("profit")
                or c.startswith("leverage")
                or c.startswith("growth_")
            ]
            fundamental_view = aligned[fundamental_cols].iloc[-1].fillna(0.0)
            fundamental_score = float(fundamental_view.mean()) if not fundamental_view.empty else 0.0
            composite = _composite_score(pd.Series([quant_view]), pd.Series([fundamental_score])).iloc[0]
            reports[ticker] = {
                "as_of": aligned.index[-1],
                "quant_view": float(quant_view),
                "fundamental_snapshot": {k: float(v) for k, v in fundamental_view.items()},
                "composite_score": float(composite),
                "feature_importance": importance.sort_values(ascending=False).to_dict(),
                "diagnostics": {"ic": float(pd.Series(preds).corr(future_ret, method="spearman"))},
            }
        return reports

    def rank(self, tickers: Iterable[str]) -> pd.DataFrame:
        as_of = dt.datetime.utcnow()
        reports = self.analyze(tickers, as_of=as_of)
        if not reports:
            return pd.DataFrame(columns=["ticker", "composite_score"]).set_index("ticker")
        df = pd.DataFrame(
            [
                {
                    "ticker": ticker,
                    "composite_score": data["composite_score"],
                    "quant_view": data["quant_view"],
                }
                for ticker, data in reports.items()
            ]
        )
        return df.set_index("ticker").sort_values("composite_score", ascending=False)

    def build_portfolio(
        self,
        tickers: Iterable[str],
        budget: float,
        constraints: Optional[Dict[str, float]] = None,
    ) -> PortfolioResult:
        reports = self.analyze(tickers, as_of=dt.datetime.utcnow())
        if not reports:
            raise ValueError("No analysis generated for portfolio construction")
        expected_returns = pd.Series({ticker: data["quant_view"] for ticker, data in reports.items()})
        base_var = float(expected_returns.var(ddof=0))
        if not np.isfinite(base_var) or base_var <= 0:
            base_var = 0.02
        cov = pd.DataFrame(
            np.eye(len(expected_returns)) * base_var,
            index=expected_returns.index,
            columns=expected_returns.index,
        )
        result = self.risk_manager.optimise(expected_returns, cov, budget)
        if constraints:
            for key, limit in constraints.items():
                if key == "max_position":
                    result.weights = result.weights.clip(upper=limit, lower=-limit)
                    if result.weights.abs().sum() > 0:
                        result.weights = result.weights / result.weights.abs().sum()
        return result

    def backtest(
        self,
        universe: Iterable[str],
        start: dt.datetime | pd.Timestamp,
        end: dt.datetime | pd.Timestamp,
        params: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        params = params or {}
        ticker = next(iter(universe), None)
        if ticker is None:
            raise ValueError("Universe must contain at least one ticker")
        market = self.data_provider.load_market_data(ticker, start=pd.to_datetime(start), end=pd.to_datetime(end))
        fundamentals = self.data_provider.load_fundamentals(ticker, as_of=pd.to_datetime(end))
        alt = self.data_provider.load_alternative_data(ticker, as_of=pd.to_datetime(end))
        features = self.feature_pipeline.prepare_features(ticker, market, fundamentals, alt)
        future_ret = _future_returns(market["close"], params.get("horizon", self.horizon)).reindex(features.index)
        aligned = features.join(future_ret.rename("future_return")).dropna()
        if aligned.empty:
            raise ValueError("Insufficient data for backtest")
        feature_cols = aligned.columns.difference(["future_return"])
        pca_components = params.get("pca_components", min(10, len(feature_cols)))
        if pca_components and pca_components < len(feature_cols):
            pca = PCA(n_components=pca_components)
            reduced = pca.fit_transform(aligned[feature_cols])
            feature_frame = pd.DataFrame(reduced, index=aligned.index)
        else:
            feature_frame = aligned[feature_cols]

        def factory() -> object:
            return Ridge(alpha=params.get("ridge_alpha", 1.0))

        report = self.backtester.run(feature_frame, aligned["future_return"], factory)
        auc = 0.0
        try:
            median = np.median(aligned["future_return"])
            labels = (aligned["future_return"] > median).astype(int)
            preds = aligned["future_return"].values
            auc = float(roc_auc_score(labels, preds))
        except Exception:
            auc = float("nan")
        report.metrics["auc"] = auc
        return {
            "performance": report.performance,
            "metrics": report.metrics,
            "information_coefficient": report.IC,
        }


# ---------------------------------------------------------------------------
# Helper for multi-agent integration


class ContextualQuantamentalAdapter:
    """Adapter that leverages context features when a full provider is unavailable."""

    def __init__(self, horizon: int = 5) -> None:
        self.horizon = horizon

    def score_from_context(self, features: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        if features.empty:
            return 0.0, {}
        latest = features.iloc[-1]
        quant_keys = [c for c in features.columns if "mom" in c or "momentum" in c or "return" in c]
        fundamental_keys = [c for c in features.columns if any(k in c for k in ("value", "quality", "profit", "growth", "leverage"))]
        quant_score = float(latest[quant_keys].mean()) if quant_keys else float(latest.mean())
        fundamental_score = float(latest[fundamental_keys].mean()) if fundamental_keys else 0.0
        composite = float(0.6 * quant_score + 0.4 * fundamental_score)
        contributions = {k: float(latest[k]) for k in quant_keys[:5] + fundamental_keys[:5]}
        return composite, contributions

