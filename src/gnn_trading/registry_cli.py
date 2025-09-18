"""Command-line interface for the disk-backed model registry."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .config import ensure_directory, load_app_config
from .validators import run_validators

DATE_CANDIDATES = ("date", "timestamp", "datetime", "as_of")


def _infer_datetime_column(frame: pd.DataFrame) -> Optional[str]:
    for candidate in DATE_CANDIDATES:
        for col in frame.columns:
            if candidate in col.lower():
                try:
                    pd.to_datetime(frame[col])
                    return col
                except Exception:  # pragma: no cover
                    continue
    if isinstance(frame.index, pd.DatetimeIndex):
        frame.reset_index(inplace=True)
        return frame.columns[0]
    return None


def _read_panel(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover
        print(f"Failed to load {path}: {exc}", file=sys.stderr)
        return None
    return None


def _discover_panels(gold_root: Path) -> List[Tuple[str, Path]]:
    panels: List[Tuple[str, Path]] = []
    for panel_path in gold_root.rglob("panel.parquet"):
        ticker = panel_path.parent.parent.name if panel_path.parent.name == "gold" else panel_path.stem
        panels.append((ticker.upper(), panel_path))
    for panel_path in gold_root.rglob("panel.csv"):
        ticker = panel_path.parent.parent.name if panel_path.parent.name == "gold" else panel_path.stem
        panels.append((ticker.upper(), panel_path))
    return panels


def _assemble_dataset(panels: List[Tuple[str, Path]]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for ticker, path in panels:
        frame = _read_panel(path)
        if frame is None or frame.empty:
            continue
        date_col = _infer_datetime_column(frame)
        if date_col is None:
            continue
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        frame = frame.dropna(subset=[date_col]).sort_values(date_col)
        frame["__ticker"] = ticker
        frame["__date"] = frame[date_col]
        frames.append(frame)
    if not frames:
        raise ValueError("No gold panels with usable datetime columns were found.")
    combined = pd.concat(frames, ignore_index=True)
    return combined


def _select_features(frame: pd.DataFrame, horizons: Iterable[int]) -> Tuple[pd.DataFrame, Dict[int, pd.Series]]:
    label_map: Dict[int, pd.Series] = {}
    numeric_cols = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    feature_cols = []
    for col in numeric_cols:
        if col.startswith("ret_h") or col.startswith("vol_h") or col.startswith("breakout_h"):
            continue
        if col in {"__date"}:
            continue
        feature_cols.append(col)
    feature_frame = frame[feature_cols].fillna(method="ffill").fillna(method="bfill").dropna()
    aligned = frame.loc[feature_frame.index]
    for horizon in horizons:
        label_col = f"breakout_h{horizon}"
        if label_col in aligned.columns:
            label_map[horizon] = aligned[label_col].astype(float)
        else:
            ret_col = f"ret_h{horizon}"
            if ret_col not in aligned.columns:
                raise ValueError(f"Missing label columns for horizon {horizon}")
            label_map[horizon] = (aligned[ret_col] > 0).astype(float)
    feature_frame["__date"] = aligned["__date"].values
    feature_frame["__ticker"] = aligned["__ticker"].values
    return feature_frame, label_map


def _purged_splits(dates: pd.Series, n_splits: int, embargo: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    unique_dates = np.sort(dates.unique())
    if len(unique_dates) < n_splits + 1:
        n_splits = max(1, len(unique_dates) - 1)
    splitter = TimeSeriesSplit(n_splits=n_splits)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for train_idx, test_idx in splitter.split(unique_dates):
        train_dates = unique_dates[train_idx]
        test_dates = unique_dates[test_idx]
        if len(test_dates) == 0:
            continue
        test_start = pd.to_datetime(test_dates[0])
        embargo_start = test_start - pd.Timedelta(days=embargo)
        train_mask = (dates.isin(train_dates)) & (dates < embargo_start)
        test_mask = dates.isin(test_dates)
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        splits.append((train_mask.to_numpy().nonzero()[0], test_mask.to_numpy().nonzero()[0]))
    if not splits:
        raise ValueError("Unable to generate walk-forward splits; consider reducing splits or embargo.")
    return splits


def _mc_dropout(probabilities: np.ndarray, samples: int = 32) -> Tuple[float, float]:
    draws = []
    rng = np.random.default_rng(42)
    for _ in range(samples):
        noise = rng.normal(loc=0.0, scale=0.05, size=probabilities.shape)
        perturbed = np.clip(probabilities + noise, 0.0, 1.0)
        draws.append(perturbed)
    stacked = np.stack(draws)
    return float(stacked.mean()), float(stacked.std())


def _train_horizon(
    features: pd.DataFrame,
    labels: pd.Series,
    horizon: int,
    embargo: int,
    splits: int,
) -> Tuple[Dict[str, object], Dict[str, float]]:
    scaler = StandardScaler()
    X = scaler.fit_transform(features.drop(columns=["__date", "__ticker"]))
    y = labels.loc[features.index].values
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    metrics: Dict[str, float] = {}
    aucs: List[float] = []
    accs: List[float] = []
    briers: List[float] = []
    mc_std: List[float] = []
    splits_idx = _purged_splits(features["__date"], splits, embargo)
    for train_idx, test_idx in splits_idx:
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        fold_model = LogisticRegression(max_iter=200)
        fold_model.fit(X_train, y_train)
        probs = fold_model.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y_test, probs))
        accs.append(accuracy_score(y_test, probs > 0.5))
        briers.append(brier_score_loss(y_test, probs))
        _, std = _mc_dropout(probs)
        mc_std.append(std)
    metrics["auc_mean"] = float(np.mean(aucs))
    metrics["accuracy_mean"] = float(np.mean(accs))
    metrics["brier_mean"] = float(np.mean(briers))
    metrics["mc_dropout_std"] = float(np.mean(mc_std))
    model_data = {
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "feature_names": [c for c in features.columns if c not in {"__date", "__ticker"}],
        "horizon": horizon,
        "baseline_return": float(labels.loc[features.index].mean()),
        "baseline_vol": float(labels.loc[features.index].std()),
    }
    return model_data, metrics


def _transform_with_model(frame: pd.DataFrame, model_data: Dict[str, object]) -> np.ndarray:
    feature_names = model_data["feature_names"]
    values = frame[feature_names].values
    mean = np.array(model_data["scaler_mean"])
    scale = np.array(model_data["scaler_scale"])
    scale = np.where(scale == 0, 1.0, scale)
    return (values - mean) / scale


def _save_run(
    run_dir: Path,
    registry_path: Path,
    run_config: Dict[str, object],
    model_payload: Dict[int, Dict[str, object]],
    feature_names: List[str],
    dataset_manifest: Dict[str, object],
    metrics: Dict[int, Dict[str, float]],
) -> None:
    ensure_directory(run_dir)
    torch.save({"horizons": model_payload}, run_dir / "model.pt")
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(run_config, handle, sort_keys=True)
    (run_dir / "feature_spec.json").write_text(json.dumps({"features": feature_names}, indent=2), encoding="utf-8")
    (run_dir / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if not (run_dir / "notes.md").exists():
        (run_dir / "notes.md").write_text("# Run Notes\n\n- Auto-generated run.\n", encoding="utf-8")
    latest = registry_path / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(run_dir.name)


def cmd_train(args: argparse.Namespace) -> None:
    cfg = load_app_config(args.config)
    registry_root = ensure_directory(Path(args.out) if args.out else Path(cfg["registry"]["path"]))
    timeframe_ok, leakage_ok = run_validators(args.config, args.gold_root)
    if not (timeframe_ok and leakage_ok):
        raise SystemExit("Validation failed. See reports/validation_*.md for details.")
    horizons = [int(h) for h in (args.horizons.split(",") if args.horizons else cfg["labels"]["horizons"])]
    panels = _discover_panels(Path(args.gold_root))
    dataset = _assemble_dataset(panels)
    features, label_map = _select_features(dataset, horizons)
    model_payload: Dict[int, Dict[str, object]] = {}
    metrics: Dict[int, Dict[str, float]] = {}
    for horizon in horizons:
        model_data, metric = _train_horizon(features, label_map[horizon], horizon, args.embargo, args.splits)
        model_payload[horizon] = model_data
        metrics[horizon] = metric
    run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    run_dir = registry_root / run_id
    dataset_manifest = {
        "rows": int(len(features)),
        "tickers": sorted(features["__ticker"].unique()),
        "horizons": horizons,
    }
    run_config = {
        "timestamp": run_id,
        "config_path": str(Path(args.config).resolve()),
        "gold_root": str(Path(args.gold_root).resolve()),
        "horizons": horizons,
        "embargo": args.embargo,
        "splits": args.splits,
    }
    feature_names = [c for c in features.columns if c not in {"__date", "__ticker"}]
    _save_run(run_dir, registry_root, run_config, model_payload, feature_names, dataset_manifest, metrics)
    print(f"Saved registry run to {run_dir}")


def _load_run(model_ref: str, registry_path: Path) -> Path:
    ref_path = Path(model_ref)
    if ref_path.name == "latest":
        return registry_path / "latest"
    if ref_path.exists():
        return ref_path
    candidate = registry_path / model_ref
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Run directory {model_ref} not found")


def _load_model(run_dir: Path) -> Dict[str, object]:
    payload = torch.load(run_dir / "model.pt", map_location="cpu")
    feature_spec = run_dir / "feature_spec.json"
    if not feature_spec.exists():
        horizon_sample = next(iter(payload["horizons"].values()))
        feature_spec.write_text(json.dumps({"features": horizon_sample["feature_names"]}, indent=2), encoding="utf-8")
    return payload


def cmd_evaluate(args: argparse.Namespace) -> None:
    cfg = load_app_config(args.config)
    registry_root = Path(cfg["registry"]["path"])
    run_dir = _load_run(args.model, registry_root).resolve()
    panels = _discover_panels(Path(args.gold_root))
    dataset = _assemble_dataset(panels)
    payload = _load_model(run_dir)
    horizons = [int(h) for h in payload["horizons"].keys()]
    features, label_map = _select_features(dataset, horizons)
    metrics: Dict[int, Dict[str, float]] = {}
    for horizon in horizons:
        model_data = payload["horizons"][horizon]
        X = _transform_with_model(features, model_data)
        y = label_map[horizon].loc[features.index].values
        weights = np.array(model_data["coef"])
        intercept = model_data["intercept"]
        logits = X @ weights + intercept
        probs = 1.0 / (1.0 + np.exp(-logits))
        metrics[horizon] = {
            "auc": float(roc_auc_score(y, probs)),
            "accuracy": float(accuracy_score(y, probs > 0.5)),
            "brier": float(brier_score_loss(y, probs)),
        }
    metrics_path = run_dir / "metrics.json"
    existing = {}
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
    existing["evaluation"] = metrics
    metrics_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def cmd_predict(args: argparse.Namespace) -> None:
    cfg = load_app_config(args.config)
    registry_root = Path(cfg["registry"]["path"])
    run_dir = _load_run(args.model, registry_root).resolve()
    payload = _load_model(run_dir)
    panel = _read_panel(Path(args.panel))
    if panel is None or panel.empty:
        raise SystemExit("Panel file is empty or unreadable")
    date_col = _infer_datetime_column(panel)
    if date_col is None:
        raise SystemExit("Could not infer a datetime column in the panel")
    panel[date_col] = pd.to_datetime(panel[date_col], errors="coerce")
    panel = panel.dropna(subset=[date_col])
    outputs: List[pd.DataFrame] = []
    generated_at = datetime.utcnow().isoformat()
    for horizon, model_data in payload["horizons"].items():
        features = model_data["feature_names"]
        missing = [col for col in features if col not in panel.columns]
        enriched = panel.copy()
        for col in missing:
            enriched[col] = 0.0
        X = _transform_with_model(enriched, model_data)
        logits = X @ np.array(model_data["coef"]) + model_data["intercept"]
        probs = 1.0 / (1.0 + np.exp(-logits))
        mean_prob, prob_std = _mc_dropout(probs)
        expected_return = probs * model_data.get("baseline_return", 0.0)
        expected_vol = np.sqrt(np.maximum(probs * (1 - probs), 1e-6))
        confidence = 1.0 - (probs * (1 - probs))
        frame = pd.DataFrame({
            "date": enriched[date_col],
            "horizon": horizon,
            "prob_breakout": probs,
            "prob_mc_mean": mean_prob,
            "prob_mc_std": prob_std,
            "expected_return": expected_return,
            "expected_vol": expected_vol,
            "confidence": confidence,
            "composite_score": probs - expected_vol,
            "model_run": run_dir.name,
            "generated_at": generated_at,
            "provenance": str(Path(args.panel).resolve()),
        })
        outputs.append(frame)
    result = pd.concat(outputs).sort_values(["date", "horizon"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.out, index=False)
    print(f"Wrote signals to {args.out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model registry commands")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train and register a new model run")
    train.add_argument("--gold-root", required=True, help="Root directory containing gold panels")
    train.add_argument("--horizons", default=None, help="Comma-separated list of horizons")
    train.add_argument("--out", default=None, help="Override registry path")
    train.add_argument("--config", default="config/app.yaml", help="Config file path")
    train.add_argument("--embargo", type=int, default=5, help="Embargo days between train/test splits")
    train.add_argument("--splits", type=int, default=3, help="Number of walk-forward splits")
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate", help="Evaluate an existing run on gold panels")
    evaluate.add_argument("--model", required=True, help="Run directory name or 'latest'")
    evaluate.add_argument("--gold-root", required=True, help="Root directory containing gold panels")
    evaluate.add_argument("--config", default="config/app.yaml", help="Config file path")
    evaluate.set_defaults(func=cmd_evaluate)

    predict = sub.add_parser("predict", help="Generate signals from a registered model")
    predict.add_argument("--model", required=True, help="Run directory name or 'latest'")
    predict.add_argument("--panel", required=True, help="Path to a gold panel file")
    predict.add_argument("--out", required=True, help="Destination parquet for signals")
    predict.add_argument("--config", default="config/app.yaml", help="Config file path")
    predict.set_defaults(func=cmd_predict)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
