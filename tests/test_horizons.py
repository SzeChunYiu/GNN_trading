import pandas as pd

from gnn_trading.horizons import compute_horizon_labels, compute_coverage


def test_compute_labels_and_coverage():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame(
        {
            "close": pd.Series(range(30), dtype=float) + 100,
            "high": pd.Series(range(30), dtype=float) + 101,
            "low": pd.Series(range(30), dtype=float) + 99,
        },
        index=dates,
    )
    horizons = [5, 20]
    enriched = compute_horizon_labels(df, horizons)
    assert all(f"ret_h{h}" in enriched for h in horizons)
    coverage = compute_coverage(enriched, horizons)
    assert len(coverage) == 2
    assert coverage[0].horizon == 5
    assert coverage[0].coverage_ratio >= 0
