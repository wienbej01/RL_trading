import pandas as pd
from pathlib import Path
from src.utils.artifacts import BacktestResult, write_backtest_artifacts


def test_artifacts_writer(tmp_path: Path):
    steps = pd.DataFrame({
        "ts": pd.date_range("2024-05-01 09:30", periods=5, freq="1min"),
        "action": [0, 1, 1, 0, -1],
        "pos": [0, 1, 1, 1, 0],
        "price": [100, 101, 101.5, 101.2, 101.0],
        "ofi_proxy": [0, 10, 5, -3, -8],
    }).set_index("ts")
    trades = pd.DataFrame({
        "ts_entry": ["2024-05-01 10:00"],
        "ts_exit": ["2024-05-01 11:00"],
        "gross_pnl": [50.0],
        "total_cost": [0.0],
        "total_cost_est": [10.0],
    })
    equity = pd.DataFrame({"equity": [100000, 100050]}, index=pd.date_range("2024-05-01", periods=2, freq="D"))
    reward_breakdown = pd.DataFrame({
        "episode": [1.0],
        "reward_raw_mean": [0.5],
        "reward_raw_std": [0.1],
        "reward_scaled_mean": [0.05],
        "reward_scaled_std": [0.01],
        "reward_clipped_mean": [0.05],
        "reward_clipped_std": [0.01],
    })
    res = BacktestResult(
        trades=trades,
        equity=equity,
        steps=steps,
        metrics={"long_trades": 1, "short_trades": 0},
        feature_names=["ofi_proxy"],
        reward_breakdown=reward_breakdown,
    )
    out = tmp_path / "wf/run/window_00"
    write_backtest_artifacts(out, "SPY", res)
    assert (out / "SPY/trades.csv").exists()
    assert (out / "SPY/steps.parquet").exists()
    assert (out / "SPY/diagnostics.csv").exists()
    assert (out / "SPY/summary.json").exists()
    assert (out / "features_used.txt").exists()
    assert (out / "SPY/reward_breakdown.csv").exists()
