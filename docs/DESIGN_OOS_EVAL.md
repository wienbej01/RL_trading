# Enhancement — OOS Evaluation (Design)

Objectives
- Config-driven OOS evaluation across three disjoint windows in the last ~18 months.
- Evaluate held-out tickers and report cadence, action mix, expectancy after costs, drawdowns, and Sharpe/MAR-style metrics.

Approach
- Implement `scripts/oos_eval.py` wrapper that:
  - Loads config, data, features, and a saved SB3 model.
  - Derives three windows from the dataset’s max index (or accepts `--windows`).
  - Runs `MultiTickerRLTrainer.backtest` per window with `allowed_tickers` set from `--test-tickers`.
  - Collects per-window metrics and computes gates.
  - Aggregates summary statistics (median, worst quintile) across windows.

Metrics & Gates
- trades_per_day: total trades divided by unique trading days.
- action_mix: long/short shares of trades (HOLD proxied by 1 − long − short).
- expectancy: average PnL per trade.
- drawdowns: max_drawdown (absolute fraction).
- Gates (illustrative defaults): cadence ≥ 1/day, both long and short trades present, expectancy > 0, max DD < 20%.

Outputs
- Writes `oos_results.json` under the requested output directory with per-window results and aggregate stats.

Safety
- Skips empty windows gracefully. No changes to training or env interfaces.
