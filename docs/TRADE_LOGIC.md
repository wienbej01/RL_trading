# Trade Logic and Environment Details

This document describes the trade logic and diagnostics for `IntradayRLEnv` (single-ticker) and `PortfolioRLEnv` (multi-ticker).

## IntradayRLEnv

- Actions: Discrete(3) mapping to {-1, 0, 1} directions (short, flat, long).
- Risk/Position Sizing: contracts sized via ATR and risk config.
- Triple-Barrier Exits: stop-loss, take-profit, and time-based exit (timeout) with EOD flattening.
- Scale-Out: optional partial profit-taking (scale_r multiple of R) when enabled.
- Kill-Switch: daily loss threshold forces flat and closes positions.
- Reward Options:
  - `pnl`, `dsr` (DifferentialSharpe), `sharpe`, or blended.
  - Penalties: drawdown, realized risk, optional hold/activity shaping.
- Transaction Costs:
  - Charged on entry/exit/scale/EOD: commission, slippage/impact (via `estimate_tc`), with spread approximation.
  - `tx_costs_total`: env-level cumulative cost.
- Diagnostics:
  - `get_action_counts()` returns `long_steps`, `short_steps`, `flat_steps`, and `flips` (consecutive non-zero intent flips).
  - Equity curve and trades list are exported by the trainer.

## PortfolioRLEnv

- Actions: MultiDiscrete(3^N) (per ticker {-1,0,1}); unit sizing per ticker.
- Sizing: Magnitude derived from per-ticker ATR and `risk_budget_per_ticker`, capped by `units_per_ticker` and gross exposure cap.
- Exposure/Gating: optional allowed tickers, min/max hold bars, entries per day.
- Penalties: turnover, exposure violation, and optional per-open-position penalty each bar.
- Transaction Costs:
  - For unit changes, charge commission + spread + slippage + impact estimates and accumulate into `tx_costs_total`.
- History & Diagnostics:
  - Step history includes `turnover`, `gross_exposure`, `exposure_pct`.
  - `get_action_counts()` counts net portfolio long/short/flat steps and per-ticker `flips`.

## Metrics and Outputs

- Backtest writes per-window and aggregate summaries including:
  - `sharpe_ratio`, `profit_factor`, `total_return`, `max_drawdown`.
  - `total_trades`, `long_trades`, `short_trades`, and diagnostics mentioned above.

