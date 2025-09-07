# RL Intraday Trading System — Improvement Plan

Owner: Quant Strategy & Systems
Updated: 2025-09-04

## Status Summary

- Phase 1 (Data Integrity) — Completed
  - Coverage QA (concise coverage reports; reduced log noise)
  - Cache rebuild logic hardened; strict RTH and early-close awareness
  - Context prep scripts with fallbacks (SPY/QQQ/VIX)

- Phase 2 (Features) — Completed (expanded)
  - ICT: PDM/OR distances, displacement bars (+density), FVG (+density), equal highs/lows proximity
  - VPA: RVOL, climax volume, churn (+z), imbalance persistence, direction EMA, intrabar volatility
  - Levels: prior/current day open/close, pivot points (S1/R1/S2/R2), rolling support/resistance, session VWAP distance
  - True structure: swing highs/lows (causal), distances to last swings, break-of-structure flags
  - SMT: instrument vs SPY, and SPY vs QQQ divergence
  - VIX: base features and term-structure ratios; Yahoo path integrated
  - Pruning: variance + correlation

## Phase 3 — Reward, Policy, and Risk Enhancements (In Progress)

- Reward
  - Blend of DSR and raw PnL with microstructure penalties
  - Time-of-day widening penalties (open/close)
  - Next: optional asymmetric penalty under accelerating drawdown

- Policy/Hyperparameters
  - n_envs=8, n_steps=1024, batch=1024, gamma=0.995, gae=0.97, target_kl=0.03
  - LR schedule (cosine/linear) and entropy annealing supported

- Execution realism
  - No-trade windows (first/last 5m), widened penalties (first/last 15m)
  - ATR time-stop added; cap trades/hour and partial scale-outs configurable

## Phase 4 — Training Process + Evaluation (In Progress)

- Walk-forward automation (monthly/biweekly folds) with embargo; aggregate OOS metrics
- Walk-forward CLI added (scripts/walkforward_train_eval.py)
- Early stopping (validation Sharpe/Calmar); checkpoint best model per fold (planned)
- Diagnostics: action histogram, trade counts, holding times; per-regime PnL buckets (VIX bins, SMT buckets) in backtests; steps/sec
- Baselines: buy-and-hold and simple rules for context

## Phase 5 — Cleanup + UX (Planned)

- Normalize logging, persistent daily coverage reports
- CLI toggles for device, normalization, and hyperparam overrides

## Acceptance Targets (Initial OOS)

- Reduce drawdown ≥ 50% vs current (-22.5% → better than -10%)
- Move OOS return toward positive and ensure stability across folds

## Action Items (Next 1–2 iterations)

- [x] LR/entropy schedules; consider asymmetric drawdown penalty
- [x] Walk-forward CLI; ensure VecNormalize parity checks
- [x] Per-regime (VIX/SMT) evaluation breakdown in backtests
- [x] Time-stop; cap trades/hour and scale-outs configurable
- [ ] Swing-based stop option for risk sizing (min of ATR and swing stop)
- [ ] Optional: microstructure PCA (grouped) and equal highs/lows distance refinement

## Phase 6 — Backtest Debugging (In Progress)

- Status: Backtest functional; trades now occur under deterministic and model-driven loops. Remaining: align model obs/normalization robustly for all saved folds and tune for profitability.

- Work Done:
  - Fixed execution/risk wiring: replaced missing `ExecutionSimulator` with `ExecutionEngine` and corrected settings access (no more `TypeError: unhashable type: 'dict'`).
  - Backtester parity fixes: load saved `VecNormalize` stats when shapes match; fallback gracefully when not; handle both 4- and 5‑tuple step APIs (SB3 vs Gymnasium).
  - Feature parity: backtester now discovers `model_features.json` or `model.zip_features.json` and aligns columns (add missing=0, drop extras).
  - Added DEBUG breadcrumbs inside env when opens are skipped (no‑trade window vs risk sizing).
  - Sanity script `scripts/debug_trade_check.py` confirms trades on SPY and BBVA slices.

- Pending Tasks:
  - [ ] Add a configurable baseline (e.g., alternating/trend SMA) to `BacktestEvaluator` for smoke tests without SB3.
  - [ ] Harden VecNormalize restoration across folds with shape guard + auto-disable on mismatch.
  - [ ] Hyperparam sweep for BBVA 2020‑09→2024‑12; select top folds and ensemble for 2025‑H1.
