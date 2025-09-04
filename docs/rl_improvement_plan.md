# RL Intraday Trading System — Improvement Plan

Owner: Quant Strategy & Systems
Updated: 2025-09-04

## Status Summary

- Phase 1 (Data Integrity) — Completed
  - Added coverage QA (concise coverage reports; reduced log noise)
  - Cache rebuild logic hardened; strict RTH and early-close awareness
  - Context prep scripts with fallbacks (SPY/QQQ/VIX)

- Phase 2 (Features) — Completed (first cut)
  - ICT: PDM/OR distances, displacement bars (+density), FVG (+density), equal highs/lows proximity
  - VPA: RVOL, climax volume, churn (+z), imbalance persistence, direction EMA, intrabar volatility
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
  - Next: LR/entropy schedules (cosine/linear decay)

- Execution realism
  - No-trade windows (first/last 5m), widened spreads (first/last 15m)
  - Next: ATR time-stop; cap trades/hour; partial scale-outs

## Phase 4 — Training Process + Evaluation (Planned)

- Walk-forward automation (monthly/biweekly folds) with embargo; aggregate OOS metrics
- Early stopping (validation Sharpe/Calmar); checkpoint best model per fold
- Diagnostics: action histogram, trade counts, holding times; per-regime PnL (VIX bins, SMT buckets); steps/sec
- Baselines: buy-and-hold and simple rules for context

## Phase 5 — Cleanup + UX (Planned)

- Normalize logging, persistent daily coverage reports
- CLI toggles for device, normalization, and hyperparam overrides

## Acceptance Targets (Initial OOS)

- Reduce drawdown ≥ 50% vs current (-22.5% → better than -10%)
- Move OOS return toward positive and ensure stability across folds

## Action Items (Next 1–2 iterations)

- [ ] Add LR/entropy schedules and optional asymmetric drawdown penalty
- [ ] Implement walk-forward CLI and VecNormalize parity checks
- [ ] Add per-regime (VIX/SMT) evaluation breakdown and summary in backtests
- [ ] Add cap trades/hour and ATR time-stop options to env
- [ ] Optional: microstructure PCA (grouped) and equal highs/lows distance refinement
