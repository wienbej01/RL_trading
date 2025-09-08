# RL Intraday Trading System — Improvement Plan

Owner: Quant Strategy & Systems
Updated: 2025-09-07

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

- Status: Backtest runs across the full evaluation window (per‑day episodes). Feature alignment and manual VecNormalize in place. Initial “no trades” fixed. Current behavior: action bias (all‑long) under aggressive shaping; goal is ~5–10 balanced positions/day, then dial down to 1–3/day.

- Work Done:
  - Parity & stability:
    - Feature alignment to training list; robust NaN filling; Gymnasium‑safe DummyVecEnv.
    - Manual VecNormalize: loads `vecnormalize.pkl` and normalizes obs on reset/step.
    - Execution/risk wiring fixed; report sums commission correctly.
  - Reward shaping upgrades:
    - Added `directional`, `hybrid`, and advanced `hybrid2` (vol‑norm PnL + DSR + drift‑neutral directional + soft activity constraint + churn + DD‑accel penalty + regime weighting).
    - Directional shaping made drift‑neutral: action_dir × (bar_return − EMA_session(bar_return)).
    - Activity shaping (open_bonus, target trades/day, soft Lagrangian penalty), hold penalty; applied consistently.
  - Training & eval flow:
    - Training episodes start at RTH per day (sequential/random).
    - Warm‑up forcing of holds to side only in early portion of session (configurable).
    - Evaluator runs day‑by‑day; prints per‑day action histograms and aggregates equity/trades across all days.

- Pending Tasks (Next Working Session):
  - [ ] Rebalance shaping to eliminate “all‑long” preference: lower `dir_weight` (→ ~400–600), reduce `open_bonus`, raise `hold_penalty`, set target_per_day=6–8, keep reward_scaling=1.0 for 20k, reduce later.
  - [ ] 20k sanity runs until per‑day action_counts show both longs/shorts and ~5–10 opens/day across the full period.
  - [ ] 300k retrain with dialed‑back shaping and lower caps once sanity passes.
  - [ ] Add decomposed reward logging (pnl_norm, dsr, directional, activity, churn, dd_accel) and daily λ (activity) traces to aid tuning.

## Phase 7 — Profitability Path (Planned)

- Targets:
  - Trade frequency: 5–10/day; avg duration 5–30m.
  - OOS Sharpe > 0.3; Max DD < 10%.

- Levers:
  - Reward: `alpha/beta`, `dir_weight`, activity shaping, microstructure penalty by regime.
  - Risk: stop/take multiples, 1% risk per trade, time‑stop 30–60m, trade/hour caps.
  - Training: entropy anneal; lr schedule; embargo; cross‑month folds.

## System File Overview (Relevant Components)

- Configuration
  - `configs/settings.yaml` — all knobs (reward, trading caps, forcing, training params).

- Environment & Simulation
  - `src/sim/env_intraday_rl.py` — RL environment: reward computation (PnL normalization, DSR, directional de‑meaned, activity Lagrangian, churn, DD‑accel, regime weighting), daily episode starts, warm‑up forcing, risk/exec integration.
  - `src/sim/execution.py` — execution costs (commission, slippage, impact); `estimate_tc` helpers.
  - `src/sim/risk.py` — risk config (risk per trade, stops/TPs, drawdown controls).
  - `src/sim/dummy_vec_env.py` — Gymnasium‑safe vec env wrapper.

- Training & Evaluation
  - `src/rl/train.py` — training entry; VecNormalize during train; vectorized env setup; callbacks.
  - `src/evaluation/backtest_evaluator.py` — day‑by‑day backtest loop; manual VecNormalize; aggregates equity/trades; metrics and plotting.

- Features & Data
  - `src/features/pipeline.py` — feature generation and alignment.
  - `src/data/data_loader.py`, `src/data/polygon_ingestor.py` — data ingestion/loaders.

- Scripts (Ops)
  - `scripts/generate_performance_report.py` — evaluation CLI (normalized, full period).
  - `scripts/collect_polygon_us_stocks.py`, `scripts/aggregate_us_stock_data.py`, `scripts/fetch_polygon_range.py` — data collection/aggregation.
  - `scripts/download_vix_data.py`, `scripts/ingest_external_vix.py` — VIX sourcing.
  - `scripts/sweep_activity_shaping.py` — quick sweep for anti‑hold policies.

- Documentation
  - `docs/user_guide.md` — normalized evaluation; reward modes.
  - `docs/TECH_OVERVIEW.md` — system overview; data completeness; Gymnasium notes.
  - `docs/rl_improvement_plan.md` — this plan.

## Best Practice Reward Mechanism
Reward design patterns (what works in practice)
  
  - Volatility‑normalized PnL:
      - Idea: Reward = pnl / (k × ATR or rolling σ). Caps outliers and stabilizes learning.
      - Pros: Scale invariant; handles heteroskedasticity; improves gradient signal.
      - Cons: Can over‑incentivize noise if too short a window.
  - Risk‑adjusted returns (Sharpe/Sortino/Calmar proxy):
      - Idea: Use per-bar or per-day updates of Sharpe-like metrics (your DSR is a good variant).
      - Pros: Aligns with risk‑adjusted goals.
      - Cons: Noisy on short horizons; demands careful smoothing.
  - Directional shaping (drift‑neutral):
      - Idea: action_dir × (bar_return − mean_return_session). You’re doing this now; this is key for balanced long/short.
      - Pros: Removes drift bias; symmetric long/short incentives.
      - Cons: Needs a good estimate of “mean” (EMA or rolling median); too big can dominate PnL.
  - Activity shaping (soft constraint on trades/day):
      - Idea: Reward bonus if trades/day <= target; penalize excess. Use a soft Lagrangian to adapt the penalty automatically.
      - Pros: Direct control over frequency; reduces hold‑only and over-trading extremes.
      - Cons: If static, can oscillate. Lagrangian helps (see below).
  - Inventory/cost penalties:
      - Idea: Penalize inventory at EOD (you have), action-change penalty (Δaction) to control churn, and explicit cost terms (slippage/commission).
      - Pros: Better realism; controls whipsaw.
      - Cons: Can suppress exploration when too large (tune gradually).
  - Drawdown acceleration penalty:
      - Idea: Penalize increases in drawdown slope (not just level). This stops “throwing good money after bad”.
      - Pros: Protects against “averaging down” patterns.
      - Cons: Needs smoothing and guardrails to avoid overreaction.
  - Opportunity/capture shaping:
      - Idea: Small positive reward for capturing a fraction of the next k-bar range when a position exists; small negative for missing large moves.
      - Pros: Pushes entries near information-rich moves; can combat hold-only plateaus.
      - Cons: If too large, can create over-eagerness; keep small.
  - Regime‑aware weights:
      - Idea: Scale terms by VIX/signal strength. E.g., reduce directional shaping in low‑vol regimes; increase PnL term in high‑vol regimes.
      - Pros: Prevents overfitting to one regime; improves stability.
      - Cons: Requires clean regime tagging (you have VIX already).
  - Potential‑based reward shaping:
      - Idea: r’ = r + γΦ(s’) − Φ(s). Choose Φ(s) = c × expected_move(s) (e.g., based on EMA of returns or distance to levels).
      - Pros: Theoretically preserves optimal policy; helps credit assignment.
      - Cons: Φ must be shaped carefully; don’t let this dominate PnL.
  - Constraint‑aware (Lagrangian) shaping:
      - Idea: Use λ (dual variable) to penalize deviation from target trades/day: r = base_reward − λ × max(0, trades_today − target). Update λ daily.
      - Pros: Soft, adaptive control to hit targets without retuning hyperparams constantly.
      - Cons: Slightly more bookkeeping; but very effective.
  
  Recommended upgrade (tailored to your system)
  
  - Adopt a multi-term, drift‑neutral, regime‑aware, soft‑constrained reward:
      - PnL term (normalized):
      - pnl_norm = pnl / (ATR_10 × point_value) (cap to [-pnl_cap, pnl_cap], start with pnl_cap=1)
  - Risk‑adjusted term (DSR):
      - Keep DSR but reduce its weight when directional term exists (e.g., alpha=0.15)
  - Directional drift‑neutral term:
      - dir = action_dir × (bar_return − EMA_30min(bar_return)); weight dir_weight ~ 200–500 (after warm‑up)
  - Activity soft constraint (Lagrangian):
      - r_activity = +b if trades_today ≤ target, else −λ × (trades_today − target)
      - Update λ_adaptive: λ ← λ + η × (trades_today − target), with λ ≥ 0 (η small; e.g., 0.01)
  - Inventory/churn penalties:
      - Keep EOD inventory penalty; add small penalty on Δaction (encourage holds when no signal).
  - Drawdown acceleration penalty:
      - r_dd = −k × max(0, dd_slope) where dd_slope is positive changes of drawdown (EMA over a short window).
  - Regime weighting:
      - Multiply directional weight by f(VIX): e.g., 0.7 in low vol (<15), 1.0 in mid vol (15–25), 1.2 in high vol (>25).
  - Reward scaling/clipping:
      - Keep reward_scaling=1 during warm‑ups; clip final reward to [-2, 2]; later reduce scaling to 0.1–0.2 and clip to [-1,1].
  
  How to implement (concrete changes)
  
  - Add to settings.yaml:
      - env.reward:
      - kind: “hybrid2” (new variant)
      - alpha, beta, dir_weight
      - pnl_norm_window: 10 (ATR window)
      - pnl_cap: 1.0
      - dir_ema_minutes: 30
      - activity: { target_per_day: 8, bonus: 0.02, lagrange_eta: 0.01, lambda_init: 0.0 }
      - dd_accel_penalty: 0.05
      - regime: { vix_low: 15, vix_high: 25, dir_weight_low: 0.7, dir_weight_high: 1.2 }
  - env.trading:
      - episode_day_mode: sequential
      - force_open_epsilon: 0.05 (warm‑up only; set to 0 in eval)
      - force_warmup_frac: 0.2
      - no_trade_open_minutes: 5; no_trade_close_minutes: 5
      - max_trades_per_hour: 3; max_holding_minutes: 30
  - Env code (you already have most pieces):
      - Keep de‑meaned directional term you added.
      - Compute ATR for pnl normalization (you already use ATR; re‑use).
      - Add Lagrange λ per day; update at EOD: λ ← max(0, λ + η × (trades_today − target)).
      - Add small penalty on action changes when pos unchanged (discourage flip‑flop).
      - Add dd acceleration penalty: track drawdown time‑series; penalize increases in slope (EMA of dd derivative).
      - Apply regime multiplier to dir_weight using VIX series if available.
  
  Suggested hyperparameter ranges
  
  - Warm‑up (20k steps sanity):
      - dir_weight=1000; open_bonus=0.08; hold_penalty=0.06; ent_coef=0.05→0.02 (anneal)
      - activity.target=8; bonus=0.02; lagrange_eta=0.01; λ_init=0.0
      - max_trades_per_hour=6 (warm‑up only); reward_scaling=1.0
  - Stable (300k+):
      - dir_weight=300–600; open_bonus=0.03–0.05; hold_penalty=0.01–0.03; ent_coef=0.02→0.01
      - max_trades_per_hour=3; reward_scaling=0.2; clip reward to [-1,1]
  
  Monitoring and validation
  
  - Log per-day action_counts and opens/day.
  - Log daily λ (Lagrange) and resulting trades/day to confirm convergence to the target.
  - Log decomposed reward terms (pnl_norm, dsr, directional, activity, dd_accel) for first N days.
  - Keep daily trade count in [5,10] initially, then gradually reduce target and weights to [1,3].
