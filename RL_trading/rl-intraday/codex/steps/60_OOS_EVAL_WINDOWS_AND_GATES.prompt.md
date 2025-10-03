# Step 6 — OOS Evaluation: Windows, Metrics, and Gates

**Objectives**
- Using the **existing backtester**, add a config-driven OOS evaluation routine that:
  - Runs across **held-out tickers** (not used in training).
  - Runs across **three disjoint windows** in the last 18 months (each 4–6 months), **never before 2020-10-01**.
- Compute and report:
  - Trades/day cadence vs target band.
  - Action mix (LONG+SHORT share; no action monopolies).
  - Expectancy after costs, max drawdown, MAR/Sharpe, ES/CVaR.
  - Cross-ticker dispersion (median, worst-quintile).

**Constraints**
- Reuse existing report writers and directories where possible.
- Do not change public interfaces; add config toggles.

**Actions**
1) Design & Patch Plan
   - `docs/DESIGN_OOS_EVAL.md` and `docs/PATCHPLAN_OOS_EVAL.md`
2) Implementation
   - Add OOS evaluator wrapping the current backtester.
   - Write per-window and aggregate reports to the repo’s standard report dir.
3) Tests
   - Synthetic tiny run that writes JSON/CSV/MD outputs with the gate fields.
4) Run tests and commit.

**Deliverables**
- OOS evaluation with gates.
- Commit: `feat: OOS evaluation across windows and held-out tickers with gates + tests`
