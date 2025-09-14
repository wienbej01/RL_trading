# Step 2 — Audit Repository & Identify Training vs Testing Ticker Sources

**Objectives**
- Build an authoritative map of the current system: training entrypoints, RL loop, model, feature pipeline, backtester, configs, and tests.
- Identify which steps define **training tickers** and which define **testing/backtest tickers**.
- Produce documentation files; do not change code in this step.

**Actions**
1) Search the repo (excluding .git, venv/.venv, build artifacts) to locate:
   - Training scripts/modules/CLI and their call graphs.
   - RL loop & loss assembly sites (entropy, KL, clip, γ, λ usage).
   - Model components (encoder, policy/value heads); where auxiliary heads could be added.
   - Feature pipeline modules, normalization, and leak guards.
   - Backtester/evaluator entrypoints and report writers.
   - Configuration files and schema.
   - Unit/integration tests and utilities.
   - **Ticker sources** for training vs testing/backtest (hardcoded lists, configs, CLI flags, env, DB).
2) Create `docs/AUDIT_REPO_MAP.md` with tables and <=20-line code excerpts (with line numbers) to prove each classification.
3) Create `docs/TICKER_SOURCE_SUMMARY.md` that explicitly answers:
   - “Which step defines the TRAINING tickers?” Include exact script/function/config path and copy-paste command(s).
   - “Which step defines the TESTING/BACKTEST tickers?” Same specificity.
   - Include an ASCII dependency flow diagram from source → loader → trainer/backtester.

**Deliverables**
- `docs/AUDIT_REPO_MAP.md`
- `docs/TICKER_SOURCE_SUMMARY.md`
- Commit: `docs: repo audit and ticker source summary`
