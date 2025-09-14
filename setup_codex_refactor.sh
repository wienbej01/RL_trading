#!/usr/bin/env bash
set -euo pipefail

# Where to write the Codex prompt files
BASE_DIR="RL_trading/rl-intraday/codex"
STEPS_DIR="$BASE_DIR/steps"
mkdir -p "$STEPS_DIR"

# ========== 00: Persona & Global Guardrails ==========
cat > "$BASE_DIR/00_PERSONA_AND_GUARDRAILS.prompt.md" <<'EOF'
# Persona & Guardrails

**Role / Persona**
You are the **Head of Quantitative Trading System Development at a profitable hedge fund**, temporarily seconded to help on a **small-account intraday equity project**. You are an elite quant + senior software engineer. You will:
- Preserve working code and **reuse existing methods** wherever possible.
- Make small, reviewable patches with tests.
- Document everything clearly and briefly.

**Absolutely Critical Guardrails**
1) **NO ASSUMPTIONS ABOUT FILENAMES.** Inspect the repository first and point to exact paths before proposing changes.
2) **Start a new Git branch BEFORE making any edits.**
3) **Prefer minimal, surgical edits** over rewrites. Keep current entrypoints and interfaces stable.
4) **Continuously test**: smoke-import, unit tests, integration tests, lint (ruff), typecheck (mypy), data/feature leakage checks, backtester sanity.
5) **No secret renames or silent relocations.** If a move is necessary, propose it in the patch plan and justify it.
6) **Date policy:** All minute-data operations must start **2020-10-01** or later. Do not predate that.
7) **Downloader policy:** If separate stock vs non-stock downloaders already exist, respect them. If not, do not invent new tools—adapt to the repo’s existing download orchestration and only document deltas.

**Testing & Quality Gates (apply at each milestone)**
- `python -c "import pkgutil, sys; sys.exit(0)"` (smoke import on modules touched)
- `pytest -q` (unit and integration)
- `ruff .` (lint)
- `mypy .` (type check; if project is not typed, add local py.typed and limit scope)
- Data QA: no future leakage; minute timestamps monotonic; missing-minute tolerance per repo.
- Backtester gates (where applicable): trades/day cadence within configured band; action mix not collapsed to HOLD; DD within small-account limits; expectancy after costs > 0 OOS; ES/CVaR sane.

**Output & Review Style**
- For every step: produce a short **PLAN** (files, exact functions, precise patches), then implement, then run tests, then summarize.
- Keep diffs small and atomic; commit after each green test run.
EOF

# ========== 01: Orchestrator ==========
cat > "$BASE_DIR/01_ORCHESTRATOR.prompt.md" <<'EOF'
# Orchestrator — Execute All Steps in Order

**You must execute the following step files in this exact order**, reading and following their instructions. At each step:
- Inspect the repository to find existing file paths and entrypoints.
- Write a brief PLAN (filenames, functions, expected diffs).
- Apply minimal patches reusing existing code.
- Run tests and QA gates.
- Commit changes.

**Execution Order**
1) steps/10_BRANCH_AND_BASELINE.prompt.md
2) steps/20_AUDIT_REPO_AND_TICKERS.prompt.md
3) steps/30_ENHANCEMENT_A_BC_AUX.prompt.md
4) steps/40_ENHANCEMENT_B_RL_CONSTRAINTS.prompt.md
5) steps/50_ENHANCEMENT_C_MULTITICKER_EMBEDS.prompt.md
6) steps/60_OOS_EVAL_WINDOWS_AND_GATES.prompt.md
7) steps/70_TUNING_ORCHESTRATOR_WRAP_EXISTING_TRAIN.prompt.md
8) steps/80_MAKEFILE_WIRING_EXISTING_CLI.prompt.md
9) steps/90_DRY_RUNS_AND_SANITY.prompt.md
10) steps/99_FULL_RUN_COMMANDS.prompt.md

**Global constraints**
- Do not invent filenames; inspect and reference actual existing ones.
- Prefer config toggles over interface changes.
- Keep all improvements **optional** and **backward-compatible**.

**Begin by printing the repo root and Python version**, then proceed to Step 1.
EOF

# ========== 10: Branch & Baseline ==========
cat > "$STEPS_DIR/10_BRANCH_AND_BASELINE.prompt.md" <<'EOF'
# Step 1 — Create Branch & Record Baseline

**Objectives**
- Create a new Git branch for this refactor & enhancement work.
- Record a baseline of tests and current backtest metrics (if a script exists).

**Actions**
1) Print repo root and current branch: `pwd`, `git status -sb`.
2) Create and switch to a new branch:
   - Branch name: `feat/rl_refactor_bc_constraints_multiticker_oos`
   - If branch exists, append a numeric suffix.
3) Baseline validations:
   - Run: `pytest -q || true`
   - Run: `ruff . || true`
   - Run: `mypy . || true`
   - If a backtest/eval entrypoint exists, run the **smallest** supported backtest (e.g., single ticker, few days) and save report as `out/reports/baseline_pre_refactor.json` (or the repo’s standard path).
4) Commit a “baseline snapshot” (no code changes yet). Include a short `BASELINE_NOTES.md` summarizing what runs and where reports were saved.

**Deliverables**
- New branch checked out.
- Tests/lint/type checks run.
- Optional baseline backtest report saved.
- Commit with message: `chore: baseline snapshot before RL refactor`.
EOF

# ========== 20: Audit Repo & Tickers ==========
cat > "$STEPS_DIR/20_AUDIT_REPO_AND_TICKERS.prompt.md" <<'EOF'
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
EOF

# ========== 30: Enhancement A (BC + Aux) ==========
cat > "$STEPS_DIR/30_ENHANCEMENT_A_BC_AUX.prompt.md" <<'EOF'
# Step 3 — Enhancement A: Behavior Cloning Warm-Start + Auxiliary Forecasting Head

**Objectives**
- Add an optional **BC warm-start** stage seeded by simple “teacher” strategies.
- Add an optional **Auxiliary Head** predicting short-horizon returns (r_{t+1}, r_{t+3}, r_{t+5}) to improve representation.
- **Zero assumptions**: adapt to existing model/feature/training layout.

**Constraints**
- Do not rename files. Extend existing modules where possible.
- Add **config toggles** to enable/disable BC and AuxHead.
- Ensure **no feature leakage**: labels and targets must be computed with data available at decision time (t-1 alignment).

**Actions**
1) Design & Patch Plan
   - Write `docs/DESIGN_BC_AUX.md`: exact files/functions to extend, config keys, and how to initialize RL from BC weights.
   - Write `docs/PATCHPLAN_BC_AUX.md`: granular diffs to apply (paths + line anchors).
2) Implementation
   - Add teacher strategies (e.g., Opening Range Breakout + retest; VWAP band fade; ATR swing) in the repo-consistent place.
   - Implement BC label generation that uses existing loaders and time index; save labels to the repo’s standard data format/dir.
   - Add an AuxHead to the model behind a config flag; incorporate aux loss (weighted).
   - Implement BC training loop that writes a checkpoint compatible with the existing RL trainer.
3) Tests
   - BC label smoke test (tiny date span) verifying ≥20% LONG+SHORT combined.
   - Model shape test with/without AuxHead.
   - BC training 1–2 epochs overfit sanity; checkpoint saved.
4) Run tests and commit.

**Deliverables**
- New BC and AuxHead capabilities (opt-in).
- Tests passing; minimal code changes reusing existing modules.
- Commit: `feat: BC warm-start and auxiliary forecasting head with tests and docs`
EOF

# ========== 40: Enhancement B (RL Constraints) ==========
cat > "$STEPS_DIR/40_ENHANCEMENT_B_RL_CONSTRAINTS.prompt.md" <<'EOF'
# Step 4 — Enhancement B: RL Constraints to Prevent HOLD Collapse and Enforce Cadence

**Objectives**
- Add optional constraints to the existing RL loop:
  - Entropy schedule (start high → decay to floor).
  - KL-to-uniform prior penalty (decay to 0).
  - Early-training action floors: π(LONG), π(SHORT) ≥ ε for a warm-up fraction.
  - Trade-count regularizer towards target band; overtrade penalty.
  - Optional small HOLD tax (training only).
  - Max trades/day cap (if environment/backtester supports it).

**Constraints**
- Do not replace the RL algorithm. Extend current trainer in place.
- All features are behind config toggles; defaults preserve current behavior.

**Actions**
1) Design & Patch Plan
   - `docs/DESIGN_RL_CONSTRAINTS.md`: exact insertion points (logits, loss assembly), where to count trades/day, and how to pass configs.
   - `docs/PATCHPLAN_RL_CONSTRAINTS.md`: exact diffs.
2) Implementation
   - Add schedules and regularizers per plan; ensure back-compat.
3) Tests
   - Unit tests for schedule shapes (entropy/KL).
   - Unit test that action floors increase LONG/SHORT early without exploding volume.
   - Tiny integration test demonstrating improved action diversity on a small slice.
4) Run tests and commit.

**Deliverables**
- Optional RL constraints wired in.
- Commit: `feat: RL constraints (entropy, KL, action floors, cadence regularizer, hold tax) + tests`
EOF

# ========== 50: Enhancement C (Multiticker + Embeddings) ==========
cat > "$STEPS_DIR/50_ENHANCEMENT_C_MULTITICKER_EMBEDS.prompt.md" <<'EOF'
# Step 5 — Enhancement C: Multi-Ticker Training with Ticker/Regime Embeddings

**Objectives**
- Reuse the existing loaders to emit a `ticker_id` (or repo-equivalent) without breaking single-ticker paths.
- Add regime tags (volatility tercile, trend slope sign, liquidity tercile, time-of-day bucket) using leak-safe rolling stats.
- Add optional embedding layers for ticker_id and regime tags; concatenate with existing encoded features.

**Constraints**
- No renames; minimal edits.
- All additions are opt-in via config.

**Actions**
1) Design & Patch Plan
   - `docs/DESIGN_MULTITICKER_EMBEDS.md`
   - `docs/PATCHPLAN_MULTITICKER_EMBEDS.md`
2) Implementation
   - Compute regime tags in the current feature pipeline style.
   - Inject `ticker_id` using or extending existing mapping logic.
   - Add embedding modules and merge into model forward pass when enabled.
3) Tests
   - Loader test: shapes & null checks; verifying regime tags and ticker_id present when enabled.
   - Model test: forward shapes with/without embeddings.
4) Run tests and commit.

**Deliverables**
- Optional multiticker + embeddings capability.
- Commit: `feat: multiticker training with ticker/regime embeddings + tests`
EOF

# ========== 60: OOS Evaluation ==========
cat > "$STEPS_DIR/60_OOS_EVAL_WINDOWS_AND_GATES.prompt.md" <<'EOF'
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
EOF

# ========== 70: Tuning Orchestrator ==========
cat > "$STEPS_DIR/70_TUNING_ORCHESTRATOR_WRAP_EXISTING_TRAIN.prompt.md" <<'EOF'
# Step 7 — Tuning Orchestrator (Wrap Existing Trainer)

**Objectives**
- Implement a lightweight orchestrator that **wraps the current training entrypoint** to run 8–16 trials over:
  - learning rate, entropy schedule, KL weight, clip, γ, λ,
  - action floor ε, aux loss weight, embedding dims (if enabled).
- Selection criteria:
  - Must pass cadence/action-mix/DD gates,
  - Rank by OOS median expectancy (after costs) and MAR.

**Constraints**
- Do not replace the trainer. Use existing CLI/config override mechanisms.

**Actions**
1) Design plan `docs/DESIGN_TUNING.md`: how to pass overrides with the repo’s current config/CLI scheme.
2) Implement the orchestrator (module or script) that writes per-trial configs and launches training + OOS evaluation.
3) Add a summarizer collecting OOS metrics, producing a top-N table.
4) Tiny dry-run test (1–2 toy trials).
5) Commit: `feat: tuning orchestrator reusing existing trainer; OOS-based selection`
EOF

# ========== 80: Makefile Wiring ==========
cat > "$STEPS_DIR/80_MAKEFILE_WIRING_EXISTING_CLI.prompt.md" <<'EOF'
# Step 8 — Makefile Wiring (Existing Entrypoints Only)

**Objectives**
- Create or update the Makefile to expose targets that call **existing entrypoints** only:
  - `tickers_train`: print resolved training tickers using current tooling.
  - `tickers_test`: print resolved backtest tickers using current tooling.
  - `bc_train`: run BC warm-start (short window acceptable if configured).
  - `rl_train`: run RL fine-tune from BC checkpoint with constraints enabled.
  - `backtest_oos`: run OOS evaluation across windows & held-out tickers.
  - `tune`: run the tuning orchestrator.

**Constraints**
- If a flag does not exist, do not invent it. Document manual edits with exact file/line references instead.

**Actions**
1) Inspect current CLI scripts and print exact commands each target will run (echo before execute).
2) Implement targets calling those commands.
3) Commit: `build: Makefile targets for tickers, bc_train, rl_train, backtest_oos, tune`
EOF

# ========== 90: Dry Runs & Sanity ==========
cat > "$STEPS_DIR/90_DRY_RUNS_AND_SANITY.prompt.md" <<'EOF'
# Step 9 — Dry Runs & Sanity Checks (Token-Safe)

**Objectives**
- Prove wiring without heavy runs:
  - DR1: Print resolved training & testing tickers (using existing “print”/“verbose”/“dry-run” modes if available).
  - DR2: Run BC on a 2–3 day span for 1–2 symbols already cached locally; verify checkpoint creation.
  - DR3: Run OOS evaluator on ~5 trading days; verify cadence/action-mix fields appear.

**Actions**
1) Create `docs/DRY_RUNS.md` with exact commands discovered for the repo.
2) Run the dry runs and collect artifacts under existing report directories.
3) Commit: `docs: dry runs completed; wiring validated`
EOF

# ========== 99: Full Run Commands ==========
cat > "$STEPS_DIR/99_FULL_RUN_COMMANDS.prompt.md" <<'EOF'
# Step 10 — Full Run Sequence (Use Existing Entrypoints)

**Objectives**
- Provide the final, end-to-end commands (Make targets or scripts) to:
  1) Print tickers for training and backtest.
  2) Run BC warm-start (full span per project settings).
  3) Run RL fine-tune with constraints.
  4) Run OOS evaluation across windows and held-out tickers.
  5) Optionally run tuning orchestrator and select top-N.

**Actions**
- Create `docs/FULL_RUNBOOK.md` listing exact commands and expected output locations consistent with this repo’s conventions.
- Do not execute heavy runs automatically here. Just output the runbook.
- Commit: `docs: full runbook`
EOF

# ========== 02: Operator README (how to use with Codex CLI) ==========
cat > "$BASE_DIR/README_HOW_TO_USE_WITH_CODEX.md" <<'EOF'
# How to Use These Prompts with Codex CLI

1) **Open a new Codex chat** in your repo root. Paste the content of:
   - `00_PERSONA_AND_GUARDRAILS.prompt.md` (once per session)
   - `01_ORCHESTRATOR.prompt.md`

2) Then, **sequentially feed** each step file from `codex/steps/` to Codex, one at a time, waiting for Codex to:
   - Print its PLAN (files, functions, patch points),
   - Apply minimal changes reusing existing code,
   - Run tests and QA gates,
   - Commit with the specified message.

**Order:**
- `steps/10_BRANCH_AND_BASELINE.prompt.md`
- `steps/20_AUDIT_REPO_AND_TICKERS.prompt.md`
- `steps/30_ENHANCEMENT_A_BC_AUX.prompt.md`
- `steps/40_ENHANCEMENT_B_RL_CONSTRAINTS.prompt.md`
- `steps/50_ENHANCEMENT_C_MULTITICKER_EMBEDS.prompt.md`
- `steps/60_OOS_EVAL_WINDOWS_AND_GATES.prompt.md`
- `steps/70_TUNING_ORCHESTRATOR_WRAP_EXISTING_TRAIN.prompt.md`
- `steps/80_MAKEFILE_WIRING_EXISTING_CLI.prompt.md`
- `steps/90_DRY_RUNS_AND_SANITY.prompt.md`
- `steps/99_FULL_RUN_COMMANDS.prompt.md`

**Notes**
- If the repo lacks certain flags, Codex will document precise file/line references for manual edits instead of guessing interfaces.
- Minute-data start date must remain **2020-10-01** or later; Codex should show exactly how your current scripts enforce this.
- Always keep diffs small; run tests after every change.
EOF

echo "✅ Codex runbook created at: $BASE_DIR"
ls -R "$BASE_DIR"
