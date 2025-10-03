# `multiticker_enhancement.md`

**Owner:** Jacob (Head of Quant Trading System Development)
**Purpose:** Enable robust **train tickers ≠ test tickers** workflows; fix zero-trade outcomes; harden data→features→env→RL→backtest piping with explicit tests and guardrails.
**Context evidence:** The most recent run logged legacy-reward usage and zero trades (Sharpe NaN; total_trades=0). 

---

## Roadmap at a Glance

* **Sprint 0:** Baseline reproduction & observability
* **Sprint 1:** Dual-universe data ingestion & calendar harmonization
* **Sprint 2:** Feature generation parity (train & test), schema lock, leakage guards
* **Sprint 3:** Normalization & schema alignment (VecNormalize stats portability)
* **Sprint 4:** Environment gating & reward path (allowed tickers, composite reward)
* **Sprint 5:** Asset-aware execution/risk & barrier telemetry
* **Sprint 6:** Backtester, metrics & baselines; guardrails fail on “zero trades”
* **Sprint 7:** Orchestration & reproducibility (multiticker wrapper + leaderboard)

Each sprint has: **Goal → Tasks → Tests → Acceptance → Evidence to record**.

> **Run discipline (all sprints):**
>
> ```
> export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
> export MPLBACKEND=Agg PYTHONHASHSEED=0
> ```

---

## Sprint 0 — Baseline Reproduction & Observability

**Goal:** Reproduce the zero-trade run with rich telemetry to pinpoint the exact choke point.

### Tasks

1. **Re-run the user scenario** (TSLA,AAPL,MSFT train; SPY test) and **persist full artifacts**:

   * `pipeline_config.json`, `train_data.parquet`, `test_data.parquet`, `train_features.parquet`, `test_features.parquet`, `metrics.json`, `trades.csv`, `reward_breakdown.csv`, `equity_curve.csv`.
2. Add **action telemetry** (episode-level CSV):

   * Columns: `ts, ticker, action_logits, action_prob_long, action_prob_short, chosen_action(-1/0/1), pos_prev, pos_next`.
3. Add **feature snapshot** CSVs (first/last 200 rows) for **each ticker** in train/test.

### Suggested command

```
PYTHONPATH=. python -X faulthandler scripts/run_multiticker_pipeline.py \
  --config configs/settings.yaml \
  --train-start 2024-06-01 --train-end 2024-07-31 \
  --test-start  2024-08-01 --test-end  2024-08-31 \
  --tickers TSLA AAPL MSFT \
  --test-tickers SPY \
  --output-dir results/tsla_aapl_msft_train_spy_test --strict-test-window
```

### Tests

* **Unit:** writer util creates all required files (non-empty).
* **Integration:** action telemetry rows == test bars × N_test_tickers.
* **Check:** `metrics.json.total_trades == len(trades.csv)`; non-negativity of costs.

### Acceptance

* Reproduced zero-trade run with **complete** artifacts; action telemetry shows policy behaviour.

### Evidence

* Attach: `results/.../metrics.json`, `trades.csv` (even if empty), `action_telemetry.csv`.

---

## Sprint 1 — Dual-Universe Data Ingestion & Calendar Harmonization

**Goal:** Guarantee data exists & aligns for **all train tickers** and **all test tickers** even when sets differ.

### Tasks

1. **Data loader**: add `--train-tickers` and `--test-tickers` flow; load both universes explicitly.
2. **Market calendar join**:

   * Build a **union minute index** over the full `[train_start, test_end]`.
   * Left-join each (ticker, split=train/test) to the union; **forward-fill only fields allowed** (e.g., OHLCV forbids FFill across session gaps).
   * Mark gaps with a `missing_bar` boolean column.
3. **RTH mask**: apply consistent RTH filter; write `rth_mask.csv`.

### Tests

* **Unit:** For a synthetic SPY present only in test, loader returns non-empty test frame.
* **Property:** `len(union_index) ≥ max(len(index[t]))` and **no duplicate timestamps** across tickers.
* **Leakage guard:** Ensure no test bars appear in train slices.

### Acceptance

* `*_data.parquet` exist for **all** tickers (train/test) with a **shared, monotonic** minute index.

### Evidence

* `data_audit.json` summarizing rows per ticker per split; `missing_bar_rates.csv`.

---

## Sprint 2 — Feature Parity (Train & Test), Schema Lock, Leakage Guards

**Goal:** Generate identical **feature catalogs** for train & test universes, and **lock the schema**.

### Tasks

1. **Feature pack** runner accepts **(ticker, split)** and writes:

   * `features/<split>/<TICKER>_<pack>_<start>_to_<end>.parquet`
2. **Schema lock**:

   * After train feature generation, write `features/schema.json` (**ordered** list of feature names & dtypes).
   * At test time, **reindex columns to `schema.json`**, dropping extras and inserting missing with zeros (or documented fill).
3. **Leakage/NaN audits**:

   * Reject features with future leakage (e.g., shift < 0).
   * Emit `feature_nan_report.csv` per (ticker, split).

### Tests

* **Unit:** Schema aligner drops extras and adds missings to match train schema exactly.
* **Failure tests:** Inject a future-leaking feature → pipeline must hard-fail.
* **Integration:** Test ticker only in test (SPY) gets full aligned feature matrix.

### Acceptance

* `test_features.parquet` has **identical column order** and dtypes to `train_features.parquet`.

### Evidence

* `features/schema.json`, `feature_nan_report.csv`, `feature_leakage_checks.log`.

---

## Sprint 3 — Normalization Portability (VecNormalize & Stats)

**Goal:** Persist train **normalization stats** and apply them to test features and any later inference.

### Tasks

1. Persist `VecNormalize` stats after train: `norm/train_stats.json`.
2. At test time, **load** `train_stats.json` and transform test features; **do not recompute**.
3. Add `--norm-policy {none, zscore, robz, minmax}` CLI; default `zscore`.

### Tests

* **Unit:** Applying stats to test does not change column count or order; no NaNs created.
* **Integration:** Re-run with and without normalization; confirm action entropy & reward magnitudes differ (sanity).

### Acceptance

* `norm/train_stats.json` exists; test features transformed with those stats (hash check of column order).

### Evidence

* `norm_apply.log`, `norm_sanity_metrics.json` (means/stds after transform).

---

## Sprint 4 — Environment Gating & Reward Path

**Goal:** Ensure **allowed tickers** gating and **reward kind** operate as expected during **backtest**, not suppressing trades inadvertently.

### Tasks

1. **Allowed tickers**:

   * Wire `allowed_trade_tickers = args.test_tickers` into the backtest env config.
   * If empty, default allow **all** test tickers; log the effective set.
   * In `step()`, only apply gating mask when `allowed_mask` is non-None.
2. **Reward**:

   * Default `kind='composite'` when `--reward-mix` present; log final weights and `reward_scaling`.
   * Ensure **reward terms** (ret, turnover, inventory, dsr) are **non-zero** on synthetic scenarios.
3. **Min/Max hold & EOD**:

   * Prevent EOD flattening from **zeroing every entry** (block new entries close to EOD, but do not cancel mid-day flips unless justified).

### Tests

* **Unit:** Gating unit test—given `allowed=['SPY']`, actions for non-SPY are forced to 0, SPY action remains unchanged.
* **Unit:** Composite reward test—monotone up series → positive `ret` reward; high flipping → positive turnover penalty.
* **Integration:** Synthetic test where policy should enter SPY during test (train on other tickers) → at least one trade logged.

### Acceptance

* Backtest over SPY-only test period yields **non-zero** `total_trades` when synthetic triggers are present; reward CSV shows non-zero magnitudes.

### Evidence

* `env_gate_test.log`, `reward_breakdown.csv` with non-zero component stats.

---

## Sprint 5 — Asset-Aware Execution/Risk & Barrier Telemetry

**Goal:** Make execution math realistic per asset class and expose **barrier reasons**.

### Tasks

1. **Costs & sizing**:

   * For equities/ETFs: `point_value=1.0`, **per-share** commission & spread; ensure **share rounding**.
   * ATR-based units: cap with `units_per_ticker`, respect `risk_budget_per_ticker`.
2. **Barriers**:

   * Log `barrier_reason ∈ {STOP, TAKE, TIME, EOD}` per exit; include ATR/time limit parameters in artifacts.
3. **Guardrails**:

   * Enforce daily loss **kill switch**; limit **max_entries_per_day** per ticker.

### Tests

* **Unit:** Fee & spread application math on a tiny trade set (exact expected PnL).
* **Integration:** Trades show `barrier_reason` populated; `metrics.json` aggregates match `trades.csv`.

### Acceptance

* `trades.csv` present with realistic share sizes; PnL reconciles with fee/spread assumptions; barrier fields populated.

### Evidence

* `execution_audit.json`, `barrier_examples.csv`.

---

## Sprint 6 — Backtester, Metrics & Baselines; Guardrails on “Zero Trades”

**Goal:** Make analytics conclusive and **fail fast** on pathological runs.

### Tasks

1. **Backtest metrics**: compute `total_trades, win_rate, profit_factor, annual_vol, Sharpe, DSR, MDD`; avoid NaN by safe-divides.
2. **Baselines** (`scripts/run_baselines.py`):

   * **OR breakout 1:2 RR**, **VWAP reversion**, **Buy-and-Hold** sharing the **same data loader/execution**.
   * Emit `baseline_leaderboard.csv`.
3. **Guardrails** (pipeline exit code):

   * If `total_trades == 0` → **FAIL** with explanation.
   * If `median(DSR) ≤ 0` (walk-forward) → **FAIL**.

### Tests

* **Unit:** Metrics safe-divide (no NaNs/inf).
* **Integration:** A trivial synthetic dataset yields trades & non-zero metrics; baseline runner outputs CSV; guardrails fail on deliberately empty trades.

### Acceptance

* Runs with no trades must **fail**; standard runs produce complete metrics & baselines for comparison.

### Evidence

* `metrics.json`, `baseline_leaderboard.csv`, `guardrail_log.txt`.

---

## Sprint 7 — Orchestration & Reproducibility (Wrapper + Leaderboard)

**Goal:** Harden the multi-ticker wrapper to support **train ≠ test universes** with consistent artifacts.

### Tasks

1. `run_multiticker_pipeline.py`:

   * Accept explicit `--tickers` (train) and `--test-tickers` (test) and pass both to loader & features.
   * Write per-ticker folders under `results/.../` for data, features, model, backtest.
   * Produce `leaderboard.csv` combining test-ticker metrics, sorted by DSR (then Sharpe), including `total_trades`.
2. **Config snapshotting**:

   * Write `run_config.yaml` (all resolved CLI + defaults) and `schema.json` hashes.
   * Save `norm/train_stats.json` and a **hash of feature column order**.
3. **Single-command reports**:

   * `generate_performance_report.py` composes HTML (equity, drawdown, trade map, per-ticker metrics).

### Tests

* **Integration:** Train on TSLA,AAPL,MSFT; test on SPY—pipeline **completes**, emits per-ticker artifacts; leaderboard shows SPY row.
* **Repro:** Re-run with different `--seed`; metric deltas < 30% (sanity).

### Acceptance

* `leaderboard.csv` exists (at least test tickers present); `run_config.yaml` + `schema.json` hashes saved; HTML report renders.

### Evidence

* `leaderboard.csv`, `run_config.yaml`, `report.html` (screenshots ok).

---

## Working Prompts (paste to Codex/VS Code / GLM 4.6)

### A) Data/Features (Sprints 1–3)

> **Role:** Senior data engineer.
> Implement dual-universe loading & calendar join with a union minute index; add `missing_bar` flags. Build feature runner that outputs per-ticker **train** and **test** parquet sets. Emit `features/schema.json` after train, then align test by that schema. Add leakage checks (forbid negative shifts), NaN audits, and `norm/train_stats.json` persistence with a loader that applies those stats to test. Provide unit tests and integration tests described above, and ensure all artifacts land under `results/<run_id>/...`.

### B) Env/Reward (Sprint 4)

> **Role:** Quant RL engineer.
> Wire `allowed_trade_tickers` from CLI test tickers; only gate when mask is set. Default reward kind to `composite` when `--reward-mix` present; log weights & `reward_scaling`. Ensure synthetic monotone series yields positive `ret` reward and that high flips increase turnover penalty. Fix EOD logic to avoid wiping eligible entries mid-session. Provide unit & synthetic integration tests.

### C) Execution/Backtest (Sprints 5–6)

> **Role:** Execution & analytics engineer.
> Make equity/ETF execution per-share with realistic fees/spread and share rounding; cap ATR units. Emit `barrier_reason`. Harden metrics (safe-divide) and add `scripts/run_baselines.py`. Implement guardrails: exit non-zero on zero-trade runs. Provide tests and example command lines.

### D) Wrapper/Reporting (Sprint 7)

> **Role:** Platform engineer.
> Update `scripts/run_multiticker_pipeline.py` to fully support train≠test universes; generate per-ticker artifacts and a `leaderboard.csv`. Snapshot configs; hash feature schema; produce a compact HTML performance report. Provide a smoke-run script and README usage notes.

---

## Final Acceptance (end-to-end)

1. **Train:** TSLA,AAPL,MSFT (Jun–Jul 2024)
   **Test:** SPY (Aug 2024)
   → **Non-zero trades on SPY**, complete metrics, and populated barrier reasons.
2. **Guardrails:** Run fails if trades=0 or if metrics are NaN/Inf; otherwise pass.
3. **Baselines:** RL DSR ≥ max(baseline DSRs) in test; drawdown ≤ worst baseline.
4. **Repro:** Rerun with a different seed; metric drift within tolerance (<30%).
5. **Artifacts:** `leaderboard.csv`, `wf_summary.csv` (if executed), `decision.json`, HTML report, `schema.json`, `norm/train_stats.json`, `*features*.parquet`, `*data*.parquet`, `trades.csv`, `metrics.json`, `reward_breakdown.csv`.

---

## “Call via codex.cli” Hints

* Keep this file under `docs/plans/multiticker_enhancement.md`.
* After each sprint/task, **append**:

  * `Status: done / in-progress / blocked`
  * `Evidence: <paths to artifacts>`
  * `Diff/PR: <branch or PR link>`
* Commit:

  ```
  git add docs/plans/multiticker_enhancement.md
  git commit -m "Plan: update status (Sprint X Task Y)"
  ```

---

### Quick Smoke After Sprint 4+

```
PYTHONPATH=. python -X faulthandler scripts/run_multiticker_pipeline.py \
  --config configs/settings.yaml \
  --train-start 2024-06-01 --train-end 2024-07-31 \
  --test-start  2024-08-01 --test-end  2024-08-31 \
  --tickers TSLA AAPL MSFT \
  --test-tickers SPY \
  --feature-pack curated --reward-mix ret=1,turnover=0.1,inventory=0.01,dsr=0.1 \
  --reward-scale 0.1 --strict-test-window \
  --output-dir results/tsla_aapl_msft_train_spy_test
```

> Expect: **non-zero** SPY trades; if still zero, guardrails must **fail** with a clear reason.

---

**End of plan.**


Medical References:
1. None — DOI: file_00000000ff6461faa4c8d361564a21c0
