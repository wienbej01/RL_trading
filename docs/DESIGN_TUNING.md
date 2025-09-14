# Tuning Orchestrator (Design)

Goal
- Run a small set of trials over PPO hyperparameters using the existing trainer, then score via OOS evaluation and select top results.

Approach
- Script `scripts/tune_orchestrator.py`:
  - Inputs: `--config`, `--data`, `--features`, `--output`, `--trials`, `--total-steps`, `--test-tickers`.
  - For each trial: sample from a small search space for learning rate, entropy, clip, gamma, gae_lambda. Train via `train_ppo_lstm` with a `TrainingConfig` override. Save to `trial_<k>/model`.
  - Run OOS evaluation (import functions from `scripts.oos_eval`), collect metrics/gates, and write `trial_<k>/oos.json`.
  - Produce `summary.json` with per‑trial metrics and the best trial by gates + median sharpe.

Constraints
- No interface changes to the trainer/environments. All additions are optional and live in `scripts/`.

Scoring
- Prefer trials that pass more gates (cadence, action diversity, expectancy, max DD). Tie‑break on median sharpe; then median trades/day.

Outputs
- `summary.json` and a CSV‑like JSON table of trials for easy review.
