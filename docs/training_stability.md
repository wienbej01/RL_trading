# Training Stability & Normalization

This note documents the stability/normalization changes that improve multi‑ticker PPO training.

## PPO Tuning

- learning_rate: 1e-4
- batch_size: 4096
- n_steps: 2048
- vf_coef: 0.7
- ent_coef: 0.015
- target_kl: 0.075

These are read from the `ppo:` block in `configs/settings.yaml` (fallback to legacy `train/training` keys).

## Normalization

Enable VecNormalize via:

```
normalize:
  obs: true
  reward: true
  per_ticker: true
  reward_scale: 0.5
```

Stats are saved to `vecnormalize.pkl` alongside the model and reloaded for evaluation.

## Data Hygiene (pre‑features)

- Strict de‑dup by (timestamp,ticker), keep last
- Bounded forward fill ≤2 bars within a session
- Drop tiny islands < 5 bars per ticker
- Align masks separately for `data` and `features` indices to avoid length mismatches

## Evaluation/Trades

- `trades.csv` always emitted
- `evaluation_results.json` includes total/win/loss counts, win_rate, avg win/loss, profit_factor, largest win/loss, avg duration, long/short counts, gross/avg/median PnL

## Usage Example

```
PYTHONPATH=. venv/bin/python scripts/run_multiticker_pipeline.py \
  --config configs/settings.yaml \
  --train-start 2024-06-01 --train-end 2024-08-31 \
  --test-start  2024-09-23 --test-end  2024-09-30 \
  --tickers "LYFT RUN SNAP PLUG SQ UBER ROKU ENPH CRWD DDOG OKTA ETSY NET SPY" \
  --test-tickers "GOOGL CHPT UAL DOCU CROX FSLR PINS ZM TWLO RBLX SOFI" \
  --output-dir results/local_multi_norm
```

## Acceptance Targets (for 100k steps)

- Median per‑ticker explained variance ≥ 0.15
- KL median ~0.03–0.07, no persistent early stops
- Reward std (post‑scale) in 0.5–2.0 range

