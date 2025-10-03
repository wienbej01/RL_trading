# Enhancement B — RL Constraints (Design)

Goals (opt‑in, backward‑compatible)
- Entropy schedule: start high → decay to floor.
- Early action floor: bias away from persistent HOLD during warm‑up.
- Cadence regularizer: target trades/day with soft constraint.
- HOLD tax: small penalty when flat and holding.
- Max trades/day cap: hard ceiling on opens per session.
- KL‑to‑uniform prior penalty: placeholder for future (requires custom loss).

What already exists
- Entropy anneal: `train.ent_anneal_final` (implemented via `EntropyAnnealCallback`).
- Warm‑up action forcing: `env.trading.force_open_epsilon`, `env.trading.force_warmup_frac`.
- Cadence regularizer (Lagrangian): `env.reward.activity.{target_per_day, lambda_init, lagrange_eta}`.
- HOLD tax: `env.reward.hold_penalty`.
- Per‑hour cap: `env.trading.max_trades_per_hour`.

New in this step
- Add daily cap: `env.trading.max_entries_per_day` enforced in `IntradayRLEnv`.
- Surface constraints in docs; keep KL prior as a documented future toggle.

Config examples
- train:
    ent_anneal_final: 0.001
- env:
    trading:
      force_open_epsilon: 0.05
      force_warmup_frac: 0.1
      max_trades_per_hour: 3
      max_entries_per_day: 5
    reward:
      hold_penalty: 0.001
      activity:
        target_per_day: 5
        lambda_init: 0.1
        lagrange_eta: 0.01

Verification
- When `max_entries_per_day` is exceeded, new opens are skipped for the rest of the day.
- Logs show skip reason; daily counter resets at date change.
- All features are no‑ops unless enabled in config.
