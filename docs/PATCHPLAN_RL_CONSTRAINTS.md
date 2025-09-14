# Patch Plan — Enhancement B (RL Constraints)

Scope (minimal, opt‑in)
- Implement `env.trading.max_entries_per_day` in `src/sim/env_intraday_rl.py`.
- No public interfaces change; config‑driven only.
- Keep KL prior penalty as a future item (custom loss required).

Files & Edits
1) `src/sim/env_intraday_rl.py`
   - At step open logic (when `desired_dir != 0 and self.pos == 0`):
     - Read `max_entries_per_day` from `self.config['env']['trading']` if present.
     - If `_daily_trade_count >= max_entries_per_day`, skip opening (set `desired_dir = 0`).
   - Add a brief debug log for skip reason.

Validation
- Daily counter already exists and resets on date change.
- Per‑hour cap remains intact and is evaluated before/after daily cap.
- Defaults remain unchanged; behavior only when config is set > 0.
