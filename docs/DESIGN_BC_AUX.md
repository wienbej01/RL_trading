# Enhancement A — Behavior Cloning Warm-Start + Auxiliary Head (Design)

Goals
- Optional BC warm-start from a teacher policy or prior checkpoint.
- Optional Auxiliary Head that predicts short-horizon returns to shape representation.
- Backward-compatible: disabled by default, no behavior/ABI changes.

Config Keys (read by `src/rl/train.py`)
- `train.bc.enabled` (bool, default false)
- `train.bc.init_from` (str, optional path to SB3 checkpoint to warm-start)
- `train.bc.notes` (str, optional freeform)
- `train.aux_head.enabled` (bool, default false) — currently logged only

Placement
- BC warm-start: before PPO model construction. If `init_from` exists, load via `RecurrentPPO.load(path, env=vec_env)` and continue training. Otherwise, fall back to fresh init.
- Aux Head: phase 1 adds configuration + logging placeholder. Future phase can wire an aux loss in a custom policy (e.g., `multi_ticker_ppo_lstm_policy.py`) without changing public interfaces.

Data & Safety
- No new file formats required. BC uses an existing SB3 model if provided.
- Training continues to honor existing env construction and VecNormalize usage.

Non-Goals (this phase)
- Building teacher datasets or supervised training loops.
- Modifying observation/feature schemas or adding new CLI flags.

Edge Cases
- If `init_from` is provided but unreadable → log warning, use fresh init.
- Device/cuda selection is preserved from current settings logic.

Verification
- When enabled with a valid checkpoint, logs will report “Warm-starting from …”.
- Otherwise, behavior is identical to previous training flow.
