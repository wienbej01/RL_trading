# Patch Plan — Enhancement A (BC Warm-Start + Aux Head)

Scope
- Minimal, opt-in hooks in `src/rl/train.py`.
- No public interface changes; only reads optional config keys.

Files & Edits
1) `src/rl/train.py`
   - Read config keys: `train.bc.enabled`, `train.bc.init_from`, `train.aux_head.enabled`.
   - After building `vec_env` and before model construction:
     - If `bc.enabled` and `bc.init_from` exists → load model via `RecurrentPPO.load(path, env=vec_env, device=effective_device)` and skip fresh init.
     - Else → construct fresh `RecurrentPPO` as before.
   - Log whether BC warm-start is used; log aux-head flag for visibility.

Non-Functional Changes
- Add `docs/DESIGN_BC_AUX.md` (this design) and this patch plan document.

Validation
- Run `pytest -q` to ensure no new failures (feature is disabled by default).
- Run a tiny train invocation with a known checkpoint to see the warm-start log.
