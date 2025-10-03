## Stabilize PPO with normalization, schedules, and eval callbacks

- Added VecNormalize (obs+reward) with persistence (checkpoints/vecnorm.pkl)
- Switched single-ticker training to SubprocVecEnv with configurable `rl.n_envs`
- Implemented linear schedules for learning rate (3e-4→1e-5) and clip range (0.2→0.1)
- Added evaluation callback saving best model and reducing LR on plateau
- Seed threaded through numpy/torch/random; logged into training summary
- Policy net arch set to ReLU MLP with [256,256] for both actor and critic
- TensorBoard logging enabled under `logs/tensorboard/`
- Configurable block `rl:` introduced in configs/settings.yaml
- Added KLStopCallback, AdaptiveLRByKL, and LiveLRBump callbacks; wired into trainer
- New script `scripts/lr_bump.sh` to nudge LR mid‑run without restart
- Static low‑price universe runner `scripts/run_lowpx_portfolio.sh` and universe list

## Align features and diagnostics to OHLCV proxies; add artifacts and audit

### Added
- OHLCV microstructure proxies alignment across packs, pipeline, diagnostics, and runner.
- L1→OHLCV remapping in pipeline when quotes are unavailable.
- `corr_action_flow` diagnostics and guaranteed flow proxy in steps.parquet (derive if missing).
- Artifacts writer (`src/utils/artifacts.py`) and audit script (`scripts/audit_pipeline.py`).
- Feature cache key augmentation and `--no-cache` flag in `scripts/run_wf.py`.

### Changed
- Curated packs now resolve to `curated_topN ∪ MICROSTRUCTURE_OHLCV` and expose `LAST_CURATED_CACHE_TOKEN`.
- Runner logs, checks, and tests updated to use OHLCV proxies nomenclature.

### Tests
- Added tests for OHLCV proxies, diagnostics flow correlation, and artifacts writing.
