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
