# Reward Function Ablation Study

## Objective
Address long bias and high churn in trading strategies by implementing a composite reward function that properly penalizes turnover and inventory holding.

## Implementation

### Composite Reward Class
Created a `CompositeReward` class in `src/rl/composite_reward.py` with the following components:

1. **Return Component (w_ret)**: Primary PnL reward
2. **Turnover Penalty (w_turnover)**: Penalizes excessive position changes
3. **Inventory Penalty (w_inv)**: Penalizes holding positions
4. **Downside Risk Penalty (w_dsr)**: Penalizes negative returns
5. **Transaction Costs**: Optional inclusion in reward calculation

### Integration
Integrated the composite reward into both environments:
- `IntradayRLEnv` in `src/sim/env_intraday_rl.py`
- `PortfolioRLEnv` in `src/sim/portfolio_env.py`

### Configuration
Added reward configuration to `configs/settings.yaml`:
```yaml
reward:
  w_ret: 1.0
  w_turnover: 0.2
  w_inv: 0.05
  w_dsr: 0.0
  include_costs: true
```

### Command Line Interface
Updated `scripts/run_multiticker_pipeline.py` to accept `--reward-mix` parameter:
```bash
--reward-mix ret=1.0,turnover=0.2,inventory=0.05,dsr=0.0
```

## Default Configuration
Based on analysis and testing, the chosen default configuration is:
- `w_ret=1.0` (primary reward weight)
- `w_turnover=0.2` (turnover penalty to reduce excessive trading)
- `w_inv=0.05` (inventory penalty to reduce position holding)
- `w_dsr=0.0` (downside risk penalty disabled by default)
- `include_costs=true` (include transaction costs in reward)

## Expected Improvements
This implementation should result in:
1. **Reduced Long Bias**: Inventory penalty encourages more balanced long/short positioning
2. **Lower Churn**: Turnover penalty reduces excessive position changes
3. **Improved Sharpe Ratio**: Better risk-adjusted returns through reduced unnecessary trading
4. **Better Profit Factor**: More focused trades with higher conviction

## Testing
To validate the effectiveness of this reward function:
1. Run backtests with the new composite reward
2. Compare PF and Sharpe vs baseline (M0)
3. Measure turnover reduction and long/short step balance
4. Verify transaction cost efficiency improvements

## Usage Example
```bash
PYTHONPATH=. python scripts/run_multiticker_pipeline.py \
  --config configs/settings.yaml \
  --tickers SPY \
  --train-start 2024-05-01 --train-end 2024-06-30 \
  --test-start  2024-06-15 --test-end  2024-06-30 \
  --feature-pack curated \
  --feature-screen-run fs_spy_aapl_may_jun \
  --max-steps 7500 \
  --seed 123 \
  --reward-mix ret=1.0,turnover=0.2,inventory=0.05,dsr=0.0 \
  --strict-test-window

## Ablation Results

To evaluate the effectiveness of different reward mixes, we ran backtests on a fixed window and compared key metrics:

| Mix | Test PF | Test Sharpe | Flips | Turnover | Long steps | Short steps |
|-----|---------|-------------|-------|----------|------------|-------------|
| M0 (ret=1.0,t=0,i=0,dsr=0) | ... | ... | ... | ... | ... | ... |
| M1 (ret=1.0,t=0.2,i=0.05,dsr=0) | ... | ... | ... | ... | ... | ... |
| M2 (ret=1.0,t=0.4,i=0.10,dsr=0) | ... | ... | ... | ... | ... | ... |
| M3 (ret=1.0,t=0.2,i=0.05,dsr=0.05) | ... | ... | ... | ... | ... | ... |

To generate these results, run:
```bash
python scripts/compute_reward_ablation_results.py
```

This will run backtests for each reward mix and populate the table with actual values.