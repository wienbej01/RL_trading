# Measurement Plan for RL Intraday System

## Overview
Unbiased evaluation framework for features/labels/rules, using walk-forward optimization (WFO), Monte Carlo ablations, attribution. Economic rationale: Ensures out-of-sample (OOS) robustness, quantifies alpha attribution to parsimonious signals, detects failures early (e.g., regime shifts). Guardrails: Costs parameterized (slippage/commission in PnL), signals bar_close+1, purged splits.

## KPIs
- **Action Balance**: 35-45% long/short/hold distribution (histogram over OOS); measures symmetry, target 40% each side/20% hold.
- **Sharpe Ratio**: >0.5 OOS (annualized, risk-free 2%); risk-adjusted performance benchmark.
- **Feature IC**: Avg Information Coefficient >0.03 (corr with 1-bar returns); predictive power.
- **Turnover**: 4-8 trades/day avg; balances activity vs over-trading costs.
- **Max DD**: <10% OOS; drawdown control.
- **Win Rate**: >55% (post-costs); expectancy from RR 1:2.

## Failure Modes & Thresholds
- **Action Bias**: >80% one side (e.g., all-long from drift); threshold: retrain with symmetric rewards if imbalance >60%.
- **Regime Overfitting**: IC std >0.01 across VIX bins (low<15, high>25); threshold: drop regime-sensitive features if delta IC>0.02.
- **High Drawdown**: >10% from poor sizing/noise; threshold: add circuit breaker if DD>5% in sim.
- **Low IC/Multicollinearity**: IC<0.01 or VIF>5; threshold: drop feature, orthogonalize via PCA.
- **Over-Trading**: Turnover >10/day; threshold: increase hold_penalty if exceeded.
- **Leakage**: Corr(label, lagged features)>0.01; threshold: embargo +1 bar, purge splits.

## Evaluation Framework
- **Backtest**: Day-by-day episodes, aggregate OOS metrics (embargo 60min between folds).
- **WFO**: Monthly folds (train 30 days, test 10 days); compute rolling Sharpe/IC.
- **Monte Carlo**: 1000 bootstraps on trades; 95% CI for KPIs (e.g., Sharpe CI [0.4,0.6]).
- **Ablations**: Subset features (e.g., no microstructure: expect Sharpe drop 0.1); SHAP for attribution.
- **Baselines**: Buy-hold, simple MA crossover; RL must outperform by 0.2 Sharpe.

## Code Sketch for Measurement
```python
from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings; warnings.filterwarnings('ignore')

def compute_ic(features, labels):
    ics = [pearsonr(f, labels)[0] for f in features.T]
    return np.mean(ics), np.std(ics)  # avg IC, stability

def vif_check(features):
    vif_data = pd.DataFrame(features)
    vif = [variance_inflation_factor(vif_data.values, i) for i in range(len(vif_data.columns))]
    return np.mean(vif), np.max(vif)  # avg/max VIF

def action_balance(actions):  # actions array -1/0/1
    longs, shorts, holds = np.sum(actions==1), np.sum(actions==-1), np.sum(actions==0)
    total = len(actions)
    return {'long_pct': longs/total, 'short_pct': shorts/total, 'hold_pct': holds/total, 'balance_ok': 0.35 <= shorts/total <= 0.45 and 0.35 <= longs/total <= 0.45}

# WFO example
for fold in range(num_folds):
    train_feats, train_labs = get_fold_data(fold, 'train')
    test_feats, test_labs = get_fold_data(fold, 'test')
    ic_mean, ic_std = compute_ic(train_feats, train_labs)
    if ic_mean < 0.03 or ic_std > 0.01: print('Regime failure')
    vmean, vmax = vif_check(train_feats)
    if vmax > 5: print('Multicollinearity: orthogonalize')
    # Backtest on test, compute Sharpe, turnover, etc.
    sharpe = backtest_sharpe(test_feats, test_labs)
    if sharpe < 0.5: print('Retraining needed')
```

## Mermaid WFO Evaluation Flow
```mermaid
graph TD
    A[Historical Data Split: Monthly Folds] --> B[Train Fold: Features + Labels]
    B --> C[Compute IC >0.03 & VIF<5?]
    C -->|No| D[Ablate/Drop Features; Retrain]
    C -->|Yes| E[Test Fold: Backtest Day-by-Day]
    E --> F[OOS Metrics: Sharpe>0.5, Balance 35-45%, DD<10%]
    F --> G{Monte Carlo 1000x: 95% CI OK?}
    G -->|No| H[Failure: Bias/Overfit Detected]
    G -->|Yes| I[Aggregate: Attribution SHAP, Compare Baselines]
    D --> E
    H --> J[Threshold Breach: e.g., IC<0.01 Drop, Sharpe<0.3 Retrain]
    I --> K[Approved: Proceed to Live Sim]
    style K fill:#90EE90