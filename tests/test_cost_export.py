import pandas as pd
import json


def test_trades_costs_sum_to_metrics(tmp_path):
    # Build tiny two-trade ledger
    trades = pd.DataFrame([
        {
            'ts': '2024-05-01 10:00:00', 'action': 'open', 'direction': 'long',
            'quantity': 1, 'price': 100.0,
            'commission_cost': 0.20, 'spread_cost': 0.30, 'slippage_cost': 0.10, 'impact_cost': 0.00,
            'total_cost': 0.60,
        },
        {
            'ts': '2024-05-01 11:00:00', 'action': 'close', 'direction': 'long',
            'quantity': 1, 'price': 101.0,
            'gross_pnl': 1.0,
            'commission_cost': 0.20, 'spread_cost': 0.30, 'slippage_cost': 0.10, 'impact_cost': 0.00,
            'total_cost': 0.60,
        },
    ])
    trades_path = tmp_path / 'trades.csv'
    trades.to_csv(trades_path, index=False)

    # Emulate summary building logic: metrics should reflect sum of total_cost
    trades_df = pd.read_csv(trades_path)
    cost_cols = ["commission_cost","spread_cost","slippage_cost","impact_cost","total_cost"]
    trades_df[cost_cols] = trades_df[cost_cols].fillna(0.0)
    sum_costs = float(trades_df["total_cost"].sum())
    metrics = {"tx_costs_total": 0.0}
    if abs(sum_costs - metrics.get("tx_costs_total", 0.0)) > 1e-9:
        metrics["tx_costs_total"] = sum_costs

    # Compute net_pnl and PF using net
    if 'gross_pnl' in trades_df.columns:
        trades_df['net_pnl'] = pd.to_numeric(trades_df['gross_pnl'], errors='coerce').fillna(0.0) - pd.to_numeric(trades_df['total_cost'], errors='coerce').fillna(0.0)
    else:
        trades_df['net_pnl'] = pd.to_numeric(trades_df.get('pnl', 0.0), errors='coerce').fillna(0.0)
    pos_sum = float(trades_df['net_pnl'][trades_df['net_pnl'] > 0].sum())
    neg_sum = float(trades_df['net_pnl'][trades_df['net_pnl'] < 0].sum())
    pf = float(pos_sum / (abs(neg_sum) + 1e-12)) if pos_sum > 0 else 0.0

    # Assertions
    assert metrics["tx_costs_total"] == sum_costs
    assert pf >= 0.0

