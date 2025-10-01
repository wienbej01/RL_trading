#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize RL runs from results/_registry/runs.csv")
    ap.add_argument('--registry', type=str, default='results/_registry/runs.csv', help='Path to registry CSV')
    ap.add_argument('--sort', type=str, default='timestamp', help='Column to sort by')
    ap.add_argument('--desc', action='store_true', help='Sort descending')
    ap.add_argument('--limit', type=int, default=50, help='Max rows to display')
    args = ap.parse_args()

    p = Path(args.registry)
    if not p.exists():
        print(f"Registry not found: {p}")
        return 1

    with p.open('r', newline='') as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    if not rows:
        print("No runs recorded yet.")
        return 0

    # Safe sort
    key = args.sort
    def _key(r):
        v = r.get(key)
        try:
            return float(v)
        except Exception:
            return str(v)
    rows.sort(key=_key, reverse=bool(args.desc))
    rows = rows[: max(1, int(args.limit))]

    # Select columns to show
    cols: List[str] = [
        'timestamp','run_name','variant','seed','timesteps',
        'sharpe','pf','ret','maxDD','trades','long','short','flips','tx_costs_total'
    ]
    cols = [c for c in cols if c in rows[0].keys()]

    # Compute column widths
    widths = {c: max(len(c), max(len(str(r.get(c,''))) for r in rows)) for c in cols}

    # Print header
    header = ' | '.join(str(c).ljust(widths[c]) for c in cols)
    print(header)
    print('-' * len(header))
    # Print rows
    for r in rows:
        line = ' | '.join(str(r.get(c, '')).ljust(widths[c]) for c in cols)
        print(line)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

