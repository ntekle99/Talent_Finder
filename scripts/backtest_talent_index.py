#!/usr/bin/env python3
"""Backtest the PERSON-LEVEL talent signals against forward stock returns.

Signal at year Y (research elite-inflow + wage-premium, z-scored cross-sectionally) is used
to predict each company's return in year Y+1. Long the top third, short the bottom third,
annually. Reports IC and long/short return per year + summary.

Honest limits: ~40 names x ~8 years = tiny sample (low power). OpenAlex join-years carry mild
look-ahead (inferred from later affiliations); the wage signal is genuinely point-in-time, so
we run wage-only as the clean check.
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path

P = Path(__file__).resolve().parents[1] / "data" / "panel"


def load_annual_returns():
    monthly = defaultdict(dict)
    with open(P / "returns.csv") as f:
        for r in csv.DictReader(f):
            y = int(r["month"][:4])
            monthly[(r["company_id"], y)].setdefault("rets", []).append(float(r["ret"]))
    annual = {}
    for (c, y), d in monthly.items():
        g = 1.0
        for x in d["rets"]:
            g *= (1 + x)
        if len(d["rets"]) >= 10:            # need a near-full year
            annual[(c, y)] = g - 1
    return annual


def load_signal(fname, col):
    out = {}
    with open(P / fname) as f:
        for r in csv.DictReader(f):
            out[(r["company_id"], int(r["year"]))] = float(r[col])
    return out


def zscore(d):
    if len(d) < 3:
        return {k: 0.0 for k in d}
    m = sum(d.values()) / len(d)
    sd = (sum((v - m) ** 2 for v in d.values()) / len(d)) ** 0.5 or 1.0
    return {k: (v - m) / sd for k, v in d.items()}


def spearman(pairs):
    if len(pairs) < 4:
        return 0.0
    xs = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
    ys = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
    rx = {i: r for r, i in enumerate(xs)}
    ry = {i: r for r, i in enumerate(ys)}
    n = len(pairs)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))


def run(name, signal_years):
    ann = load_annual_returns()
    years = sorted({y for (_, y) in signal_years})
    ls_returns, ics, rows = [], [], []
    for Y in years:
        sig = {c: v for (c, y), v in signal_years.items() if y == Y}
        z = zscore(sig)
        cross = []
        for c, s in z.items():
            fr = ann.get((c, Y + 1))
            if fr is not None:
                cross.append((c, s, fr))
        if len(cross) < 9:
            continue
        cross.sort(key=lambda x: x[1])
        k = max(1, len(cross) // 3)
        bottom = cross[:k]
        top = cross[-k:]
        ls = sum(x[2] for x in top) / k - sum(x[2] for x in bottom) / k
        ic = spearman([(s, fr) for _, s, fr in cross])
        ls_returns.append(ls); ics.append(ic)
        rows.append((Y, len(cross), ls, ic, top[-1][0], bottom[0][0]))
    print(f"\n===== {name} =====")
    print(f"  {'yr->ret':8s} {'n':>3} {'L/S':>8} {'IC':>6}  best-ranked / worst-ranked")
    for Y, n, ls, ic, tname, bname in rows:
        print(f"  {Y}->{Y+1}  {n:>3} {ls:+8.1%} {ic:+6.2f}  {tname} / {bname}")
    if ls_returns:
        mean = sum(ls_returns) / len(ls_returns)
        sd = (sum((r - mean) ** 2 for r in ls_returns) / len(ls_returns)) ** 0.5 or 1e-9
        print(f"  --- mean L/S {mean:+.1%}/yr | Sharpe {mean/sd:+.2f} | "
              f"mean IC {sum(ics)/len(ics):+.3f} | hit {sum(1 for r in ls_returns if r>0)}/{len(ls_returns)}")


def main():
    research = load_signal("talent_openalex.csv", "elite_inflow")
    wage = load_signal("talent_wage.csv", "wage_ratio")
    # combined = z(research) + z(wage) per year
    combined = {}
    years = {y for (_, y) in research} | {y for (_, y) in wage}
    for Y in years:
        zr = zscore({c: v for (c, y), v in research.items() if y == Y})
        zw = zscore({c: v for (c, y), v in wage.items() if y == Y})
        for c in set(zr) | set(zw):
            parts = [z[c] for z in (zr, zw) if c in z]
            combined[(c, Y)] = sum(parts) / len(parts)

    run("COMBINED (research + wage)", combined)
    run("WAGE-ONLY (clean point-in-time)", wage)
    run("RESEARCH-ONLY (mild look-ahead)", research)


if __name__ == "__main__":
    main()
