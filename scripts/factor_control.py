#!/usr/bin/env python3
"""Factor-control test (Fama-MacBeth): does the wage-premium talent signal predict forward
returns AFTER controlling for the growth/momentum factor and sector?

Each year we cross-sectionally regress next-year return on:
  A) talent only
  B) talent + trailing-momentum   (momentum ~ the growth factor: growth names ran up)
  C) talent + momentum + sector dummies
and collect the TALENT coefficient. Average across years (Fama-MacBeth) with a t-stat.
If the talent coefficient stays positive/stable A->C, there's alpha beyond the factors.
If it shrinks to ~0, the earlier result was just a growth/sector tilt.

Also reports a SECTOR-NEUTRAL long/short (rank within sector) for intuition.
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from company_map import COMPANIES
P = ROOT / "data" / "panel"
SECTOR = {c.company_id: c.sector for c in COMPANIES}


def annual_returns():
    m = defaultdict(list)
    with open(P / "returns.csv") as f:
        for r in csv.DictReader(f):
            m[(r["company_id"], int(r["month"][:4]))].append(float(r["ret"]))
    out = {}
    for (c, y), rets in m.items():
        if len(rets) >= 10:
            g = 1.0
            for x in rets:
                g *= (1 + x)
            out[(c, y)] = g - 1
    return out


def load_wage():
    out = {}
    with open(P / "talent_wage.csv") as f:
        for r in csv.DictReader(f):
            out[(r["company_id"], int(r["year"]))] = float(r["wage_ratio"])
    return out


def z(a):
    a = np.asarray(a, float)
    s = a.std()
    return (a - a.mean()) / s if s > 1e-9 else a * 0


def fama_macbeth(ann, wage):
    years = sorted({y for (_, y) in wage})
    sectors = sorted(set(SECTOR.values()))
    coefs = {"A talent": [], "B +momentum": [], "C +mom+sector": []}
    for Y in years:
        rec = []
        for (c, y), w in wage.items():
            if y != Y:
                continue
            mom = ann.get((c, Y)); fwd = ann.get((c, Y + 1))
            if mom is None or fwd is None:
                continue
            rec.append((c, w, mom, fwd))
        if len(rec) < 12:
            continue
        tal = z([r[1] for r in rec])
        mom = z([r[2] for r in rec])
        fwd = np.array([r[3] for r in rec])
        n = len(rec)
        # A: fwd ~ 1 + talent
        XA = np.column_stack([np.ones(n), tal])
        coefs["A talent"].append(np.linalg.lstsq(XA, fwd, rcond=None)[0][1])
        # B: + momentum
        XB = np.column_stack([np.ones(n), tal, mom])
        coefs["B +momentum"].append(np.linalg.lstsq(XB, fwd, rcond=None)[0][1])
        # C: + sector dummies (drop one baseline)
        secs = [SECTOR[r[0]] for r in rec]
        dums = np.column_stack([[1.0 if s == sec else 0.0 for s in secs]
                                for sec in sectors[1:]])
        XC = np.column_stack([np.ones(n), tal, mom, dums])
        coefs["C +mom+sector"].append(np.linalg.lstsq(XC, fwd, rcond=None)[0][1])
    print("=== Fama-MacBeth: average TALENT coefficient (return per +1 sigma of signal) ===")
    print(f"  {'model':16s} {'mean coef':>10} {'t-stat':>8} {'years':>6}")
    for k, v in coefs.items():
        v = np.array(v)
        t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 and v.std() > 0 else 0
        print(f"  {k:16s} {v.mean():>+10.3%} {t:>+8.2f} {len(v):>6}")
    print("  (coef = extra annual return per 1-sigma of talent; t>2 ~ significant. "
          "If A->C shrinks to ~0, it was the growth/sector factor.)")


def sector_neutral_ls(ann, wage):
    years = sorted({y for (_, y) in wage})
    rets = []
    for Y in years:
        by_sec = defaultdict(list)
        for (c, y), w in wage.items():
            if y != Y:
                continue
            fwd = ann.get((c, Y + 1))
            if fwd is not None:
                by_sec[SECTOR[c]].append((c, w, fwd))
        # within-sector demeaned signal, then global top/bottom third of the residual
        resid = []
        for sec, items in by_sec.items():
            if len(items) < 3:
                continue
            wm = np.mean([x[1] for x in items])
            for c, w, fwd in items:
                resid.append((w - wm, fwd))
        if len(resid) < 9:
            continue
        resid.sort(key=lambda x: x[0])
        k = max(1, len(resid) // 3)
        ls = np.mean([x[1] for x in resid[-k:]]) - np.mean([x[1] for x in resid[:k]])
        rets.append(ls)
    if rets:
        rets = np.array(rets)
        sh = rets.mean() / (rets.std() or 1e-9)
        print(f"\n=== SECTOR-NEUTRAL long/short (rank within sector) ===")
        print(f"  mean {rets.mean():+.1%}/yr | Sharpe {sh:+.2f} | hit {int((rets>0).sum())}/{len(rets)}")
        print("  (removes the 'short Intel/semis, long software' sector tilt)")


def main():
    ann = annual_returns()
    wage = load_wage()
    fama_macbeth(ann, wage)
    sector_neutral_ls(ann, wage)


if __name__ == "__main__":
    main()
