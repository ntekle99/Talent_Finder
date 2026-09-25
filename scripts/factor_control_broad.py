#!/usr/bin/env python3
"""Broad-universe factor-control backtest of the wage-premium talent signal (~349 public firms).

Same Fama-MacBeth framework as factor_control.py, but on the full SEC-matched universe instead
of 50 names -- the statistical-power test. Each year: cross-sectionally regress next-year return
on the talent signal, controlling for momentum (growth factor) and sector. Report the average
talent coefficient + t-stat, plus decile L/S and sector-neutral L/S.
"""
from __future__ import annotations
import csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parents[1] / "data" / "panel"


def annual_returns(fname):
    m = defaultdict(list)
    with open(P / fname) as f:
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


def load_wage(fname):
    out = {}
    with open(P / fname) as f:
        for r in csv.DictReader(f):
            out[(r["ticker"], int(r["year"]))] = (float(r["wage_ratio"]), int(r["hires"]))
    return out


def load_sectors(fname):
    sec = {}
    path = P / fname
    if path.exists():
        with open(path) as f:
            for r in csv.DictReader(f):
                sec[r["ticker"]] = r.get("sector") or "Unknown"
    return sec


def z(a):
    a = np.asarray(a, float); s = a.std()
    return (a - a.mean()) / s if s > 1e-9 else a * 0


def main():
    TECH_ONLY = True   # restrict to tech-sector companies (SIC-classified)
    ann = annual_returns("returns_broad.csv")
    wage = load_wage("talent_wage_broad.csv")
    sec = load_sectors("sectors.csv")
    if TECH_ONLY:
        tech = {t for t, s in sec.items() if s != "Non-Tech"}
        wage = {(t, y): v for (t, y), v in wage.items() if t in tech}
        print(f"[tech-only] restricted to {len(tech)} tech tickers\n")
    years = sorted({y for (_, y) in wage})
    sectors = sorted({s for s in sec.values() if s != "Non-Tech"}) if sec else []

    A, B, C = [], [], []
    ls_dec, ls_secn, ics, ns = [], [], [], []
    for Y in years:
        rec = []
        for (t, y), (w, h) in wage.items():
            if y != Y:
                continue
            mom = ann.get((t, Y)); fwd = ann.get((t, Y + 1))
            if mom is None or fwd is None:
                continue
            rec.append((t, w, mom, fwd))
        if len(rec) < 30:
            continue
        tal = z([r[1] for r in rec]); mom = z([r[2] for r in rec])
        fwd = np.array([r[3] for r in rec]); n = len(rec)
        A.append(np.linalg.lstsq(np.column_stack([np.ones(n), tal]), fwd, rcond=None)[0][1])
        B.append(np.linalg.lstsq(np.column_stack([np.ones(n), tal, mom]), fwd, rcond=None)[0][1])
        if sectors:
            secs = [sec.get(r[0], "Unknown") for r in rec]
            dums = np.column_stack([[1.0 if s == sk else 0.0 for s in secs] for sk in sectors[1:]])
            XC = np.column_stack([np.ones(n), tal, mom, dums])
            C.append(np.linalg.lstsq(XC, fwd, rcond=None)[0][1])
        # decile L/S (top 20% vs bottom 20%)
        order = sorted(rec, key=lambda x: x[1]); k = max(1, n // 5)
        ls_dec.append(np.mean([x[3] for x in order[-k:]]) - np.mean([x[3] for x in order[:k]]))
        # sector-neutral L/S
        if sectors:
            by = defaultdict(list)
            for t, w, mo, fw in rec:
                by[sec.get(t, "Unknown")].append((w, fw))
            resid = []
            for s, items in by.items():
                if len(items) < 3:
                    continue
                wm = np.mean([x[0] for x in items])
                resid += [(w - wm, fw) for w, fw in items]
            if len(resid) >= 15:
                resid.sort(key=lambda x: x[0]); kk = max(1, len(resid) // 5)
                ls_secn.append(np.mean([x[1] for x in resid[-kk:]]) - np.mean([x[1] for x in resid[:kk]]))
        # IC
        rr = fwd.argsort().argsort(); tr = tal.argsort().argsort()
        ics.append(np.corrcoef(rr, tr)[0, 1]); ns.append(n)

    def fm(v, label):
        v = np.array(v)
        t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 and v.std() > 0 else 0
        print(f"  {label:22s} {v.mean():>+9.3%}  t={t:>+5.2f}  ({len(v)} yrs)")

    print(f"universe: {len(set(t for t,_ in wage))} tickers, ~{int(np.mean(ns))} names/yr, {len(A)} test years\n")
    print("=== Fama-MacBeth talent coefficient (return per +1 sigma) ===")
    fm(A, "A talent only"); fm(B, "B +momentum")
    if C: fm(C, "C +momentum+sector")

    def summ(v, label):
        v = np.array(v)
        if len(v) == 0: return
        sh = v.mean() / (v.std() or 1e-9)
        print(f"  {label:26s} mean {v.mean():+.1%}/yr  Sharpe {sh:+.2f}  hit {int((v>0).sum())}/{len(v)}")
    print("\n=== portfolios ===")
    summ(ls_dec, "quintile L/S (raw)")
    if ls_secn: summ(ls_secn, "quintile L/S (sector-neutral)")
    print(f"\n  mean rank-IC {np.mean(ics):+.3f}  over {len(ics)} years")


if __name__ == "__main__":
    main()
