#!/usr/bin/env python3
"""Fuse the three talent signals into ONE talent index -> data/panel/talent_index.csv.

Signals (each z-scored across companies so they're comparable despite different units):
  research  = OpenAlex elite-researcher inflow, recent (2022+)      [caliber, has trajectory]
  wage      = H1B pay-premium vs national rate for same roles       [comp,    has trajectory]
  github    = GitHub OSS impact (log stars across top repos)        [eng infl, snapshot only]

talent_level      = equal-weighted mean of available signal z-scores (who has talent density now)
talent_trajectory = mean of available trajectory z-scores (research + wage; github has none)

Weights are tunable at the top. Companies missing a signal are scored on what's available.
"""
from __future__ import annotations
import csv, math, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from company_map import ticker_of
PANEL = ROOT / "data" / "panel"
OUT = PANEL / "talent_index.csv"

WEIGHTS = {"research": 1.0, "wage": 1.0, "github": 1.0}  # tune here


def zscore(d: dict[str, float]) -> dict[str, float]:
    vals = list(d.values())
    if len(vals) < 2:
        return {k: 0.0 for k in d}
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5 or 1.0
    return {k: (v - m) / sd for k, v in d.items()}


def load_research():
    prior, recent = defaultdict(float), defaultdict(float)
    with open(PANEL / "talent_openalex.csv") as f:
        for r in csv.DictReader(f):
            y, v = int(r["year"]), float(r["elite_inflow"])
            if 2018 <= y <= 2021: prior[r["company_id"]] += v
            elif 2022 <= y <= 2025: recent[r["company_id"]] += v
    level = dict(recent)
    traj = {c: recent[c] - prior.get(c, 0) for c in recent}
    return level, traj


def load_wage():
    pr, rc = defaultdict(lambda: [0.0, 0]), defaultdict(lambda: [0.0, 0])
    with open(PANEL / "talent_wage.csv") as f:
        for r in csv.DictReader(f):
            y, ratio, h = int(r["year"]), float(r["wage_ratio"]), int(r["hires"])
            if 2016 <= y <= 2020: pr[r["company_id"]][0] += ratio*h; pr[r["company_id"]][1] += h
            elif 2022 <= y <= 2025: rc[r["company_id"]][0] += ratio*h; rc[r["company_id"]][1] += h
    level = {c: rc[c][0]/rc[c][1] for c in rc if rc[c][1] > 0}
    traj = {c: level[c] - (pr[c][0]/pr[c][1]) for c in level if pr.get(c) and pr[c][1] > 0}
    return level, traj


def load_github():
    level = {}
    with open(PANEL / "talent_github.csv") as f:
        for r in csv.DictReader(f):
            level[r["company_id"]] = math.log10(int(r["oss_impact_stars"]) + 1)
    return level, {}


def main():
    r_lvl, r_tr = load_research()
    w_lvl, w_tr = load_wage()
    g_lvl, _ = load_github()
    zr, zw, zg = zscore(r_lvl), zscore(w_lvl), zscore(g_lvl)
    ztr_r, ztr_w = zscore(r_tr), zscore(w_tr)

    companies = set(zr) | set(zw) | set(zg)
    tk = ticker_of()
    rows = []
    for c in companies:
        parts, wsum = 0.0, 0.0
        for key, z in (("research", zr), ("wage", zw), ("github", zg)):
            if c in z:
                parts += WEIGHTS[key] * z[c]; wsum += WEIGHTS[key]
        level = parts / wsum if wsum else 0.0
        tparts, twsum = 0.0, 0.0
        for z in (ztr_r, ztr_w):
            if c in z:
                tparts += z[c]; twsum += 1
        traj = tparts / twsum if twsum else 0.0
        present = "".join(s for s, z in (("R", zr), ("W", zw), ("G", zg)) if c in z)
        rows.append((c, tk.get(c, ""), round(level, 3), round(traj, 3), present))

    rows.sort(key=lambda x: -x[2])
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["company_id", "ticker", "talent_level", "talent_trajectory", "signals"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT}\n")

    print("=== TALENT DENSITY (fused level: research + wage + github, z-scored) ===")
    print(f"  {'#':>2} {'company':18s} {'tkr':5s} {'level':>6} {'traj':>6}  signals")
    for i, r in enumerate(rows[:20], 1):
        print(f"  {i:>2} {r[0]:18s} {r[1]:5s} {r[2]:+6.2f} {r[3]:+6.2f}  {r[4]}")

    print("\n=== RISING TALENT TRAJECTORY (research + wage momentum) ===")
    for r in sorted(rows, key=lambda x: -x[3])[:12]:
        if r[3] != 0:
            print(f"  {r[0]:18s} {r[1]:5s} traj={r[3]:+.2f}  level={r[2]:+.2f}")


if __name__ == "__main__":
    main()
