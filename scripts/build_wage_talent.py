#!/usr/bin/env python3
"""Wage-percentile talent signal from H1B data we already have -> data/panel/talent_wage.csv.

Idea: talent shows up in PRICE. For each certified filing we annualize the offered wage and
bucket it by role (SOC code) and year. We build the NATIONAL average wage per (role, year)
across ALL employers, then ask, per company-year:

    wage_ratio = (total wages the company offered)
               / (what those same hires would cost at the national average for their roles)

wage_ratio > 1.0  =>  the company pays above market for the exact roles it hires = buying
premium talent. This is role-mix-adjusted (a company loading up on high-paid ML roles isn't
automatically "premium" -- it's compared to the national rate for ML roles specifically).

Covers ANY visa sponsor (incl. non-publishing firms OpenAlex can't see: Palantir, etc.),
with full history. Single streaming pass, bounded memory.
"""
from __future__ import annotations
import sys, csv, os, time
from collections import defaultdict
from datetime import datetime, date
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import openpyxl
from talent_finder.ingest.h1b import _pick, _to_float
from company_map import Matcher, COMPANIES, ticker_of

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "panel" / "talent_wage.csv"
SKIP = {"H-1B_Case_Data_FY2008.xlsx", "H-1B_Case_Data_FY2009.xlsx", "Icert_LCA_FY2009.xlsx"}

ALIASES = {
    "employer": ["EMPLOYER_NAME", "LCA_CASE_EMPLOYER_NAME"],
    "soc": ["SOC_CODE", "LCA_CASE_SOC_CODE"],
    "wage": ["WAGE_RATE_OF_PAY_FROM", "WAGE_RATE_OF_PAY_FROM_1", "LCA_CASE_WAGE_RATE_FROM",
             "WAGE_RATE_1"],
    "unit": ["WAGE_UNIT_OF_PAY", "WAGE_RATE_OF_PAY_UNIT", "LCA_CASE_WAGE_RATE_UNIT",
             "WAGE_UNIT_OF_PAY_1", "PW_UNIT_OF_PAY_1"],
    "status": ["CASE_STATUS", "STATUS", "APPROVAL_STATUS"],
    "date": ["DECISION_DATE", "DOL_DECISION_DATE", "RECEIVED_DATE", "SUBMITTED_DATE"],
}
_MULT = [("hour", 2080), ("bi", 26), ("week", 52), ("month", 12), ("year", 1), ("annual", 1)]


def annualize(wage: float, unit: str) -> float:
    u = (unit or "").lower()
    for key, mult in _MULT:
        if key in u:
            return wage * mult
    return wage  # assume annual if unit missing


def _year(v) -> int | None:
    if isinstance(v, (datetime, date)):
        return v.year
    if v is None:
        return None
    s = str(v)
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S", "%m/%d/%y"):
        try:
            return datetime.strptime(s.strip(), fmt).year
        except ValueError:
            continue
    return None


def process(path_str: str) -> dict:
    """Worker -> {'nat': {(soc,year):[n,sum]}, 'co': {(cid,soc,year):[n,sum]}}."""
    path = Path(path_str)
    matcher = Matcher(COMPANIES)
    nat = defaultdict(lambda: [0, 0.0])
    co = defaultdict(lambda: [0, 0.0])
    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb[wb.sheetnames[0]]
        rows = ws.iter_rows(values_only=True)
        header = [str(c).strip() if c is not None else "" for c in next(rows)]
        ci = {k: _pick(header, v) for k, v in ALIASES.items()}
        if ci["employer"] is None or ci["soc"] is None or ci["wage"] is None or ci["date"] is None:
            wb.close(); return {"nat": {}, "co": {}, "file": path.name, "note": "missing cols"}
        for row in rows:
            st = ci["status"]
            if st is not None and st < len(row):
                s = row[st]
                if s is None or "CERTIF" not in str(s).upper():
                    continue
            if ci["wage"] >= len(row) or ci["soc"] >= len(row) or ci["date"] >= len(row):
                continue
            wage = _to_float(str(row[ci["wage"]])) if row[ci["wage"]] is not None else None
            if not wage:
                continue
            unit = row[ci["unit"]] if ci["unit"] is not None and ci["unit"] < len(row) else "Year"
            wage = annualize(wage, str(unit))
            if not (20_000 < wage < 2_000_000):
                continue
            year = _year(row[ci["date"]])
            soc = str(row[ci["soc"]])[:7].strip() if row[ci["soc"]] else ""
            if year is None or not soc:
                continue
            key = (soc, year)
            nat[key][0] += 1; nat[key][1] += wage
            cid = matcher.match(str(row[ci["employer"]]) if row[ci["employer"]] is not None else "")
            if cid:
                ck = (cid, soc, year)
                co[ck][0] += 1; co[ck][1] += wage
        wb.close()
    except Exception as e:
        return {"nat": {}, "co": {}, "file": path.name, "note": f"ERR {e}"}
    return {"nat": {k: v for k, v in nat.items()}, "co": {k: v for k, v in co.items()},
            "file": path.name}


def main():
    files = sorted(str(p) for p in RAW.glob("*.xlsx") if p.name not in SKIP)
    print(f"streaming {len(files)} files for wage-percentile talent...\n", flush=True)
    nat = defaultdict(lambda: [0, 0.0])
    co = defaultdict(lambda: [0, 0.0])
    t0 = time.time()
    with Pool(min(8, os.cpu_count() or 4)) as pool:
        for res in pool.imap_unordered(process, files):
            for k, v in res["nat"].items():
                nat[k][0] += v[0]; nat[k][1] += v[1]
            for k, v in res["co"].items():
                co[k][0] += v[0]; co[k][1] += v[1]
            print(f"  {res['file']:42s} {res.get('note','ok')}", flush=True)

    nat_mean = {k: v[1] / v[0] for k, v in nat.items() if v[0] > 0}
    # per (company, year): actual wages vs expected-at-national-rate for same role mix
    agg = defaultdict(lambda: [0, 0.0, 0.0])  # [hires, actual_sum, expected_sum]
    for (cid, soc, year), (n, s) in co.items():
        ref = nat_mean.get((soc, year))
        if ref is None:
            continue
        cell = agg[(cid, year)]
        cell[0] += n; cell[1] += s; cell[2] += n * ref

    tk = ticker_of()
    rows = []
    for (cid, year), (hires, actual, expected) in sorted(agg.items()):
        if hires < 15 or expected <= 0:
            continue
        rows.append((cid, year, hires, round(actual / expected, 4)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["company_id", "year", "hires", "wage_ratio"])
        w.writerows(rows)
    print(f"\nwrote {len(rows)} company-year rows -> {OUT}  ({time.time()-t0:.0f}s)")

    # recent premium (2022+) ranking
    recent = defaultdict(lambda: [0.0, 0])
    for cid, year, hires, ratio in rows:
        if year >= 2022:
            recent[cid][0] += ratio * hires; recent[cid][1] += hires
    print("\n=== PAYS-FOR-TALENT (wage vs national rate for same roles, 2022+, hire-weighted) ===")
    ranked = sorted(((c, w / n) for c, (w, n) in recent.items() if n > 0), key=lambda x: -x[1])
    for cid, r in ranked[:25]:
        bar = "#" * int((r - 0.8) * 40) if r > 0.8 else ""
        print(f"  {cid:20s} {tk.get(cid,''):6s} {r:5.2f}x  {bar}")


if __name__ == "__main__":
    main()
