#!/usr/bin/env python3
"""Fetch SEC SIC industry codes for the broad ticker universe -> data/panel/sectors.csv.

SEC is reachable from the laptop (unlike finance sites). SIC codes give a clean, official
industry classification. We keep tech industries (semis, computers, software, IT services,
electronics, comm equipment) and label everything else 'Non-Tech' so the backtest can filter.
"""
from __future__ import annotations
import json, csv, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "panel"
H = {"User-Agent": "talent-finder research ntekle@nvidia.com"}


def subsector(sic: str) -> str:
    s = (sic or "").zfill(4)
    p3 = s[:3]
    if s in {"3674", "3559", "3671", "3672", "3673", "3675", "3676", "3677", "3678", "3679"} or p3 == "367":
        return "Semiconductors"
    if p3 == "357":                                   # computers & office equipment
        return "Hardware"
    if s == "7372":
        return "Software"
    if p3 == "737":                                   # other computer services
        return "IT Services"
    if p3 == "366" or s in {"3576", "4899", "4813"}:  # comm equipment / services
        return "Comm/Internet"
    if s in {"3812", "3827", "3826", "3829"}:
        return "Instruments"
    return "Non-Tech"


def main():
    sec = json.loads((P.parent / "sec_tickers.json").read_text())
    cik = {r["ticker"].upper(): r["cik_str"] for r in sec.values()}
    tickers = [t.strip() for t in (P / "broad_tickers.txt").read_text().splitlines() if t.strip()]
    rows, tech = [], 0
    for i, t in enumerate(tickers):
        c = cik.get(t.upper())
        if not c:
            rows.append((t, "Non-Tech", "", "")); continue
        try:
            url = f"https://data.sec.gov/submissions/CIK{c:010d}.json"
            d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=20))
            sic = str(d.get("sic", "")); desc = d.get("sicDescription", "")
        except Exception:
            sic, desc = "", ""
        sub = subsector(sic)
        if sub != "Non-Tech":
            tech += 1
        rows.append((t, sub, sic, desc))
        if i % 50 == 0:
            print(f"  {i}/{len(tickers)} tech={tech}", flush=True)
        time.sleep(0.11)  # SEC ~10 req/s
    with (P / "sectors.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["ticker", "sector", "sic", "sic_desc"]); w.writerows(rows)
    print(f"\nwrote {len(rows)} tickers, {tech} tech -> {P/'sectors.csv'}")


if __name__ == "__main__":
    main()
