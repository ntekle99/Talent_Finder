#!/usr/bin/env python3
"""Fetch monthly returns for the broad ticker universe from Nasdaq (no key) -> returns_broad.csv.

Reads tickers from broad_tickers.txt (one per line). Runs on the Brev box (Nasdaq reachable there;
corp firewall blocks it on the laptop). Daily history -> monthly closes -> monthly returns.

Usage:  python3 fetch_returns_broad.py tickers.txt out.csv
"""
import csv, json, sys, time, urllib.request, urllib.parse
from datetime import date

TICKERS_FILE = sys.argv[1] if len(sys.argv) > 1 else "broad_tickers.txt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "returns_broad.csv"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/125.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nasdaq.com", "Referer": "https://www.nasdaq.com/",
}


def fetch(sym):
    q = urllib.parse.urlencode({"assetclass": "stocks", "fromdate": "2008-01-01",
                                "limit": "9999", "todate": date.today().isoformat()})
    url = f"https://api.nasdaq.com/api/quote/{sym}/historical?{q}"
    req = urllib.request.Request(url, headers=HEADERS)
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                j = json.load(r)
            break
        except Exception:
            time.sleep(4)
    else:
        return None
    rows = (((j or {}).get("data") or {}).get("tradesTable") or {}).get("rows")
    if not rows:
        return None
    monthly = {}
    for row in rows:
        try:
            mm, dd, yy = row["date"].split("/")
            month = f"{yy}-{int(mm):02d}"
            close = float(row["close"].replace("$", "").replace(",", ""))
        except (KeyError, ValueError):
            continue
        o = int(yy) * 10000 + int(mm) * 100 + int(dd)
        if month not in monthly or o > monthly[month][0]:
            monthly[month] = (o, close)
    series = sorted((m, c) for m, (_, c) in monthly.items())
    out, prev = [], None
    for month, close in series:
        if prev is not None and prev > 0:
            out.append((month, round(close / prev - 1, 5)))
        prev = close
    return out


def main():
    tickers = [t.strip() for t in open(TICKERS_FILE) if t.strip()]
    allrows, ok, miss = [], 0, 0
    for i, t in enumerate(tickers):
        data = fetch(t)
        if data:
            allrows += [(t, m, r) for m, r in data]; ok += 1
        else:
            miss += 1
        if i % 25 == 0:
            print(f"  {i}/{len(tickers)} ok={ok} miss={miss}", flush=True)
        time.sleep(0.8)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["company_id", "month", "ret"]); w.writerows(sorted(allrows))
    print(f"\nwrote {len(allrows)} rows for {ok}/{len(tickers)} tickers -> {OUT}")


if __name__ == "__main__":
    main()
