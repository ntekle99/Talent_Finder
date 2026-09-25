#!/usr/bin/env python3
"""Fetch FULL-history monthly returns from Yahoo's chart API (no key) -> returns_long.csv.

Yahoo's v8 chart API returns ~20yr of split/dividend-adjusted monthly closes. Reachable from
the Brev box (the earlier 429 was a temporary rate-limit that reset). Paced with backoff so we
don't re-trip it. Uses adjusted close for total return.

Usage:  python3 fetch_returns_yahoo.py tickers.txt out.csv
"""
import csv, json, sys, time, urllib.request, datetime

TICKERS = sys.argv[1] if len(sys.argv) > 1 else "tech_tickers.txt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "returns_long.csv"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/125.0 Safari/537.36"}


def fetch(sym):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1mo&range=25y"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25) as r:
                d = json.load(r)
            res = (d.get("chart") or {}).get("result")
            if not res:
                return None
            res = res[0]
            ts = res.get("timestamp") or []
            q = res["indicators"]["quote"][0]
            adj = (res["indicators"].get("adjclose") or [{}])[0].get("adjclose")
            closes = adj if adj else q.get("close")
            out, prev = [], None
            for t, c in zip(ts, closes):
                if c is None:
                    continue
                month = datetime.date.fromtimestamp(t).strftime("%Y-%m")
                if prev is not None and prev > 0:
                    out.append((month, round(c / prev - 1, 5)))
                prev = c
            return out
        except Exception as e:
            code = getattr(e, "code", None)
            time.sleep(30 if code == 429 else 5)   # back off hard on rate limit
    return None


def main():
    tickers = [t.strip() for t in open(TICKERS) if t.strip()]
    allrows, ok, miss = [], 0, 0
    for i, t in enumerate(tickers):
        data = fetch(t)
        if data:
            allrows += [(t, m, r) for m, r in data]; ok += 1
        else:
            miss += 1
        if i % 20 == 0:
            print(f"  {i}/{len(tickers)} ok={ok} miss={miss}", flush=True)
        time.sleep(1.2)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["company_id", "month", "ret"]); w.writerows(sorted(allrows))
    print(f"\nwrote {len(allrows)} rows for {ok}/{len(tickers)} tickers -> {OUT}")


if __name__ == "__main__":
    main()
