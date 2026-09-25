#!/usr/bin/env python3
"""Fetch monthly returns from Nasdaq's public API (no key) -> returns.csv. Stdlib only.

Runs on the Brev box (Nasdaq doesn't block cloud IPs, unlike Yahoo; and the box has no
corporate firewall). Nasdaq's /historical endpoint returns ~10 years of daily OHLC; we
keep each month's last close and compute monthly returns.

Usage:  python3 fetch_returns_nasdaq.py [output_csv]
"""
import csv, json, sys, time, urllib.request, urllib.parse
from datetime import date

OUT = sys.argv[1] if len(sys.argv) > 1 else "returns.csv"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/125.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}

# (company_id, [symbols to try in order]). Block/VMware use delisted tickers for history.
MAP = [
    ("nvidia",["NVDA"]),("apple",["AAPL"]),("microsoft",["MSFT"]),("alphabet",["GOOGL"]),
    ("meta",["META","FB"]),("amazon",["AMZN"]),("tesla",["TSLA"]),("netflix",["NFLX"]),
    ("amd",["AMD"]),("intel",["INTC"]),("oracle",["ORCL"]),("salesforce",["CRM"]),
    ("adobe",["ADBE"]),("cisco",["CSCO"]),("ibm",["IBM"]),("qualcomm",["QCOM"]),
    ("broadcom",["AVGO"]),("texas_instruments",["TXN"]),("micron",["MU"]),("paypal",["PYPL"]),
    ("uber",["UBER"]),("lyft",["LYFT"]),("airbnb",["ABNB"]),("snowflake",["SNOW"]),
    ("palantir",["PLTR"]),("servicenow",["NOW"]),("workday",["WDAY"]),("datadog",["DDOG"]),
    ("crowdstrike",["CRWD"]),("zscaler",["ZS"]),("okta",["OKTA"]),("mongodb",["MDB"]),
    ("twilio",["TWLO"]),("snap",["SNAP"]),("pinterest",["PINS"]),("roku",["ROKU"]),
    ("block",["XYZ","SQ"]),("shopify",["SHOP"]),("atlassian",["TEAM"]),("intuit",["INTU"]),
    ("vmware",["VMW"]),("dell",["DELL"]),("hp",["HPQ"]),("cloudflare",["NET"]),("unity",["U"]),
    ("roblox",["RBLX"]),("doordash",["DASH"]),("coinbase",["COIN"]),("ebay",["EBAY"]),
    ("applied_materials",["AMAT"]),
]


def fetch_symbol(sym):
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
            time.sleep(5)
    else:
        return None
    rows = (((j or {}).get("data") or {}).get("tradesTable") or {}).get("rows")
    if not rows:
        return None
    # collapse daily -> last close per month
    monthly = {}
    for row in rows:
        try:
            mm, dd, yy = row["date"].split("/")
            month = f"{yy}-{int(mm):02d}"
            close = float(row["close"].replace("$", "").replace(",", ""))
        except (KeyError, ValueError):
            continue
        key = (month, f"{yy}{mm}{dd}")
        monthly.setdefault(month, (row["date"], close))
        # keep the latest calendar day within the month
        prev_date = monthly[month][0]
        if _to_ord(row["date"]) > _to_ord(prev_date):
            monthly[month] = (row["date"], close)
    series = sorted((m, c) for m, (d, c) in monthly.items())
    out, prev = [], None
    for month, close in series:
        if prev is not None and prev > 0:
            out.append((month, round(close / prev - 1, 5)))
        prev = close
    return out


def _to_ord(mdy):
    mm, dd, yy = mdy.split("/")
    return int(yy) * 10000 + int(mm) * 100 + int(dd)


def main():
    allrows, miss = [], []
    for cid, syms in MAP:
        data = None
        for sym in syms:
            data = fetch_symbol(sym)
            if data:
                used = sym
                break
        if data:
            allrows += [(cid, m, r) for m, r in data]
            print(f"ok   {cid:20s} {used:6s} {len(data)}mo {data[0][0]}..{data[-1][0]}", flush=True)
        else:
            miss.append(cid)
            print(f"MISS {cid:20s} {'/'.join(syms)}", flush=True)
        time.sleep(1.5)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["company_id", "month", "ret"]); w.writerows(sorted(allrows))
    print(f"\nwrote {len(allrows)} rows -> {OUT}")
    if miss:
        print("missing:", ", ".join(miss))


if __name__ == "__main__":
    main()
