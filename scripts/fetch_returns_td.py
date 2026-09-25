#!/usr/bin/env python3
"""Fetch monthly returns from Twelve Data (free key) -> returns.csv. Stdlib only.

Runs on the Brev box (cloud network reaches Twelve Data; the corporate firewall does not).
Free tier = 8 requests/min, so we pace ~8s between symbols. One call per ticker.

Usage:  TD_KEY=xxxx python3 fetch_returns_td.py [output_csv]
"""
import csv, json, os, sys, time, urllib.request, urllib.parse

KEY = os.environ.get("TD_KEY", "")
OUT = sys.argv[1] if len(sys.argv) > 1 else "returns.csv"

# (company_id, twelvedata_symbol). Block uses legacy SQ for longer history.
MAP = [
    ("nvidia","NVDA"),("apple","AAPL"),("microsoft","MSFT"),("alphabet","GOOGL"),
    ("meta","META"),("amazon","AMZN"),("tesla","TSLA"),("netflix","NFLX"),("amd","AMD"),
    ("intel","INTC"),("oracle","ORCL"),("salesforce","CRM"),("adobe","ADBE"),("cisco","CSCO"),
    ("ibm","IBM"),("qualcomm","QCOM"),("broadcom","AVGO"),("texas_instruments","TXN"),
    ("micron","MU"),("paypal","PYPL"),("uber","UBER"),("lyft","LYFT"),("airbnb","ABNB"),
    ("snowflake","SNOW"),("palantir","PLTR"),("servicenow","NOW"),("workday","WDAY"),
    ("datadog","DDOG"),("crowdstrike","CRWD"),("zscaler","ZS"),("okta","OKTA"),("mongodb","MDB"),
    ("twilio","TWLO"),("snap","SNAP"),("pinterest","PINS"),("roku","ROKU"),("block","SQ"),
    ("shopify","SHOP"),("atlassian","TEAM"),("intuit","INTU"),("vmware","VMW"),("dell","DELL"),
    ("hp","HPQ"),("cloudflare","NET"),("unity","U"),("roblox","RBLX"),("doordash","DASH"),
    ("coinbase","COIN"),("ebay","EBAY"),("applied_materials","AMAT"),
]


def fetch(symbol):
    q = urllib.parse.urlencode({"symbol": symbol, "interval": "1month",
                                "outputsize": "5000", "apikey": KEY})
    url = f"https://api.twelvedata.com/time_series?{q}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.load(r)
        except Exception as e:
            print(f"     retry ({e})", flush=True); time.sleep(10); continue
        if isinstance(data, dict) and data.get("status") == "error":
            msg = data.get("message", "")
            if "run out of API credits" in msg or data.get("code") == 429:
                print("     rate limited, waiting 60s", flush=True); time.sleep(60); continue
            return None, msg
        vals = data.get("values") if isinstance(data, dict) else None
        if not vals:
            return None, "no values"
        # values are newest-first; sort ascending, then compute returns
        rows = sorted(((v["datetime"][:7], float(v["close"])) for v in vals), key=lambda x: x[0])
        out, prev = [], None
        for month, close in rows:
            if prev is not None and prev > 0:
                out.append((month, round(close / prev - 1, 5)))
            prev = close
        return out, None
    return None, "failed after retries"


def main():
    if not KEY:
        sys.exit("set TD_KEY env var to your Twelve Data API key")
    allrows, miss = [], []
    for cid, sym in MAP:
        data, err = fetch(sym)
        if data:
            allrows += [(cid, m, r) for m, r in data]
            print(f"ok   {cid:20s} {sym:6s} {len(data)}mo {data[0][0]}..{data[-1][0]}", flush=True)
        else:
            miss.append(f"{cid}({sym}:{err})")
            print(f"MISS {cid:20s} {sym:6s} {err}", flush=True)
        time.sleep(8)  # free tier: 8 req/min
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["company_id", "month", "ret"]); w.writerows(sorted(allrows))
    print(f"\nwrote {len(allrows)} rows -> {OUT}")
    if miss:
        print("missing:", ", ".join(miss))


if __name__ == "__main__":
    main()
