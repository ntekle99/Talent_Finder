#!/usr/bin/env python3
"""Standalone Stooq monthly-returns fetcher (stdlib + curl only, no pip deps).

Runs anywhere with curl + python3 -- notably on a Brev cloud box, to sidestep a
corporate firewall that blocks finance sites. Solves Stooq's SHA-256 proof-of-work
challenge once, then pulls monthly closes for each ticker and writes returns.csv.

Usage:  python3 fetch_returns_stooq.py [output_csv]
"""
import csv, io, re, sys, hashlib, subprocess, tempfile, os

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")
COOKIES = os.path.join(tempfile.gettempdir(), "stooq_cookies.txt")

# (company_id, stooq_symbol) -- Block trades as SQ, Unity as 'u'.
MAP = [
    ("nvidia","nvda"),("apple","aapl"),("microsoft","msft"),("alphabet","googl"),
    ("meta","meta"),("amazon","amzn"),("tesla","tsla"),("netflix","nflx"),("amd","amd"),
    ("intel","intc"),("oracle","orcl"),("salesforce","crm"),("adobe","adbe"),("cisco","csco"),
    ("ibm","ibm"),("qualcomm","qcom"),("broadcom","avgo"),("texas_instruments","txn"),
    ("micron","mu"),("paypal","pypl"),("uber","uber"),("lyft","lyft"),("airbnb","abnb"),
    ("snowflake","snow"),("palantir","pltr"),("servicenow","now"),("workday","wday"),
    ("datadog","ddog"),("crowdstrike","crwd"),("zscaler","zs"),("okta","okta"),("mongodb","mdb"),
    ("twilio","twlo"),("snap","snap"),("pinterest","pins"),("roku","roku"),("block","sq"),
    ("shopify","shop"),("atlassian","team"),("intuit","intu"),("vmware","vmw"),("dell","dell"),
    ("hp","hpq"),("cloudflare","net"),("unity","u"),("roblox","rblx"),("doordash","dash"),
    ("coinbase","coin"),("ebay","ebay"),("applied_materials","amat"),
]


def curl(url, post=None):
    cmd = ["curl", "-sS", "--max-time", "30", "-A", UA, "-c", COOKIES, "-b", COOKIES]
    if post:
        for k, v in post.items():
            cmd += ["--data-urlencode", f"{k}={v}"]
    cmd.append(url)
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=40).stdout
    except Exception:
        return ""


def pow_n(html):
    mc, md = re.search(r'c="([^"]+)"', html), re.search(r'd=(\d+)', html)
    if not mc or not md:
        return None
    c, d = mc.group(1), int(md.group(1))
    target, n = "0" * d, 0
    while not hashlib.sha256(f"{c}{n}".encode()).hexdigest().startswith(target):
        n += 1
    return c, n


def fetch(sym):
    url = f"https://stooq.com/q/d/l/?s={sym}.us&i=m"
    text = curl(url)
    if "requires JavaScript" in text or "__verify" in text:
        pw = pow_n(text)
        if pw:
            curl("https://stooq.com/__verify", {"c": pw[0], "n": pw[1]})
            text = curl(url)
    if not text or text.startswith("<") or "Date" not in text[:20]:
        return []
    out, prev = [], None
    for row in csv.DictReader(io.StringIO(text)):
        try:
            close, month = float(row["Close"]), row["Date"][:7]
        except (ValueError, KeyError):
            continue
        if prev is not None and prev > 0:
            out.append((month, round(close / prev - 1, 5)))
        prev = close
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "returns.csv"
    rows, miss = [], []
    for cid, sym in MAP:
        data = fetch(sym)
        if data:
            rows += [(cid, m, r) for m, r in data]
            print(f"ok   {cid:20s} {sym:6s} {len(data)}mo {data[0][0]}..{data[-1][0]}", flush=True)
        else:
            miss.append(sym)
            print(f"MISS {cid:20s} {sym}", flush=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["company_id", "month", "ret"])
        w.writerows(sorted(rows))
    print(f"\nwrote {len(rows)} rows -> {out_path}")
    if miss:
        print("missing:", ", ".join(miss))


if __name__ == "__main__":
    main()
