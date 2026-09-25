#!/usr/bin/env python3
"""Fetch monthly stock returns for the mapped tickers from Stooq (free, no key) ->
data/panel/returns.csv  (company_id, month, ret).

Stooq monthly CSV: https://stooq.com/q/d/l/?s=<sym>.us&i=m  (Date,Open,High,Low,Close,Volume).
Return_t = Close_t / Close_{t-1} - 1. Not dividend-adjusted; fine for a momentum backtest.
"""
from __future__ import annotations
import csv, io, sys, time, re, hashlib, subprocess, tempfile, os
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from company_map import COMPANIES

OUT = ROOT / "data" / "panel"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"}

# Stooq symbol overrides where the current ticker lacks full history.
STOOQ_SYM = {"XYZ": "sq", "U": "u"}  # Block traded as SQ; Unity is single-letter 'u'


def stooq_symbol(ticker: str) -> str:
    return STOOQ_SYM.get(ticker, ticker.lower()) + ".us"


def _solve_and_verify(session: requests.Session, html: str) -> bool:
    """Stooq gates data behind a SHA-256 proof-of-work: find n where
    SHA256(c+n) starts with d zeros, POST to /__verify to earn a session cookie."""
    mc = re.search(r'c="([^"]+)"', html)
    md = re.search(r'd=(\d+)', html)
    if not mc or not md:
        return False
    c, d = mc.group(1), int(md.group(1))
    target = "0" * d
    n = 0
    while True:
        if hashlib.sha256(f"{c}{n}".encode()).hexdigest().startswith(target):
            break
        n += 1
    r = session.post("https://stooq.com/__verify", headers=UA,
                     data={"c": c, "n": n}, timeout=30)
    return r.ok


def _pow_n(html: str) -> tuple[str, int] | None:
    mc = re.search(r'c="([^"]+)"', html)
    md = re.search(r'd=(\d+)', html)
    if not mc or not md:
        return None
    c, d = mc.group(1), int(md.group(1))
    target, n = "0" * d, 0
    while not hashlib.sha256(f"{c}{n}".encode()).hexdigest().startswith(target):
        n += 1
    return c, n


def _curl(url: str, cookies: str, post: dict | None = None) -> str:
    cmd = ["curl", "-sS", "--max-time", "30", "-A", UA["User-Agent"],
           "-c", cookies, "-b", cookies]
    if post:
        cmd += ["--data-urlencode", f"c={post['c']}", "--data-urlencode", f"n={post['n']}"]
    cmd.append(url)
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=40).stdout
    except Exception:
        return ""


def fetch_monthly(ticker: str, cookies: str) -> list[tuple[str, float]]:
    """Fetch via curl (reliably reaches Stooq), solving the PoW challenge once per cookie jar."""
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol(ticker)}&i=m"
    text = _curl(url, cookies)
    if "requires JavaScript" in text or "__verify" in text:
        pw = _pow_n(text)
        if pw:
            _curl("https://stooq.com/__verify", cookies, {"c": pw[0], "n": pw[1]})
            text = _curl(url, cookies)
    if not text or text.startswith("<") or "Date" not in text[:20]:
        return []
    rows = list(csv.DictReader(io.StringIO(text)))
    out, prev = [], None
    for row in rows:
        try:
            close = float(row["Close"]); month = row["Date"][:7]
        except (ValueError, KeyError):
            continue
        if prev is not None and prev > 0:
            out.append((month, round(close / prev - 1, 5)))
        prev = close
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cookies = os.path.join(tempfile.gettempdir(), "stooq_cookies.txt")
    all_rows, missing = [], []
    for co in COMPANIES:
        data = fetch_monthly(co.ticker, cookies)
        if not data:
            missing.append(co.ticker)
            print(f"  MISS {co.company_id:20s} {co.ticker}")
        else:
            for month, ret in data:
                all_rows.append((co.company_id, month, ret))
            print(f"  ok   {co.company_id:20s} {co.ticker:6s} {len(data)} months "
                  f"({data[0][0]}..{data[-1][0]})")
        time.sleep(0.3)  # be polite

    path = OUT / "returns.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["company_id", "month", "ret"])
        w.writerows(sorted(all_rows))
    print(f"\nwrote {len(all_rows)} return rows -> {path}")
    if missing:
        print("missing tickers (need manual source):", ", ".join(missing))


if __name__ == "__main__":
    main()
