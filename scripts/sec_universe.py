"""Map H1B employer names -> public tickers using SEC's official company list (10k+ firms).

Exact normalized-name matching (high precision; avoids fuzzy false positives). Plus curated
overrides for big subsidiary cases where the filing entity != the listed parent
(GOOGLE LLC -> GOOGL, AMAZON.COM SERVICES -> AMZN, FACEBOOK -> META).
"""
from __future__ import annotations
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEC_JSON = ROOT / "data" / "sec_tickers.json"

_SUFFIX = {"corp", "corporation", "inc", "incorporated", "co", "company", "companies",
           "ltd", "limited", "llc", "lp", "plc", "holdings", "holding", "group",
           "the", "sa", "ag", "nv", "se", "usa", "us", "america", "americas"}

# Subsidiary / DBA -> ticker overrides (filing name won't match the listed parent).
OVERRIDES = {
    "google": "GOOGL", "alphabet": "GOOGL", "youtube": "GOOGL", "google cloud": "GOOGL",
    "amazon": "AMZN", "amazon com services": "AMZN", "amazon web services": "AMZN",
    "amazon development center": "AMZN", "aws": "AMZN",
    "facebook": "META", "meta platforms": "META", "instagram": "META", "whatsapp": "META",
    "alphabet inc": "GOOGL",
}


def normalize(name: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", (name or "").lower())
    s = re.sub(r"\bthe\b", " ", s)
    toks = [t for t in s.split() if t]
    while toks and toks[-1] in _SUFFIX:
        toks.pop()
    return " ".join(toks)


class SecMatcher:
    def __init__(self):
        data = json.loads(SEC_JSON.read_text())
        self.name2ticker: dict[str, str] = {}
        for row in data.values():
            key = normalize(row["title"])
            if key and key not in self.name2ticker:  # first wins (dedupe share classes)
                self.name2ticker[key] = row["ticker"].upper()
        self.overrides = {normalize(k): v for k, v in OVERRIDES.items()}

    def match(self, employer: str) -> str | None:
        n = normalize(employer)
        if not n:
            return None
        if n in self.overrides:
            return self.overrides[n]
        return self.name2ticker.get(n)
