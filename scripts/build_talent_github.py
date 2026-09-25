#!/usr/bin/env python3
"""GitHub OSS talent signal -> data/panel/talent_github.csv (CURRENT snapshot, no history).

For each company's GitHub org we measure open-source strength as a proxy for engineering
caliber/influence:
    oss_impact   = total stars across the org's top ~30 repos (do they ship OSS people use?)
    org_followers= the org's follower count (audience/mindshare)
    public_repos = breadth

Token read from env GITHUB_TOKEN (never hard-coded). Snapshot only -- GitHub's API exposes no
employment history, so unlike OpenAlex/wage this can't give trajectory. Covers eng-heavy firms
(Palantir, Datadog) that don't publish research.
"""
from __future__ import annotations
import os, sys, csv, json, time, urllib.request, urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from company_map import COMPANIES, ticker_of

TOK = os.environ.get("GITHUB_TOKEN", "")
OUT = ROOT / "data" / "panel" / "talent_github.csv"

# Known org logins where they differ from our company_id.
ORG_OVERRIDE = {
    "alphabet": "google", "meta": "facebook", "amazon": "aws", "tesla": "teslamotors",
    "snap": "Snapchat", "snowflake": "snowflakedb", "unity": "Unity-Technologies",
    "servicenow": "ServiceNow", "ebay": "eBay", "datadog": "DataDog",
    "crowdstrike": "CrowdStrike", "workday": "Workday", "texas_instruments": "",
    "applied_materials": "", "hp": "", "block": "block", "coinbase": "coinbase",
    "doordash": "DoorDash", "qualcomm": "quic", "broadcom": "Broadcom",
}


def gh(path: str) -> dict | list | None:
    url = f"https://api.github.com/{path}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {TOK}", "Accept": "application/vnd.github+json",
        "User-Agent": "talent-finder"})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 403:  # secondary/search rate limit
                time.sleep(20); continue
            return None
        except Exception:
            time.sleep(3)
    return None


# Companies with no meaningful/appropriate GitHub org (avoid bad fuzzy matches).
NO_ORG = {"micron"}  # fallback kept matching "micronaut-projects" (the Java framework)


def find_org(co) -> str | None:
    if co.company_id in NO_ORG:
        return None
    cand = ORG_OVERRIDE.get(co.company_id, co.company_id)
    if cand:
        d = gh(f"orgs/{urllib.parse.quote(cand)}")
        if isinstance(d, dict) and d.get("login"):
            return d["login"]
    # fallback: search orgs by the company's primary name fragment (strict token match)
    frag = co.fragments[0]
    ftok = frag.split()[0].lower()
    d = gh(f"search/users?q={urllib.parse.quote(frag)}+type:org&per_page=5")
    time.sleep(2)
    if isinstance(d, dict):
        for it in d.get("items", []):
            login = it["login"].lower()
            if login == ftok or login.startswith(ftok + "-") or login.startswith(ftok + "inc"):
                return it["login"]
    return None


def oss_impact(org: str) -> tuple[int, str, int]:
    """Sum stars of the org's top repos; return (total_stars, top_repo, top_stars)."""
    d = gh(f"search/repositories?q=org:{urllib.parse.quote(org)}&sort=stars&order=desc&per_page=30")
    time.sleep(2)
    if not isinstance(d, dict) or not d.get("items"):
        return 0, "", 0
    stars = sum(r["stargazers_count"] for r in d["items"])
    top = d["items"][0]
    return stars, top["name"], top["stargazers_count"]


def main():
    if not TOK:
        sys.exit("set GITHUB_TOKEN")
    tk = ticker_of()
    rows = []
    for co in COMPANIES:
        org = find_org(co)
        if not org:
            print(f"MISS {co.company_id:20s} (no org)", flush=True)
            continue
        info = gh(f"orgs/{org}") or {}
        stars, top_repo, top_stars = oss_impact(org)
        rows.append((co.company_id, tk.get(co.company_id, ""), org,
                     info.get("followers", 0), info.get("public_repos", 0), stars,
                     top_repo, top_stars))
        print(f"ok   {co.company_id:20s} @{org:20s} oss_stars={stars:>8}  "
              f"top={top_repo}({top_stars})", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["company_id", "ticker", "github_org", "org_followers",
                    "public_repos", "oss_impact_stars", "top_repo", "top_repo_stars"])
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {OUT}")
    print("\n=== OSS IMPACT (total stars across org's top repos) ===")
    for r in sorted(rows, key=lambda x: -x[5])[:25]:
        print(f"  {r[0]:20s} {r[5]:>9,} stars   @{r[2]}  (top: {r[6]} {r[7]:,})")


if __name__ == "__main__":
    main()
