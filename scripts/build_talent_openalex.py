#!/usr/bin/env python3
"""Person-level talent signal from OpenAlex: track ELITE RESEARCHERS moving INTO each company.

For each company we resolve its OpenAlex institution(s), pull its highest-cited authors,
keep only Computer-Science / AI / Engineering people (drop bio/medicine collaborators),
and for each one find the YEAR they first became affiliated with the company (a "join") plus
where they came from (provenance). We weight each join by the person's caliber (citations,
h-index) -- so one Dieter Fox counts far more than ten unknowns.

Output: data/panel/talent_openalex.csv  (company, year, joiners, elite_inflow, mean_h)
and a printed ranking of recent elite-talent inflow.

Scope: OpenAlex sees RESEARCH talent. Non-publishing firms (Palantir, DoorDash) have little
footprint -- those need the interview-bar / other signals, tracked separately.
"""
from __future__ import annotations
import sys, json, time, math, csv, urllib.request, urllib.parse
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from company_map import COMPANIES

MAILTO = "ntekle@nvidia.com"
OUT = ROOT / "data" / "panel" / "talent_openalex.csv"
TOP_AUTHORS = 1500          # per company, by citations -- we only care about elite talent
PER_PAGE = 200

TECH_FIELDS = {
    "computer science", "artificial intelligence", "machine learning", "deep learning",
    "computer vision", "natural language processing", "robotics", "engineering",
    "electrical engineering", "computer engineering", "mathematics", "data science",
    "pattern recognition", "algorithm", "distributed computing", "parallel computing",
    "computer hardware", "operating system", "programming language", "computer network",
}

# Some companies have no research footprint in OpenAlex (eng-heavy, non-publishing) ->
# these are the ones the interview-bar / GitHub-OSS signals must cover instead.
KNOWN_NO_RESEARCH = {"palantir", "doordash", "block", "shopify", "roku", "lyft",
                     "coinbase", "roblox", "okta", "cloudflare", "datadog",
                     "crowdstrike", "zscaler", "twilio"}


import urllib.error
INST_CACHE = ROOT / "scripts" / "openalex_institutions.json"
_inst_cache: dict = json.loads(INST_CACHE.read_text()) if INST_CACHE.exists() else {}


def api(path: str) -> dict:
    url = f"https://api.openalex.org/{path}{'&' if '?' in path else '?'}mailto={MAILTO}"
    for _ in range(6):
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            time.sleep(12 if e.code in (429, 403) else 3)   # back off hard on rate limit
        except Exception:
            time.sleep(3)
    return {}


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum() or ch == " ").strip()


def resolve_institutions(co) -> list[str]:
    """Search OpenAlex by the company's short name AND its real-name fragments, keep
    company-type entities matching any of them. Merges US/UK variants and subsidiaries
    (DeepMind->Alphabet, Instagram->Meta, Advanced Micro Devices->AMD)."""
    if co.company_id in _inst_cache:
        return _inst_cache[co.company_id]
    terms = [co.company_id.replace("_", " ")] + list(co.fragments)
    matchers = {_norm(t) for t in terms if len(_norm(t)) >= 3}
    ids: dict[str, None] = {}
    searched: set[str] = set()
    for term in terms:
        q = _norm(term)
        if len(q) < 3 or q in searched:
            continue
        searched.add(q)
        d = api(f"institutions?search={urllib.parse.quote(term)}&per-page=25")
        for r in d.get("results", []):
            if r.get("type") != "company" or r["works_count"] <= 20:
                continue
            disp = _norm(r["display_name"])
            dtok = disp.split()[0]
            for m in matchers:
                mtok = m.split()[0]
                if m in disp or (len(mtok) >= 4 and dtok == mtok):
                    ids[r["id"].rsplit("/", 1)[-1]] = None
                    break
        time.sleep(0.5)
    res = list(ids)
    if res:  # only cache successful resolutions (so throttled misses retry next run)
        _inst_cache[co.company_id] = res
        INST_CACHE.write_text(json.dumps(_inst_cache, indent=0))
    return res


def is_tech(author: dict) -> bool:
    for c in author.get("x_concepts", []):
        if c.get("score", 0) >= 0.3 and c["display_name"].lower() in TECH_FIELDS:
            return True
    return False


def harvest(company_id: str, name: str, inst_ids: list[str]) -> list[dict]:
    """Return per-year join records for a company: [{year, n, inflow, h_sum}]."""
    filt = "|".join(inst_ids)
    cursor, seen, joins = "*", 0, []
    while cursor and seen < TOP_AUTHORS:
        d = api(f"authors?filter=affiliations.institution.id:{filt}"
                f"&select=display_name,cited_by_count,summary_stats,affiliations,x_concepts"
                f"&sort=cited_by_count:desc&per-page={PER_PAGE}&cursor={cursor}")
        results = d.get("results", [])
        if not results:
            break
        for a in results:
            seen += 1
            if not is_tech(a):
                continue
            years = [y for af in a["affiliations"]
                     if af["institution"]["id"].rsplit("/", 1)[-1] in inst_ids
                     for y in af.get("years", [])]
            uniq = sorted(set(years))
            # advisory/visiting/"Scholar" affiliations show up as a single year -> drop them;
            # require a real multi-year tenure to count as a genuine join.
            if len(uniq) < 2:
                continue
            join_year = uniq[0]
            caliber = math.log10(a["cited_by_count"] + 1)      # field-robust weight
            h = a.get("summary_stats", {}).get("h_index", 0) or 0
            joins.append({"year": join_year, "caliber": caliber, "h": h,
                          "name": a["display_name"], "cited": a["cited_by_count"]})
        cursor = d.get("meta", {}).get("next_cursor")
        if seen >= TOP_AUTHORS:
            break
    return joins


def main():
    rows = []                      # (company, year, joiners, inflow, mean_h)
    recent = {}                    # company -> recent elite inflow (last 3 yrs)
    notable = {}                   # company -> top joiners for display
    for co in COMPANIES:
        if co.company_id in KNOWN_NO_RESEARCH:
            print(f"skip  {co.company_id:20s} (no OpenAlex research footprint)", flush=True)
            continue
        inst_ids = resolve_institutions(co)
        if not inst_ids:
            print(f"MISS  {co.company_id:20s} (no company institution found)", flush=True)
            continue
        joins = harvest(co.company_id, co.company_id, inst_ids)
        by_year = defaultdict(lambda: [0, 0.0, 0.0])
        for j in joins:
            cell = by_year[j["year"]]
            cell[0] += 1; cell[1] += j["caliber"]; cell[2] += j["h"]
        for yr in sorted(by_year):
            n, inflow, hsum = by_year[yr]
            rows.append((co.company_id, yr, n, round(inflow, 3), round(hsum / n, 1)))
        recent[co.company_id] = sum(j["caliber"] for j in joins if j["year"] >= 2022)
        notable[co.company_id] = sorted(joins, key=lambda x: -x["cited"])[:3]
        print(f"ok    {co.company_id:20s} {len(joins):>4} tech joiners  "
              f"recent(22+)={recent[co.company_id]:.1f}  insts={inst_ids}", flush=True)
        time.sleep(0.2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["company_id", "year", "joiners", "elite_inflow", "mean_h"])
        w.writerows(sorted(rows))
    print(f"\nwrote {len(rows)} company-year rows -> {OUT}")

    print("\n=== WHERE ELITE RESEARCH TALENT IS FLOWING (caliber-weighted inflow, 2022+) ===")
    for cid, score in sorted(recent.items(), key=lambda x: -x[1])[:20]:
        names = ", ".join(f"{j['name']}({j['cited']//1000}k)" for j in notable[cid][:2])
        print(f"  {cid:20s} {score:6.1f}   {names}")


if __name__ == "__main__":
    main()
