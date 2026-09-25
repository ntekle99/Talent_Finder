#!/usr/bin/env python3
"""Download every DOL LCA / H-1B disclosure file (FY2008-present) into data/raw/.

Streaming, resumable (skips files already fully downloaded), tolerant of 404s (some
quarters may not be published yet). ~3 GB total across all fiscal years.
"""
from __future__ import annotations
import sys
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
BASE = "https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/"

# Annual files (older vintages, heterogeneous names).
ANNUAL = {
    "FY2008": "H-1B_Case_Data_FY2008.xlsx",
    "FY2009a": "Icert_LCA_FY2009.xlsx",
    "FY2009b": "H-1B_Case_Data_FY2009.xlsx",
    "FY2010": "H-1B_FY2010.xlsx",
    "FY2011": "H-1B_iCert_LCA_FY2011_Q4.xlsx",
    "FY2012": "LCA_FY2012_Q4.xlsx",
    "FY2013": "LCA_FY2013.xlsx",
    "FY2014": "H-1B_FY14_Q4.xlsx",
    "FY2015": "H-1B_Disclosure_Data_FY15_Q4.xlsx",
    "FY2016": "H-1B_Disclosure_Data_FY16.xlsx",
    "FY2017": "H-1B_Disclosure_Data_FY17.xlsx",
    "FY2018": "H-1B_Disclosure_Data_FY2018_EOY.xlsx",
    "FY2019": "H-1B_Disclosure_Data_FY2019.xlsx",
}


def quarterly_names() -> dict[str, str]:
    # FY2020-FY2026 are quarterly LCA_Disclosure_Data_FYxxxx_Qn.xlsx
    out = {}
    for fy in range(2020, 2027):
        for q in range(1, 5):
            out[f"FY{fy}_Q{q}"] = f"LCA_Disclosure_Data_FY{fy}_Q{q}.xlsx"
    return out


def download(name: str, fname: str) -> str:
    url = BASE + fname
    dest = RAW / fname
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return f"skip  {name:12s} {fname} ({dest.stat().st_size//1_048_576} MB, exists)"
    try:
        with requests.get(url, stream=True, timeout=60,
                          headers={"User-Agent": "talent-finder-research/0.1"}) as r:
            if r.status_code != 200:
                return f"MISS  {name:12s} {fname} (HTTP {r.status_code})"
            tmp = dest.with_suffix(".part")
            n = 0
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk); n += len(chunk)
            tmp.rename(dest)
            return f"OK    {name:12s} {fname} ({n//1_048_576} MB)"
    except Exception as e:
        return f"ERR   {name:12s} {fname} ({e})"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    files = {**ANNUAL, **quarterly_names()}
    print(f"downloading up to {len(files)} files into {RAW}\n", flush=True)
    ok = 0
    for name, fname in files.items():
        msg = download(name, fname)
        print(msg, flush=True)
        if msg.startswith(("OK", "skip")):
            ok += 1
    print(f"\ndone: {ok} files present in {RAW}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
