"""H1B / LCA disclosure ingest -> talent panel rows.

The U.S. DOL publishes LCA disclosure data (free, quarterly, back to ~2008) at
  https://www.dol.gov/agencies/eta/foreign-labor/performance
Download the "LCA Programs (H-1B, H-1B1, E-3)" file for a year, convert the .xlsx to
CSV, and point this adapter at it. Each certified record is a visa-sponsored hire with an
employer, job title, wage, and decision date -- our inflow + seniority + comp signal.

Real employer names are messy ("GOOGLE LLC", "Google Inc.", "GOOGLE CLOUD"); pass a
`name_to_company` mapping (normalized substring -> company_id) to fold them together. This
adapter emits partial panel rows (hires/senior_hires/mean_wage, departures=0); WARN layoff
ingest fills `departures`, and the two are merged on (company_id, month) in build_panel.
"""
from __future__ import annotations
import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Title -> seniority in [0,1]. Higher = scarcer / more decision authority.
_SENIORITY = [
    (re.compile(r"\b(chief|cto|ceo|cfo|vp|vice president|head of|principal|distinguished|fellow)\b", re.I), 1.0),
    (re.compile(r"\b(director|staff|lead|architect|manager)\b", re.I), 0.8),
    (re.compile(r"\b(senior|sr\.?|snr)\b", re.I), 0.6),
    (re.compile(r"\b(engineer|scientist|developer|analyst|designer)\b", re.I), 0.4),
    (re.compile(r"\b(junior|jr\.?|associate|intern|entry)\b", re.I), 0.2),
]

# Common column aliases across DOL vintages (schemas drift year to year).
# Ordered by preference; first present column wins.
_ALIASES = {
    "employer": ["EMPLOYER_NAME", "LCA_CASE_EMPLOYER_NAME"],
    "title": ["JOB_TITLE", "LCA_CASE_JOB_TITLE", "SOC_TITLE", "LCA_CASE_SOC_NAME"],
    "wage": ["WAGE_RATE_OF_PAY_FROM", "WAGE_RATE_OF_PAY_FROM_1", "LCA_CASE_WAGE_RATE_FROM",
             "WAGE_RATE_1", "PREVAILING_WAGE_1"],
    "status": ["CASE_STATUS", "STATUS", "APPROVAL_STATUS"],
    "date": ["DECISION_DATE", "DOL_DECISION_DATE", "RECEIVED_DATE", "SUBMITTED_DATE",
             "LCA_CASE_SUBMIT"],
}


def seniority_of(title: str) -> float:
    for pat, s in _SENIORITY:
        if pat.search(title or ""):
            return s
    return 0.4  # default: individual contributor


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (name or "").lower())


def _pick(header: list[str], names: list[str]) -> int | None:
    upper = [h.strip().upper() for h in header]
    for n in names:
        if n in upper:
            return upper.index(n)
    return None


@dataclass
class Cell:
    hires: int = 0
    senior_sum: float = 0.0
    wage_sum: float = 0.0
    wage_n: int = 0
    def senior_hires(self) -> float:
        return round(self.senior_sum, 2)
    def mean_wage(self) -> int:
        return int(self.wage_sum / self.wage_n) if self.wage_n else 0


def parse_lca_csv(path: str | Path, name_to_company: dict[str, str]) -> dict[tuple[str, str], Cell]:
    """Aggregate certified LCA rows into {(company_id, 'YYYY-MM'): Cell}."""
    path = Path(path)
    agg: dict[tuple[str, str], Cell] = defaultdict(Cell)
    norm_map = {_norm(k): v for k, v in name_to_company.items()}

    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        ci = {k: _pick(header, v) for k, v in _ALIASES.items()}
        if ci["employer"] is None or ci["date"] is None:
            raise ValueError(f"could not locate employer/date columns in {path.name}")

        for row in reader:
            if ci["status"] is not None and ci["status"] < len(row):
                if "CERTIF" not in row[ci["status"]].upper():
                    continue
            employer = _norm(row[ci["employer"]]) if ci["employer"] < len(row) else ""
            company = next((cid for frag, cid in norm_map.items() if frag and frag in employer), None)
            if company is None:
                continue
            raw_date = row[ci["date"]] if ci["date"] < len(row) else ""
            month = _to_month(raw_date)
            if month is None:
                continue
            title = row[ci["title"]] if ci["title"] is not None and ci["title"] < len(row) else ""
            cell = agg[(company, month)]
            cell.hires += 1
            cell.senior_sum += seniority_of(title)
            if ci["wage"] is not None and ci["wage"] < len(row):
                w = _to_float(row[ci["wage"]])
                if w and w > 10_000:  # ignore hourly/garbage
                    cell.wage_sum += w
                    cell.wage_n += 1
    return agg


def _to_month(s: str) -> str | None:
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S", "%m/%d/%y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m")
        except ValueError:
            continue
    return None


def _to_float(s: str) -> float | None:
    try:
        return float(re.sub(r"[,$]", "", s))
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser(description="Aggregate an LCA disclosure CSV into panel cells.")
    ap.add_argument("csv_path")
    ap.add_argument("--map", nargs="*", default=[], help="employer=company_id pairs, e.g. google=google")
    args = ap.parse_args()
    mapping = dict(p.split("=", 1) for p in args.map) or {"google": "google", "nvidia": "nvidia"}
    cells = parse_lca_csv(args.csv_path, mapping)
    w = csv.writer(sys.stdout)
    w.writerow(["company_id", "month", "hires", "senior_hires", "mean_wage", "departures", "source_mask"])
    for (cid, month), c in sorted(cells.items()):
        w.writerow([cid, month, c.hires, c.senior_hires(), c.mean_wage(), 0, 2])
