import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .ingest.sec import company_item_502_events
from .models import Company, TalentEvent

REQUIRED_UNIVERSE_COLUMNS = {"cik", "name"}


@dataclass
class BatchIngestResult:
    events: list[TalentEvent]
    errors: list[dict[str, str]]


def load_universe_csv(path: str | Path) -> list[Company]:
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []
    missing = REQUIRED_UNIVERSE_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(f"universe CSV missing required column(s): {', '.join(sorted(missing))}")

    companies: list[Company] = []
    for row in rows:
        cik = str(row["cik"]).strip()
        name = str(row["name"]).strip()
        if not cik or not name:
            continue
        companies.append(
            Company(
                id=(row.get("company_id") or row.get("ticker") or cik).strip(),
                name=name,
                ticker=(row.get("ticker") or "").strip() or None,
                cik=cik,
                sector=(row.get("sector") or "").strip() or None,
                market_cap=float(row["market_cap"]) if row.get("market_cap") else None,
            )
        )
    return companies


def ingest_sec_universe(
    companies: list[Company],
    limit_per_company: int = 25,
    continue_on_error: bool = True,
    fetcher: Callable[[Company, int], list[TalentEvent]] | None = None,
) -> BatchIngestResult:
    fetch = fetcher or (lambda company, limit: company_item_502_events(company, limit=limit))
    events: list[TalentEvent] = []
    errors: list[dict[str, str]] = []
    for company in companies:
        try:
            events.extend(fetch(company, limit_per_company))
        except Exception as exc:
            if not continue_on_error:
                raise
            errors.append(
                {
                    "company_id": company.id,
                    "ticker": company.ticker or "",
                    "cik": company.cik or "",
                    "error": str(exc),
                }
            )
    return BatchIngestResult(events=events, errors=errors)
