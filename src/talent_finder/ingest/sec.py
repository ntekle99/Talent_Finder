"""Small SEC adapter for 8-K talent movement research."""

import os
import re
from datetime import UTC, date, datetime

import requests

from ..models import Company, EventType, Evidence, Person, TalentEvent

SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"


def _headers() -> dict[str, str]:
    return {"User-Agent": os.getenv("TALENT_SEC_USER_AGENT", "talent-signal research contact@example.com")}


def submissions(cik: str, limit: int = 100) -> list[dict]:
    cik = str(cik).zfill(10)
    response = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers=_headers(),
        timeout=30,
    )
    response.raise_for_status()
    recent = response.json()["filings"]["recent"]
    records = []
    for i, form in enumerate(recent["form"]):
        if form not in {"8-K", "8-K/A"}:
            continue
        items = str(recent.get("items", [""] * len(recent["form"]))[i])
        if "5.02" in items:
            records.append(
                {
                    "accession": recent["accessionNumber"][i],
                    "cik": cik,
                    "filing_date": datetime.fromisoformat(recent["filingDate"][i]).date(),
                    "primary_document": recent["primaryDocument"][i],
                    "items": items,
                }
            )
        if len(records) >= limit:
            break
    return records


def filing_document_url(cik: str, accession: str, primary_document: str) -> str:
    cik_number = str(int(cik))
    accession_no_dashes = accession.replace("-", "")
    return f"{SEC_ARCHIVES_BASE}/{cik_number}/{accession_no_dashes}/{primary_document}"


def filing_text(cik: str, accession: str, primary_document: str) -> tuple[str, str]:
    url = filing_document_url(cik, accession, primary_document)
    response = requests.get(url, headers=_headers(), timeout=30)
    response.raise_for_status()
    return response.text, url


_PERSON = r"([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){1,3})"
_ROLE = r"([^.;,\n]{3,90})"
_APPOINTED = re.compile(
    rf"{_PERSON}\s+(?i:(?:was\s+)?(?:appointed|named|elected|promoted)\s+(?:as|to serve as)?\s*(?:the\s+)?){_ROLE}",
)
_RESIGNED = re.compile(
    rf"{_PERSON}\s+(?i:(?:resigned|retired|departed|ceased serving)\s+(?:as|from)?\s*(?:the\s+)?){_ROLE}",
)


def events_from_item_502_text(
    text: str,
    company: Company,
    filing_date: date | datetime,
    first_seen_at: datetime,
    source_url: str,
) -> list[TalentEvent]:
    """Extract conservative executive movement events from Item 5.02 filing text.

    This is intentionally a first-pass parser. Ambiguous filings should be sent to a
    review queue rather than silently inflated into strong signals.
    """
    event_at = (
        filing_date
        if isinstance(filing_date, datetime)
        else datetime.combine(filing_date, datetime.min.time(), tzinfo=UTC)
    )
    events: list[TalentEvent] = []
    for event_type, pattern in [(EventType.ARRIVAL, _APPOINTED), (EventType.DEPARTURE, _RESIGNED)]:
        for match in pattern.finditer(text):
            person_name = " ".join(match.group(1).split())
            role = " ".join(match.group(2).split())
            excerpt_start = max(0, match.start() - 120)
            excerpt_end = min(len(text), match.end() + 120)
            excerpt = " ".join(text[excerpt_start:excerpt_end].split())
            event_id = f"sec-{company.cik or company.id}-{event_at.date()}-{event_type}-{person_name}".lower()
            events.append(
                TalentEvent(
                    id=event_id,
                    person=Person(id=person_name.lower().replace(" ", "-"), name=person_name),
                    destination=company,
                    event_type=event_type,
                    event_at=event_at,
                    first_seen_at=first_seen_at,
                    evidence=[
                        Evidence(
                            source_url=source_url,
                            source_name="sec",
                            first_seen_at=first_seen_at,
                            excerpt=excerpt,
                        )
                    ],
                    role=role,
                    confidence=0.75,
                    seniority=0.5,
                    impact=0.5,
                    fit=0.5,
                    scarcity=0.5,
                    sector=company.sector,
                )
            )
    return events


def company_item_502_events(
    company: Company,
    limit: int = 25,
    first_seen_at: datetime | None = None,
) -> list[TalentEvent]:
    if not company.cik:
        raise ValueError("company.cik is required for SEC ingestion")
    seen_at = first_seen_at or datetime.now(UTC)
    events: list[TalentEvent] = []
    for filing in submissions(company.cik, limit=limit):
        text, source_url = filing_text(company.cik, filing["accession"], filing["primary_document"])
        events.extend(
            events_from_item_502_text(
                text=text,
                company=company,
                filing_date=filing["filing_date"],
                first_seen_at=seen_at,
                source_url=source_url,
            )
        )
    return events
