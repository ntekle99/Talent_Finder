from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class EventType(StrEnum):
    ARRIVAL = "arrival"
    DEPARTURE = "departure"
    PROMOTION = "promotion"
    RESEARCH = "research"
    PATENT = "patent"
    OPEN_SOURCE = "open_source"


class Person(BaseModel):
    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)


class Company(BaseModel):
    id: str
    name: str
    ticker: str | None = None
    cik: str | None = None
    sector: str | None = None
    market_cap: float | None = None


class Evidence(BaseModel):
    source_url: HttpUrl | str
    source_name: str
    first_seen_at: datetime
    excerpt: str = ""
    extraction_version: str = "1"


class TalentEvent(BaseModel):
    id: str
    person: Person
    destination: Company
    event_type: EventType
    event_at: datetime
    first_seen_at: datetime
    evidence: list[Evidence] = Field(min_length=1)
    role: str | None = None
    source_company: Company | None = None
    impact: float = Field(default=0.5, ge=0, le=1)
    scarcity: float = Field(default=0.5, ge=0, le=1)
    fit: float = Field(default=0.5, ge=0, le=1)
    seniority: float = Field(default=0.5, ge=0, le=1)
    confidence: float = Field(default=0.8, ge=0, le=1)
    sector: str | None = None


class DailyCompanyScore(BaseModel):
    as_of: date
    company: Company
    score: float
    event_count: int
    factors: dict[str, float]
    explanations: list[str]


class PaperPosition(BaseModel):
    as_of: date
    ticker: str
    direction: int = Field(ge=-1, le=1)
    weight: float
    score: float
    rationale: str


def model_from_row(row: dict[str, Any]) -> TalentEvent:
    """Build a normalized event from a CSV-like mapping."""
    person = Person(id=str(row["person_id"]), name=str(row["person_name"]))
    company = Company(
        id=str(row["company_id"]),
        name=str(row["company_name"]),
        ticker=row.get("ticker") or None,
        sector=row.get("sector") or None,
    )
    source = None
    if row.get("source_company_id"):
        source = Company(
            id=str(row["source_company_id"]),
            name=str(row.get("source_company_name", row["source_company_id"])),
        )
    evidence = Evidence(
        source_url=str(row["source_url"]),
        source_name=str(row.get("source_name", "user_csv")),
        first_seen_at=row["first_seen_at"],
        excerpt=str(row.get("excerpt", "")),
    )
    return TalentEvent(
        id=str(row.get("event_id", f"{person.id}-{row['event_at']}")),
        person=person,
        destination=company,
        event_type=EventType(str(row["event_type"])),
        event_at=row["event_at"],
        first_seen_at=row["first_seen_at"],
        evidence=[evidence],
        role=row.get("role") or None,
        source_company=source,
        impact=float(row.get("impact", 0.5)),
        scarcity=float(row.get("scarcity", 0.5)),
        fit=float(row.get("fit", 0.5)),
        seniority=float(row.get("seniority", 0.5)),
        confidence=float(row.get("confidence", 0.8)),
        sector=row.get("sector") or None,
    )
