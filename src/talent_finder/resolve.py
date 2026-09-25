from dataclasses import dataclass
from difflib import SequenceMatcher

from .models import Company, Person


@dataclass(frozen=True)
class Match:
    candidate_id: str | None
    confidence: float
    needs_review: bool


def resolve_name(name: str, candidates: list[Person], threshold: float = 0.88) -> Match:
    normalized = " ".join(name.lower().split())
    scored = [
        (candidate.id, max(
            SequenceMatcher(None, normalized, " ".join(candidate.name.lower().split())).ratio(),
            *[
                SequenceMatcher(None, normalized, " ".join(alias.lower().split())).ratio()
                for alias in candidate.aliases
            ],
        ))
        for candidate in candidates
    ]
    if not scored:
        return Match(None, 0.0, True)
    candidate_id, confidence = max(scored, key=lambda item: item[1])
    return Match(candidate_id if confidence >= threshold else None, confidence, confidence < 0.95)


def resolve_company(name: str, candidates: list[Company]) -> Match:
    return resolve_name(name, [Person(id=c.id, name=c.name) for c in candidates])
