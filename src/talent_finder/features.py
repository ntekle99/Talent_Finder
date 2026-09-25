from statistics import mean

from .models import EventType, TalentEvent

ROLE_IMPORTANCE_KEYWORDS = {
    "chief executive": 1.0,
    "ceo": 1.0,
    "chief financial": 0.95,
    "cfo": 0.95,
    "chief technology": 0.95,
    "cto": 0.95,
    "chief scientific": 0.95,
    "chief medical": 0.9,
    "president": 0.85,
    "founder": 0.85,
    "svp": 0.8,
    "senior vice president": 0.8,
    "vp": 0.7,
    "vice president": 0.7,
    "head of": 0.7,
    "director": 0.55,
    "principal": 0.5,
}

SCARCE_SKILL_KEYWORDS = {
    "artificial intelligence": 0.95,
    "ai": 0.9,
    "machine learning": 0.9,
    "gpu": 0.9,
    "semiconductor": 0.9,
    "clinical": 0.85,
    "drug discovery": 0.85,
    "cybersecurity": 0.8,
    "quant": 0.8,
    "distributed systems": 0.75,
    "infrastructure": 0.7,
}

SOURCE_RELIABILITY = {
    "sec": 1.0,
    "company_press_release": 0.9,
    "press_release": 0.85,
    "openalex": 0.75,
    "uspto": 0.75,
    "github": 0.65,
    "user_csv": 0.55,
}


def _keyword_score(text: str, scores: dict[str, float], default: float) -> float:
    normalized = text.lower()
    matches = [score for keyword, score in scores.items() if keyword in normalized]
    return max(matches, default=default)


def role_importance(event: TalentEvent) -> float:
    return _keyword_score(event.role or "", ROLE_IMPORTANCE_KEYWORDS, 0.35)


def scarce_skill_score(event: TalentEvent) -> float:
    text = " ".join([event.role or "", *event.person.skills, event.sector or "", event.destination.sector or ""])
    return _keyword_score(text, SCARCE_SKILL_KEYWORDS, 0.35)


def evidence_strength(event: TalentEvent) -> float:
    source_scores = [SOURCE_RELIABILITY.get(e.source_name.lower(), 0.5) for e in event.evidence]
    excerpt_bonus = 0.1 if any(e.excerpt for e in event.evidence) else 0.0
    multi_source_bonus = min(0.15, 0.05 * max(0, len(event.evidence) - 1))
    return min(1.0, mean(source_scores) + excerpt_bonus + multi_source_bonus)


def fit_score(event: TalentEvent) -> float:
    role = (event.role or "").lower()
    sector = " ".join([event.sector or "", event.destination.sector or ""]).lower()
    if not role or not sector:
        return 0.45
    if any(term in role for term in ["ai", "machine learning", "gpu", "semiconductor"]) and any(
        term in sector for term in ["technology", "semiconductor", "software", "cloud"]
    ):
        return 0.9
    if any(term in role for term in ["clinical", "medical", "drug"]) and any(
        term in sector for term in ["health", "biotech", "pharma"]
    ):
        return 0.9
    if any(term in role for term in ["sales", "revenue", "go-to-market", "marketing"]):
        return 0.65
    return 0.5


def surprise_score(event: TalentEvent) -> float:
    if event.event_type == EventType.PROMOTION:
        return 0.4
    if event.source_company and event.source_company.id != event.destination.id:
        return 0.7
    if event.event_type == EventType.DEPARTURE:
        return 0.65
    return 0.5


def inferred_quality(event: TalentEvent) -> dict[str, float]:
    """Blend supplied features with explainable heuristics from the event text."""
    seniority = max(event.seniority, role_importance(event))
    scarcity = max(event.scarcity, scarce_skill_score(event))
    fit = max(event.fit, fit_score(event))
    source = evidence_strength(event)
    confidence = event.confidence * source
    impact = max(event.impact, 0.5 * seniority + 0.3 * fit + 0.2 * surprise_score(event))
    return {
        "impact": min(1.0, impact),
        "scarcity": min(1.0, scarcity),
        "fit": min(1.0, fit),
        "seniority": min(1.0, seniority),
        "confidence": min(1.0, confidence),
        "source_reliability": source,
        "surprise": surprise_score(event),
    }
