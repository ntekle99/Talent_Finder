from datetime import date
from math import exp

from .features import inferred_quality
from .models import DailyCompanyScore, EventType, TalentEvent

_DIRECTION = {
    EventType.ARRIVAL: 1.0,
    EventType.PROMOTION: 0.7,
    EventType.RESEARCH: 0.5,
    EventType.PATENT: 0.5,
    EventType.OPEN_SOURCE: 0.4,
    EventType.DEPARTURE: -1.0,
}


def score_event(event: TalentEvent, as_of: date, half_life_days: float = 90) -> float:
    if event.first_seen_at.date() > as_of:
        return 0.0
    age = max(0, (as_of - event.event_at.date()).days)
    decay = exp(-0.693147 * age / half_life_days)
    quality_features = inferred_quality(event)
    quality = (
        0.25 * quality_features["impact"]
        + 0.18 * quality_features["scarcity"]
        + 0.18 * quality_features["fit"]
        + 0.18 * quality_features["seniority"]
        + 0.13 * quality_features["confidence"]
        + 0.08 * quality_features["surprise"]
    )
    return _DIRECTION[event.event_type] * quality * decay


def score_company(events: list[TalentEvent], as_of: date) -> DailyCompanyScore:
    relevant = [e for e in events if e.destination.id and e.first_seen_at.date() <= as_of]
    contributions = [score_event(e, as_of) for e in relevant]
    quality_features = [inferred_quality(e) for e in relevant]
    score = sum(contributions)
    arrivals = sum(1 for e in relevant if _DIRECTION[e.event_type] > 0)
    departures = sum(1 for e in relevant if e.event_type == EventType.DEPARTURE)
    factors = {
        name: sum(c * q[name] for q, c in zip(quality_features, contributions))
        for name in ["impact", "scarcity", "fit", "seniority", "confidence", "source_reliability", "surprise"]
    }
    strongest = max(quality_features, key=lambda q: q["impact"], default=None)
    explanations = [
        f"{arrivals} positive talent event(s), {departures} departure(s)",
        "recent events receive more weight through 90-day half-life",
    ]
    if strongest:
        explanations.append(
            f"strongest inferred drivers: impact {strongest['impact']:.2f}, "
            f"fit {strongest['fit']:.2f}, seniority {strongest['seniority']:.2f}"
        )
    if departures:
        explanations.append("departure pressure reduces the score")
    return DailyCompanyScore(
        as_of=as_of,
        company=relevant[0].destination if relevant else events[0].destination,
        score=score,
        event_count=len(relevant),
        factors=factors,
        explanations=explanations,
    )


def rank_companies(events: list[TalentEvent], as_of: date) -> list[DailyCompanyScore]:
    companies = {e.destination.id: e.destination for e in events}
    return sorted(
        [
            score_company([e for e in events if e.destination.id == cid], as_of)
            for cid in companies
        ],
        key=lambda item: item.score,
        reverse=True,
    )
