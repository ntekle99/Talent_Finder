from pathlib import Path

from .models import TalentEvent


def read_jsonl(path: str | Path) -> list[TalentEvent]:
    source = Path(path)
    events: list[TalentEvent] = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(TalentEvent.model_validate_json(line))
    return events


def write_jsonl(events: list[TalentEvent], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(event.model_dump_json() + "\n")
