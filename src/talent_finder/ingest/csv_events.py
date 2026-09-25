import csv
from datetime import datetime
from pathlib import Path

from ..models import TalentEvent, model_from_row


def load_csv(path: str | Path) -> list[TalentEvent]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return [
            model_from_row(
                {
                    **row,
                    "event_at": datetime.fromisoformat(row["event_at"]),
                    "first_seen_at": datetime.fromisoformat(row["first_seen_at"]),
                }
            )
            for row in rows
        ]
