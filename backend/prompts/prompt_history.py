from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field


class PromptHistoryRecord(BaseModel):
    model: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    response: str
    timestamp: datetime


class PromptHistoryStore:
    def __init__(self, history_file: str | Path = "data/prompt_history.jsonl") -> None:
        self._history_file = Path(history_file)
        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        self._history_file.touch(exist_ok=True)

    def add_entry(self, *, model: str, prompt: str, response: str) -> PromptHistoryRecord:
        record = PromptHistoryRecord(
            model=model,
            prompt=prompt,
            response=response,
            timestamp=datetime.now(timezone.utc),
        )

        with self._history_file.open("a", encoding="utf-8") as history_stream:
            history_stream.write(record.model_dump_json())
            history_stream.write("\n")

        return record

    def list_entries(self, limit: int = 100) -> list[PromptHistoryRecord]:
        if limit < 1:
            return []

        records: list[PromptHistoryRecord] = []

        with self._history_file.open("r", encoding="utf-8") as history_stream:
            for line in history_stream:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    records.append(PromptHistoryRecord.model_validate_json(stripped))
                except ValueError:
                    # Skip malformed lines instead of breaking history reads.
                    continue

        return records[-limit:]
