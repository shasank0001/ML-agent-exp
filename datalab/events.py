"""Event model and the append-only JSONL log.

The agent runtime yields :class:`Event` objects; the same objects are appended
to ``runs/<session_id>/events.jsonl``. The log doubles as a replay source and as
raw material for the paper's evaluation, so the schema is kept stable.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

EventType = Literal[
    "session_start",
    "user_message",
    "assistant_delta",
    "assistant_message",
    "tool_start",
    "tool_result",
    "approval_request",
    "approval_response",
    "ask_user",
    "ask_user_response",
    "warning",
    "state_update",
    "error",
    "done",
]

#: Delta events are high-volume; they are rendered live but not written to disk.
TRANSIENT_EVENT_TYPES: frozenset[str] = frozenset({"assistant_delta"})


def utc_now() -> str:
    """ISO-8601 UTC timestamp with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Event(BaseModel):
    """One thing that happened, in order, during a session."""

    seq: int = 0
    ts: str = Field(default_factory=utc_now)
    session_id: str = ""
    type: EventType
    data: dict[str, Any] = Field(default_factory=dict)


class EventLogger:
    """Sequences events for one session and appends them to a JSONL file."""

    def __init__(self, session_id: str, path: Path, *, secrets: tuple[str, ...] = ()) -> None:
        self.session_id = session_id
        self.path = Path(path)
        self._seq = 0
        self._lock = threading.Lock()
        self._buffer: list[Event] = []
        self._secrets = tuple(secrets)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- construction ---------------------------------------------------
    def make(self, type: EventType, data: dict[str, Any] | None = None) -> Event:
        """Create the next event in the sequence (not yet written)."""
        with self._lock:
            self._seq += 1
            seq = self._seq
        return Event(seq=seq, ts=utc_now(), session_id=self.session_id, type=type, data=data or {})

    def emit(self, type: EventType, data: dict[str, Any] | None = None) -> Event:
        """Create *and* persist an event, unless it is transient."""
        event = self.make(type, data)
        self.log(event)
        return event

    # -- persistence ----------------------------------------------------
    def log(self, event: Event) -> None:
        """Record an event; persist it unless it is a transient stream delta.

        Stream deltas are deliberately *not* kept in memory either: a long
        session would otherwise accumulate every token it ever rendered. The
        final ``assistant_message`` carries the same text.
        """
        if event.type in TRANSIENT_EVENT_TYPES:
            return
        payload = redact_data(event.model_dump(mode="json"), self._secrets)
        line = json.dumps(payload, ensure_ascii=False, default=str)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        with self._lock:
            self._buffer.append(event)

    # -- reading --------------------------------------------------------
    @property
    def events(self) -> list[Event]:
        """Every persisted event from this process, in order."""
        return list(self._buffer)

    def read_all(self) -> list[dict[str, Any]]:
        """Parse the JSONL file back into a list of dicts (replay helper)."""
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out


def redact_data(data: Any, secrets: tuple[str, ...]) -> Any:
    """Recursively replace secret literals in strings inside ``data``."""
    if not secrets:
        return data
    if isinstance(data, str):
        for secret in secrets:
            cleaned = (secret or "").strip() if isinstance(secret, str) else ""
            if len(cleaned) >= 4 and cleaned in data:
                data = data.replace(cleaned, "***redacted***")
        return data
    if isinstance(data, dict):
        return {k: redact_data(v, secrets) for k, v in data.items()}
    if isinstance(data, list):
        return [redact_data(v, secrets) for v in data]
    return data
