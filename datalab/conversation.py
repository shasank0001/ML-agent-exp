"""Message-history plumbing for the agent loop.

Keeping the OpenAI message list well formed is fiddly enough to deserve its
own module: an assistant message carrying ``tool_calls`` must be followed by
exactly one tool message per call, or every later request is rejected. Three
things maintain that invariant — building messages, bounding the window, and
repairing orphans left behind by a cancellation.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .llm import ToolCall

#: Model output longer than this is truncated head+tail before it goes back into
#: the request, so one noisy cell cannot eat the whole context.
DEFAULT_MAX_TOOL_CHARS = 8_000


def jsonable(value: Any) -> Any:
    """Best-effort conversion of a tool payload to something json.dumps accepts."""
    try:
        json.dumps(value, default=str)
        return value
    except (TypeError, ValueError):
        return json.loads(json.dumps(value, default=str))


def tool_message(call: ToolCall, text: str) -> dict[str, Any]:
    """The ``role="tool"`` message that answers ``call``."""
    return {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": text}


def assistant_tool_calls(text: str, calls: list[ToolCall]) -> dict[str, Any]:
    """The assistant message that requested ``calls``."""
    return {
        "role": "assistant",
        "content": text or None,
        "tool_calls": [c.to_message() for c in calls],
    }


def user_message(text: str, attachments: list[str]) -> str:
    """User text, plus a note about which files this session can reach."""
    if not attachments:
        return text
    names = ", ".join(attachments)
    return (
        f"{text}\n\n[files available in DATA_DIR for this session: {names} — "
        "load one with lab.load(<name>) or pd.read_csv(DATA_DIR + '/<name>')]"
    )


def answered_ids(messages: list[dict[str, Any]]) -> Counter:
    """How many responses exist per tool-call id.

    A multiset, not a set: a local model or proxy that reuses an id across
    turns would otherwise make every later occurrence look already answered.
    """
    return Counter(
        m.get("tool_call_id") for m in messages if m.get("role") == "tool"
    )


def orphan_ids(messages: list[dict[str, Any]], start: int = 0) -> list[str]:
    """Tool call ids in ``messages[start:]`` that have no response."""
    remaining = answered_ids(messages[start:])
    orphans: list[str] = []
    for m in messages[start:]:
        if m.get("role") != "assistant":
            continue
        for c in m.get("tool_calls") or []:
            if remaining[c["id"]] > 0:
                remaining[c["id"]] -= 1
            else:
                orphans.append(c["id"])
    return orphans


def answer_orphans(
    messages: list[dict[str, Any]],
    known: dict[str, ToolCall],
    reason: str = "Cancelled: the run was stopped before this finished.",
) -> int:
    """Answer every unanswered tool call, so the history stays sendable.

    Cancellation can break the tool_call/tool pairing in several places, so it
    is repaired from the whole history rather than tracked at each one.
    Returns how many synthetic answers were appended.
    """
    added = 0
    for call_id in orphan_ids(messages):
        call = known.get(call_id) or ToolCall(id=call_id, name="unknown", arguments={})
        messages.append(tool_message(call, reason))
        added += 1
    return added


def answer_skipped(messages: list[dict[str, Any]], calls: list[ToolCall]) -> None:
    """Answer the calls that were skipped after a cancellation."""
    for call in calls:
        messages.append(
            tool_message(call, "Skipped: the run was cancelled before this tool ran.")
        )


def safe_history_start(messages: list[dict[str, Any]], wanted: int) -> int:
    """Smallest index >= ``wanted`` at which the kept tail has no orphan call."""
    index = max(0, min(wanted, len(messages)))
    while index < len(messages) and orphan_ids(messages, index):
        index += 1
    if index >= len(messages):  # never trim the history to nothing
        index = len(messages)
    return index


def trim(messages: list[dict[str, Any]], keep: int, head: int = 2) -> list[dict[str, Any]]:
    """Bound the history, cutting only where no tool call is orphaned."""
    if len(messages) <= keep:
        return messages
    head = min(head, max(0, keep - 1))
    start = safe_history_start(messages, len(messages) - (keep - head))
    return messages[:head] + messages[start:]


def truncate(text: str, limit: int = DEFAULT_MAX_TOOL_CHARS) -> str:
    """Shorten long tool output, keeping the head and the tail."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    if tail <= 0:
        return text[:limit] + f"\n\n[... {len(text) - limit} characters truncated. ...]\n\n"
    return (
        text[:head]
        + f"\n\n[... {len(text) - limit} characters truncated from the middle. "
        f"The full output is in the event log. ...]\n\n"
        + text[-tail:]
    )


def window(messages: list[dict[str, Any]], system_prompt: str, size: int) -> list[dict[str, Any]]:
    """System prompt plus the last ``size`` messages."""
    if size <= 0:
        return [{"role": "system", "content": system_prompt}]
    return [{"role": "system", "content": system_prompt}, *messages[-size:]]
