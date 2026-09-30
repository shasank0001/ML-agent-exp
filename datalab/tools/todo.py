"""The ``todo`` tool — the agent replaces the whole to-do list with one call."""

from __future__ import annotations

from typing import Any

from .base import ToolContext, ToolResult

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "description": "The complete to-do list for the current task, in order.",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Short stable id, e.g. '1', '2'."},
                    "text": {"type": "string", "description": "One concrete, short step."},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "done"],
                        "description": "pending | in_progress | done",
                    },
                },
                "required": ["id", "text", "status"],
            },
        },
        "note": {
            "type": "string",
            "description": "Optional one-line note shown next to the list, e.g. why it changed.",
        },
    },
    "required": ["items"],
}

DESCRIPTION = (
    "Replace the visible to-do list. Send the FULL list every time, including the items that "
    "have not changed, and mark finished work as 'done'. Call this before starting a multi-step "
    "task and update it as you go so the user can follow along."
)


def render(items: list[Any]) -> str:
    marks = {"pending": "[ ]", "in_progress": "[~]", "done": "[x]"}
    return "\n".join(f"{marks.get(i.status, '[ ]')} {i.id}. {i.text}" for i in items)


async def handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    raw_items = args.get("items")
    if not isinstance(raw_items, list):
        return ToolResult.fail("todo needs an 'items' array. Example: {'items': [{'id': '1', 'text': 'Profile the data', 'status': 'done'}]}")
    cleaned: list[dict[str, Any]] = []
    for i, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            return ToolResult.fail(f"todo item #{i} is not an object: {item!r}")
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        status = str(item.get("status", "pending")).strip().lower()
        if status not in {"pending", "in_progress", "done"}:
            status = "pending"
        cleaned.append({"id": str(item.get("id") or i).strip(), "text": text, "status": status})
    if not cleaned:
        return ToolResult.fail("todo items were empty; pass at least one {'id', 'text', 'status'} item.")
    plan = ctx.state.set_plan(cleaned)
    done = sum(1 for p in plan if p.status == "done")
    return ToolResult.ok(
        f"To-do list updated ({done}/{len(plan)} done):\n{render(plan)}",
        data={"plan": ctx.state.summary_dict()["plan"]},
    )
