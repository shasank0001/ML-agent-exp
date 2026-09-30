"""Tools over the research state: ``record_finding`` and ``query_state``."""

from __future__ import annotations

import json
from typing import Any

from ..state import ResearchState
from .base import ToolContext, ToolResult

# -- record_finding -----------------------------------------------------
FINDING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {
            "type": "string",
            "description": (
                "One short, factual, evidence-based sentence. Must be checkable against the "
                "logged experiments, e.g. 'RandomForest beat logistic regression by 0.04 macro-F1 "
                "on the held-out test set'."
            ),
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional short tags, e.g. ['feature', 'data-quality'].",
        },
    },
    "required": ["text"],
}
FINDING_DESCRIPTION = (
    "Record a durable conclusion in the research state so it survives into later turns and "
    "later turns of the conversation. Only record things you actually measured; never guess."
)


async def record_finding(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    text = str(args.get("text", "")).strip()
    if not text:
        return ToolResult.fail("record_finding needs a non-empty 'text'.")
    tags = args.get("tags")
    if isinstance(tags, list) and tags:
        clean_tags = [str(t).strip() for t in tags if str(t).strip()][:6]
        if clean_tags:
            text = f"{text} [tags: {', '.join(clean_tags)}]"
    if len(text) > 600:
        text = text[:597] + "..."
    existing = ctx.state.findings
    if text in existing:
        return ToolResult.ok("That finding is already recorded; not duplicated.", data={"findings": len(existing)})
    ctx.state.add_finding(text)
    return ToolResult.ok(
        f"Finding recorded ({len(ctx.state.findings)} total): {text}",
        data={"n_findings": len(ctx.state.findings)},
    )


# -- query_state --------------------------------------------------------
QUERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "section": {
            "type": "string",
            "enum": ["dataset", "task", "plan", "experiments", "findings", "all"],
            "description": "Which part of the research state to return. Default: all.",
        }
    },
}
QUERY_DESCRIPTION = (
    "Read one section of the research state in full: dataset profile, task, plan, every logged "
    "experiment with its metrics, or the findings list. Use this before answering a question such as "
    "'why did model X beat model Y?' — the answer is in the recorded experiments."
)

SECTIONS = ("dataset", "task", "plan", "experiments", "findings")


def _section_text(state: ResearchState, section: str) -> str:
    if section == "dataset":
        return state.dataset.model_dump_json(indent=2) if state.dataset else "No dataset profiled yet."
    if section == "task":
        return state.task.model_dump_json(indent=2) if state.task else "No task defined yet."
    if section == "plan":
        if not state.plan:
            return "Plan is empty."
        return "\n".join(f"[{t.status}] {t.id}. {t.text}" for t in state.plan)
    if section == "experiments":
        if not state.experiments:
            return "No experiments logged yet."
        return json.dumps([e.model_dump(mode="json") for e in state.sorted_experiments()], indent=2)
    if section == "findings":
        return "\n".join(f"- {f}" for f in state.findings) if state.findings else "No findings recorded yet."
    return "Unknown section."


async def query_state(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    section = str(args.get("section", "all")).strip().lower() or "all"
    if section not in SECTIONS and section != "all":
        return ToolResult.fail(
            f"Unknown section {section!r}. Valid sections: {', '.join(SECTIONS)}, all."
        )
    state = ctx.state
    if section == "all":
        body = "\n\n".join(_section_text(state, s) for s in SECTIONS)
        extra = ""
        if ctx.executor is not None:
            extra = (
                f"\n\nPython namespace variables: {', '.join(ctx.executor.variable_names()) or 'none yet'}\n"
                f"{ctx.executor.history_note()}"
            )
        return ToolResult.ok(body + extra, data={"section": section})
    return ToolResult.ok(_section_text(state, section), data={"section": section})
