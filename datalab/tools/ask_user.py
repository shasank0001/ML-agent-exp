"""The ``ask_user`` tool — pauses the agent and asks the user a question.

The UI callback is injected into the agent (``request_ask_user``); this module
only builds the question and validates the answer. The agent loop emits
``ask_user`` / ``ask_user_response`` events around it so the transcript shows
the round trip.
"""

from __future__ import annotations

from typing import Any

from .base import ToolContext, ToolResult

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "description": (
                "One clear question. Say why you are asking and what each option means. "
                "Do not ask a question you can answer yourself from the data profile."
            ),
        },
        "options": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional short list of choices. Omit for a free-text answer.",
        },
    },
    "required": ["question"],
}

DESCRIPTION = (
    "Ask the user a question and wait for the answer. Use it when the target column, the goal, or "
    "a business decision is ambiguous, or when a choice of trade-off is genuinely the user's to make. "
    "Do not use it to confirm something you can determine from the data yourself."
)

MAX_QUESTION_CHARS = 1_200


async def handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    question = str(args.get("question", "")).strip()
    if not question:
        return ToolResult.fail("ask_user needs a 'question'.")
    if len(question) > MAX_QUESTION_CHARS:
        question = question[: MAX_QUESTION_CHARS - 3] + "..."

    raw_options = args.get("options")
    options: list[str] = []
    if isinstance(raw_options, list):
        options = [str(o).strip() for o in raw_options if str(o).strip()][:6]

    agent = ctx.agent
    if agent is None or not hasattr(agent, "ask_user"):
        return ToolResult.fail(
            "ask_user is not available in this context (no UI attached). "
            f"Ask the user directly in your reply instead. Question that was pending: {question}"
        )

    answer = await agent.ask_user(question, options)
    answer = (answer or "").strip()
    if not answer:
        return ToolResult.ok(
            "The user did not answer (or dismissed the question). "
            "Proceed with a stated assumption, or stop and wait for their next message.",
            data={"question": question, "answer": ""},
        )
    return ToolResult.ok(
        f"User answered: {answer}", data={"question": question, "answer": answer, "options": options}
    )
