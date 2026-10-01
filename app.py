"""Chainlit entrypoint: renders the agent's event stream, nothing else.

Every piece of agent behaviour lives in :mod:`datalab`. This file translates
events into Chainlit widgets and routes the user's clicks back into the agent
through the two injected callbacks (`request_approval`, `ask_user`).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import chainlit as cl

from datalab.agent import Agent
from datalab.approvals import ApprovalRequest
from datalab.config import Settings, load_settings
from datalab.llm import LLMClient, SelfTestResult

APPROVAL_TIMEOUT_S = 1800
ASK_USER_TIMEOUT_S = 1_800
SELF_TEST_TIMEOUT_S = 30
PREVIEW_CHARS = 60
MAX_STEP_IMAGES = 8

#: The agent's to-do statuses mapped onto Chainlit's TaskStatus enum.
TODO_STATUS = {
    "pending": cl.TaskStatus.READY,
    "in_progress": cl.TaskStatus.RUNNING,
    "done": cl.TaskStatus.DONE,
}


# -- lifecycle ----------------------------------------------------------
@cl.on_chat_start
async def on_chat_start() -> None:
    """Create the session, say hello, and probe the model's tool-calling support."""
    settings: Settings = load_settings()
    agent = Agent(
        session_id=cl.context.session.id,
        settings=settings,
        llm=LLMClient(settings),
        request_approval=_approval_prompt,
        ask_user=_ask_prompt,
    )
    cl.user_session.set("agent", agent)
    cl.user_session.set("tasklist", None)

    await cl.Message(
        content=(
            f"Session `{agent.session_id}` — working in `{agent.paths['root']}`.\n\n"
            "Upload a CSV and tell me what to predict, or just ask a question about your data."
        ),
        author="DataLab",
    ).send()

    if not settings.is_configured:
        await cl.Message(
            content=(
                "**The LLM is not configured yet.**\n\n"
                "Set these in `.env` and restart:\n\n```\n"
                + "\n".join(settings.missing_pieces())
                + "\n```\n"
            ),
            author="DataLab",
        ).send()
        return

    probe = cl.Message(content="Checking that the model can call tools…", author="DataLab")
    await probe.send()
    # Bounded: an unreachable base_url would otherwise block chat start for
    # minutes behind the OpenAI client's own (much longer) retry budget.
    try:
        result = await asyncio.wait_for(agent.run_self_test(), timeout=SELF_TEST_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 - a failed probe must not block chat start
        reason = (
            f"no response within {SELF_TEST_TIMEOUT_S}s"
            if isinstance(exc, asyncio.TimeoutError)
            else f"{type(exc).__name__}: {exc}"
        )
        result = SelfTestResult(False, settings.model or "(unset)", reason)
    await probe.remove()
    if result.ok:
        await cl.Message(
            content=f"Model `{result.model}` ready — tool calling works.", author="DataLab"
        ).send()
    else:
        await cl.Message(
            content=(
                f"**Heads up: `{result.model}` may not support tool calling.**\n\n"
                f"{result.detail}\n\n"
                "Without tool calling the agent cannot reach its tools, so it can only chat. "
                "Try a model with function-calling support, or switch provider in `.env`:\n\n"
                "```\nLLM_PROVIDER=openrouter\nOPENROUTER_MODEL=<a tool-calling model>\n```\n"
            ),
            author="DataLab",
        ).send()


@cl.on_stop
async def on_stop() -> None:
    """Chainlit's stop button: cancel the in-flight turn."""
    agent: Agent | None = cl.user_session.get("agent")
    if agent is not None:
        agent.cancel()


# -- message handling ---------------------------------------------------
@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Copy uploads into the session, then render the agent's event stream."""
    agent: Agent | None = cl.user_session.get("agent")
    if agent is None:
        await cl.Message(
            content="This session is not initialised. Reload the page to start a new one.",
            author="DataLab",
        ).send()
        return

    if not agent.settings.is_configured:
        await cl.Message(
            content="Set the missing values in `.env` (see the start of this chat) and restart.",
            author="DataLab",
        ).send()
        return

    notes: list[str] = []
    for element in message.elements:
        if not isinstance(element, cl.File):
            continue
        try:
            copied = agent.add_attachment(Path(element.path))
        except Exception as exc:  # noqa: BLE001 - a bad upload must not kill the handler
            notes.append(f"Could not copy `{element.name}` into the session: {type(exc).__name__}: {exc}")
            continue
        notes.append(f"Uploaded `{copied.name}` ({copied.stat().st_size:,} bytes) into `DATA_DIR`.")

    text = (message.content or "").strip()
    text = f"{text}\n\n" + "\n".join(notes) if (text and notes) else (text or "\n".join(notes))
    if not text:
        await cl.Message(
            content="I didn't get any text or file — upload a CSV or type a question.",
            author="DataLab",
        ).send()
        return

    await _render_events(agent, text)


async def _render_events(agent: Agent, text: str) -> None:
    """Drive the agent and render every event it produces."""
    stream: cl.Message | None = None
    steps: dict[str, cl.Step] = {}
    results_before = _results_signature(agent)

    try:
        async for event in agent.run(text):
            kind, data = event.type, event.data

            if kind == "assistant_delta":
                if stream is None:
                    stream = cl.Message(content="", author="DataLab")
                    await stream.send()
                await stream.stream_token(data.get("text", ""))

            elif kind == "assistant_message":
                if stream is None:
                    stream = cl.Message(content=str(data.get("text", "")), author="DataLab")
                    await stream.send()
                else:
                    stream.content = str(data.get("text", "") or stream.content)
                    await stream.update()
                stream = None

            elif kind == "tool_start":
                call_id = data.get("id")
                if call_id is None:
                    continue
                steps[str(call_id)] = await _open_step(data, agent)

            elif kind == "tool_result":
                call_id = data.get("id")
                await _close_step(
                    steps.pop(str(call_id), None) if call_id is not None else None,
                    data, agent,
                )

            elif kind == "approval_response":
                if not data.get("approved"):
                    await cl.Message(content="_You denied that action._", author="DataLab").send()

            elif kind == "state_update":
                await _publish_plan(agent, data)

            elif kind == "warning":
                await cl.Message(content=data.get("message", "")).send()

            elif kind == "error":
                await cl.ErrorMessage(
                    content=str(data.get("message", "Something went wrong."))
                ).send()

    except asyncio.CancelledError:
        await cl.ErrorMessage(content="Run cancelled.").send()
        raise
    finally:
        if stream is not None:
            try:
                await stream.update()
            except Exception:  # noqa: BLE001 - render must not mask cancellation
                pass

    if _results_signature(agent) != results_before:
        await _render_results(agent)


def _results_signature(agent: Agent) -> str:
    """Fingerprint of the results table (count + names + metrics)."""
    try:
        table = agent.lab.results_table()
    except Exception:  # noqa: BLE001 - signature is best-effort
        return ""
    if table is None or len(table) == 0:
        return "empty"
    return str(table.to_dict())


# -- tool steps ---------------------------------------------------------
async def _open_step(data: dict[str, Any], agent=None) -> cl.Step:
    """Create the collapsible step for a tool call and show its input."""
    name = str(data.get("name", "tool"))
    if name == "python":
        est = (data.get("arguments") or {}).get("est_seconds")
        est_text = f"  (~{int(est)}s)" if isinstance(est, (int, float)) and not isinstance(est, bool) else ""
        description = str(data.get("description") or "Run Python")
        title = f"python · {description}" + est_text
        step = cl.Step(name=title, type="tool", language="python", show_input="code")
        step.input = str(data.get("code", ""))[:8000]
    else:
        args = {k: v for k, v in (data.get("arguments") or {}).items() if k != "code"}
        shown = ", ".join(f"{k}={_preview(v)}" for k, v in list(args.items())[:4])
        step = cl.Step(name=f"{name}({shown})" if shown else name, type="tool")
        step.input = ""
    await step.send()
    return step


async def _close_step(step: cl.Step | None, data: dict[str, Any], agent=None) -> None:
    """Fill a tool step with its output, figures and error state."""
    if step is None:  # e.g. a tool the agent never announced
        step = cl.Step(name=str(data.get("name", "tool")), type="tool")
    step.output = str(data.get("text", "")) or "_no output_"
    step.is_error = bool(data.get("error"))

    elements: list[cl.Element] = []
    allowed_roots: list[Path] = []
    if agent is not None:
        try:
            allowed_roots = [
                agent.paths["figures"].resolve(),
                agent.paths["outputs"].resolve(),
            ]
        except Exception:  # noqa: BLE001 - fall back to no images
            allowed_roots = []
    for raw in (data.get("images") or [])[:MAX_STEP_IMAGES]:
        try:
            path = Path(str(raw)).resolve()
        except (OSError, ValueError):
            continue
        if not allowed_roots or not any(
            path == root or path.is_relative_to(root) for root in allowed_roots
        ):
            continue
        if path.exists() and path.is_file() and path.stat().st_size <= 20_000_000:
            elements.append(
                cl.Image(path=str(path), name=path.name, display="inline", size="small")
            )
    step.elements = elements
    await step.update()


# -- results and plan ---------------------------------------------------
async def _render_results(agent: Agent) -> None:
    """Show the harness-owned results table for whatever was logged this turn."""
    try:
        table = agent.lab.results_table()
    except Exception:  # noqa: BLE001 - never let a render failure kill the turn
        return
    if table is None or len(table) == 0:
        return
    await cl.Dataframe(
        data=table, name="Results (from state.json)", display="inline", size="medium"
    ).send()


async def _publish_plan(agent: Agent, data: dict[str, Any] | None) -> None:
    """Keep a single pinned to-do list in sync with the research state."""
    summary = data or agent.state.summary_dict()
    plan = (summary.get("plan") or [])[:20]
    if not plan:
        return

    tasklist = cl.user_session.get("tasklist")
    if tasklist is None:
        tasklist = cl.TaskList(name="Plan", tasks=[_to_task(item) for item in plan], status="Plan")
        await tasklist.send()
        cl.user_session.set("tasklist", tasklist)
        return

    tasklist.tasks = [_to_task(item) for item in plan]
    done = sum(1 for item in plan if item.get("status") == "done")
    tasklist.status = f"{done}/{len(plan)} done"
    best = summary.get("best")
    if best:
        metrics = ", ".join(
            f"{k} {v:.4f}" for k, v in (best.get("metrics") or {}).items() if isinstance(v, float)
        )
        tasklist.name = f"Plan — best: {best.get('name')}" + (f"  ({metrics})" if metrics else "")
    await tasklist.update()


def _to_task(item: dict[str, Any]) -> cl.Task:
    return cl.Task(
        title=str(item.get("text", "")),
        status=TODO_STATUS.get(str(item.get("status")), cl.TaskStatus.READY),
    )


def _preview(value: Any, limit: int = PREVIEW_CHARS) -> str:
    text = str(value).replace("\n", " ")
    return text[:limit] + "…" if len(text) > limit else text


# -- UI callbacks injected into the agent -------------------------------
async def _approval_prompt(request: ApprovalRequest) -> bool:
    """Render an Approve / Deny question and wait for the click."""
    parts = [request.reason]
    if request.code:
        parts.append(f"\n```python\n{request.code}\n```")
    message = cl.AskActionMessage(
        content="\n\n".join(parts),
        author="DataLab",
        timeout=APPROVAL_TIMEOUT_S,
        actions=[
            cl.Action(
                name="approve", payload={"approved": True}, label="Approve", style="primary"
            ),
            cl.Action(
                name="deny", payload={"approved": False}, label="Deny", style="secondary"
            ),
        ],
    )
    try:
        response = await message.send()
    except Exception:  # noqa: BLE001 - a dropped connection must not hang the agent
        return False
    if not response:
        return False
    payload = response.get("payload") or {}
    if "approved" in payload:
        return bool(payload["approved"])
    return str(response.get("name", "")).lower() in {"approve", "approved", "yes"}


async def _ask_prompt(question: str, options: list[str]) -> str:
    """Render a free-text question (plus optional choices) and wait for the answer."""
    content = question
    if options:
        content += "\n\n" + "\n".join(f"- {opt}" for opt in options)
    message = cl.AskUserMessage(content=content, author="DataLab", timeout=ASK_USER_TIMEOUT_S)
    try:
        response = await message.send()
    except Exception:  # noqa: BLE001
        return ""
    if not response:
        return ""
    return str(response.get("output", "")).strip()
