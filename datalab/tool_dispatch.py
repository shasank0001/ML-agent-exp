"""Tool dispatch for one agent turn.

Separated from the loop in :mod:`datalab.agent` so the "what happens when a
tool call arrives" question lives in one place: resolve the tool, decide
whether the user must approve it, run the handler, and turn whatever comes back
— success, error or cancellation — into events and a message in the history.

Nothing here raises at the caller except ``asyncio.CancelledError``: an error
becomes a tool result the model can read and repair from.
"""

from __future__ import annotations

import asyncio
import time
import traceback
from collections.abc import Awaitable, Callable
from typing import Any

from .approvals import ApprovalRequest, needs_approval
from .config import Settings
from .conversation import jsonable, tool_message, truncate
from .events import Event
from .llm import ToolCall
from .state import ResearchState
from .tools import REGISTRY, ToolContext, ToolResult

EmitFn = Callable[..., Event]
PushFn = Callable[[Event], Awaitable[None]]


class ToolDispatcher:
    """Runs tool calls for one session, with approval and error handling."""

    def __init__(
        self,
        *,
        ctx: ToolContext,
        state: ResearchState,
        settings: Settings,
        emit: EmitFn,
        push: PushFn,
        append_message: Callable[[dict[str, Any]], None],
        record_call: Callable[[ToolCall], None],
        request_approval: Callable[[ApprovalRequest], Awaitable[bool | None]],
    ) -> None:
        self.ctx = ctx
        self.state = state
        self.settings = settings
        self._emit = emit
        self._push = push
        self._append = append_message
        self._record = record_call
        self.request_approval = request_approval
        self.calls_run = 0
        self.last_timed_out = False
        self.last_elapsed_s = 0.0

    # -- the one entry point ---------------------------------------------
    async def run(self, call: ToolCall) -> bool:
        """Run one tool call. Returns whether it succeeded.

        Raises ``asyncio.CancelledError`` if the run was stopped while an
        approval dialog was open; the call is answered first so the message
        history stays sendable.
        """
        self.last_timed_out = False
        spec = REGISTRY.get(call.name)
        if spec is None:
            await self._answer(
                call,
                f"Unknown tool '{call.name}'. Available tools: {', '.join(sorted(REGISTRY))}. "
                "Call one of those instead.",
                error=True,
            )
            return False

        # Tracked before the approval prompt: cancelling while the dialog is
        # open must still be able to answer this call in the history.
        self._record(call)

        if await self._permission(call) is False:
            return True

        return await self._execute(spec.handler, call)

    # -- approval --------------------------------------------------------
    async def _permission(self, call: ToolCall) -> bool | None:
        """True to proceed, False on denial, None when no prompt is needed."""
        try:
            request: ApprovalRequest | None = needs_approval(
                call,
                self.state,
                session_dir=self.ctx.root,
                threshold_seconds=self.settings.approval_seconds_threshold,
                python_soft_timeout_s=self.settings.python_soft_timeout_s,
            )
        except Exception:  # noqa: BLE001 - a broken heuristic must not block the run
            return None
        if request is None:
            return None

        await self._push(self._emit("approval_request", request.to_dict()))
        try:
            decision = await self.request_approval(request)
        except asyncio.CancelledError:
            # The stop button was pressed while the dialog was open. Answer the
            # call so the history stays valid, then let the turn end.
            await self._answer(
                call, f"Cancelled by the user while waiting for approval of {request.title}."
            )
            raise
        except Exception:  # noqa: BLE001 - an errored prompt is neither deny nor timeout
            decision = None

        if decision is None:
            await self._push(
                self._emit("approval_response", {"id": call.id, "approved": False, "timed_out": True})
            )
            await self._answer(
                call,
                f"No answer arrived for ({request.title}) — the approval dialog timed out "
                "or the connection dropped. This is NOT a denial: you may ask again, "
                "do something smaller, or wait for the user.",
            )
            return True
        await self._push(self._emit("approval_response", {"id": call.id, "approved": bool(decision)}))
        if decision:
            return True
        await self._answer(
            call,
            f"The user DENIED this action ({request.title}). Do not retry it as-is. "
            "Either do something smaller, or explain what you would do differently and ask.",
        )
        return False

    # -- execution -------------------------------------------------------
    async def _execute(self, handler, call: ToolCall) -> bool:
        self.calls_run += 1
        self.last_timed_out = False
        await self._push(self._emit("tool_start", tool_start_data(call)))
        started = time.monotonic()
        try:
            result = await handler(call.arguments, self.ctx)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a handler bug must not kill the turn
            result = ToolResult.fail(
                f"Tool '{call.name}' raised {type(exc).__name__}: {exc}\n"
                f"{traceback.format_exc(limit=8)}"
            )
        elapsed = time.monotonic() - started
        self.last_elapsed_s = elapsed
        data = result.data if isinstance(result.data, dict) else {}
        self.last_timed_out = bool(data.get("timed_out"))

        text = self.settings.redact(truncate(result.text, self.settings.max_tool_output_chars))
        try:
            self.state.save()
        except Exception as exc:  # noqa: BLE001 - disk/value errors must not kill the turn
            text += f"\n\n[warning: research-state save failed ({exc}); results above are still valid]"
        await self._push(
            self._emit(
                "tool_result",
                {
                    "id": call.id,
                    "name": call.name,
                    "text": text,
                    "error": result.error,
                    "images": list(result.images),
                    "data": jsonable(result.data),
                    "elapsed_s": round(elapsed, 2),
                },
            )
        )
        self._append(tool_message(call, text))
        if self.state.plan:
            await self._push(self._emit("state_update", self.state.summary_dict()))
        return not result.error

    async def _answer(self, call: ToolCall, text: str, *, error: bool = False) -> None:
        """Emit a tool result and record it in the message history."""
        await self._push(
            self._emit(
                "tool_result",
                {
                    "id": call.id,
                    "name": call.name,
                    "text": text,
                    "error": error,
                    "images": [],
                    "data": {},
                    "elapsed_s": 0.0,
                },
            )
        )
        self._append(tool_message(call, text))


def tool_start_data(call: ToolCall) -> dict[str, Any]:
    """The payload the UI needs to open a step for ``call``."""
    data: dict[str, Any] = {
        "id": call.id,
        "name": call.name,
        "arguments": jsonable(call.arguments),
    }
    if call.name == "python":
        data["code"] = str(call.arguments.get("code") or "")
        data["description"] = str(call.arguments.get("description") or "Run Python")
        data["language"] = "python"
    return data
