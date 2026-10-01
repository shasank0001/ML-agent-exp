"""The agent loop.

UI-agnostic: it never imports Chainlit. It takes two async callbacks from the
UI (`request_approval` and `ask_user`) and yields :class:`~datalab.events.Event`
objects, each of which is also appended to the session's JSONL log.
"""

from __future__ import annotations

import asyncio
import traceback
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

from .approvals import ApprovalRequest
from .config import Settings, ensure_session_dirs
from .conversation import (
    answer_orphans,
    answer_skipped,
    assistant_tool_calls,
    tool_message,
    trim,
    user_message,
    window,
)
from .events import Event, EventLogger, redact_data
from .lab import LabSession
from .llm import LLMClient, SelfTestResult, ToolCall, is_transient
from .prompts import build_system_prompt
from .state import ResearchState
from .tool_dispatch import ToolDispatcher
from .tools import TOOL_SCHEMAS, ToolContext
from .tools.files import copy_upload
from .tools.python_exec import PythonExecutor

#: How many times a step that failed *before* showing any output is re-issued.
TRANSIENT_STEP_RETRIES = 2

ApprovalFn = Callable[[ApprovalRequest], Awaitable[bool]]
AskUserFn = Callable[[str, list[str]], Awaitable[str]]


class AgentError(RuntimeError):
    """Fatal problem with the runtime configuration."""


class Agent:
    """One agent per chat session: loop, tools, executor and research state."""

    def __init__(
        self,
        *,
        session_id: str,
        settings: Settings,
        llm: LLMClient,
        request_approval: ApprovalFn,
        ask_user: AskUserFn,
        attachments: list[Path] | None = None,
    ) -> None:
        self.session_id = session_id
        self.settings = settings
        self.llm = llm
        self.request_approval = request_approval
        self.ask_user_cb = ask_user

        self.paths = ensure_session_dirs(settings.runs_dir, session_id)
        self.logger = EventLogger(session_id, self.paths["events"], secrets=settings.secrets)
        self.state = ResearchState(session_id=session_id, root_dir=str(self.paths["root"]))
        self.state.save()

        self.lab = LabSession(
            self.state, data_dir=self.paths["data"], output_dir=self.paths["outputs"]
        )
        self.executor = PythonExecutor(
            session_dir=self.paths["root"],
            data_dir=self.paths["data"],
            output_dir=self.paths["outputs"],
            figures_dir=self.paths["figures"],
            lab=self.lab,
        )
        self.ctx = ToolContext(
            session_id=session_id,
            paths=self.paths,
            state=self.state,
            settings=settings,
            logger=self.logger,
            executor=self.executor,
            agent=self,
        )
        self.attachments: list[Path] = list(attachments or [])
        self.messages: list[dict[str, Any]] = []
        self.tool_call_count = 0
        self.usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._turn_steps = 0
        self._turn_tool_calls = 0
        self._known_calls: list[ToolCall] = []
        self._queue: asyncio.Queue | None = None
        self._cancelled = False
        self._self_test: SelfTestResult | None = None
        self._task: asyncio.Task | None = None

        self.dispatcher = ToolDispatcher(
            ctx=self.ctx,
            state=self.state,
            settings=settings,
            emit=self._emit,
            push=self._push_event,
            append_message=self._append_message,
            record_call=self._record_call,
            request_approval=self.request_approval,
        )

    # -- public API -----------------------------------------------------
    @property
    def self_test(self) -> SelfTestResult | None:
        return self._self_test

    async def run_self_test(self) -> SelfTestResult:
        """Probe the configured model's tool-calling support once per session."""
        if not self.settings.is_configured:
            self._self_test = SelfTestResult(
                False, self.settings.model or "(unset)", "; ".join(self.settings.missing_pieces())
            )
            return self._self_test
        self._self_test = await self.llm.self_test()
        return self._self_test

    def add_attachment(self, path: Path) -> Path:
        """Copy an uploaded file into the session's data folder and remember it."""
        copied = copy_upload(Path(path), self.ctx)
        if copied not in self.attachments:
            self.attachments.append(copied)
        return copied

    def cancel(self) -> None:
        """Request cancellation of the in-flight run (Chainlit's stop button)."""
        self._cancelled = True
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def ask_user(self, question: str, options: list[str] | None = None) -> str:
        """Ask the user through the UI, emitting the round trip as events."""
        options = list(options or [])
        await self._maybe_push("ask_user", {"question": question, "options": options})
        answer = await self.ask_user_cb(question, options)
        await self._maybe_push("ask_user_response", {"question": question, "answer": answer})
        return answer

    async def _maybe_push(self, type_: str, data: dict[str, Any]) -> None:
        """Emit an event outside a turn (e.g. a question asked from a tool)."""
        if self._queue is None:
            return
        await self._queue.put(self._emit(type_, data))

    async def run(self, user_text: str) -> AsyncIterator[Event]:
        """Run one turn: stream, call tools, repeat until the model stops."""
        if self._task is not None and not self._task.done():
            raise AgentError("a turn is already running in this session")
        self._cancelled = False
        self._turn_steps = 0
        self._turn_tool_calls = 0
        self._known_calls = []
        usage_before = dict(self.usage_total)
        done_emitted = False
        try:
            yield self._emit(
                "user_message",
                {"text": user_text, "attachments": [str(p) for p in self.attachments]},
            )
            self.messages.append({"role": "user", "content": self._user_content(user_text)})
            async for event in self._loop():
                yield event
            yield self._emit("done", self._done_payload(usage_before))
            done_emitted = True
        finally:
            if not done_emitted:
                # The consumer walked away (stop button, closed tab). The log must
                # still end with a `done` record so replay does not look truncated.
                self._emit("done", self._done_payload(usage_before))

    def _done_payload(self, usage_before: dict[str, int]) -> dict[str, Any]:
        return {
            "steps": self._turn_steps,
            "tool_calls": self._turn_tool_calls,
            "cancelled": self._cancelled,
            "usage": {
                "turn": {k: self.usage_total[k] - usage_before.get(k, 0) for k in self.usage_total},
                "session": dict(self.usage_total),
            },
        }

    # -- the loop -------------------------------------------------------
    async def _loop(self) -> AsyncIterator[Event]:
        """Bridge the driver task to a generator, so `cancel()` can stop the turn."""
        queue: asyncio.Queue[Event | None] = asyncio.Queue()
        self._turn_steps = 0
        runner = asyncio.create_task(self._drive(queue))
        self._task = runner
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            if not runner.done():
                runner.cancel()
                try:
                    await runner
                except asyncio.CancelledError:
                    pass
            self._task = None

    async def _drive(self, queue: asyncio.Queue) -> None:
        """The actual turn. Pushes events into ``queue``; always ends with None."""
        self._queue = queue
        try:
            repairs = 0
            for _step in range(self.settings.max_steps):
                self._turn_steps = _step + 1
                if self._cancelled:
                    await self._push(queue, self._emit("error", {"message": "Run cancelled."}))
                    return
                text, tool_calls = await self._stream_step(queue)
                if text is None:
                    return

                if text:
                    self.messages.append({"role": "assistant", "content": text})
                    await self._push(queue, self._emit("assistant_message", {"text": text}))
                if not tool_calls:
                    return

                self.messages.append(assistant_tool_calls(text, tool_calls))
                for position, call in enumerate(tool_calls):
                    if self._cancelled:
                        # The OpenAI API rejects a history where an assistant
                        # tool_call has no matching tool response, so answer the
                        # ones we are skipping rather than dropping them.
                        answer_skipped(self.messages, tool_calls[position:])
                        await self._push(queue, self._emit("error", {"message": "Run cancelled."}))
                        return
                    self._turn_tool_calls += 1
                    ok = await self.dispatcher.run(call)
                    if getattr(self.dispatcher, "last_timed_out", False):
                        # Slow != broken: a timed-out training cell must not burn
                        # the repair budget. The timeout text already tells the
                        # model to split/shrink the work.
                        repairs = 0
                    else:
                        repairs = 0 if ok else repairs + 1
                if repairs > self.settings.max_repairs:
                    await self._push(
                        queue,
                        self._emit(
                            "error",
                            {
                                "message": (
                                    f"Too many failed attempts ({repairs} in a row, limit "
                                    f"{self.settings.max_repairs}). Stopping so you can give guidance."
                                ),
                                "needs_user": True,
                            },
                        ),
                    )
                    return
            await self._push(
                queue,
                self._emit(
                    "error",
                    {
                        "message": (
                            f"Step budget reached ({self.settings.max_steps} LLM turns for this "
                            "message). Stopping here — ask me to continue if you were mid-task."
                        ),
                        "needs_user": True,
                    },
                ),
            )
        except asyncio.CancelledError:
            self._answer_orphans("Cancelled: the run was stopped before this finished.")
            try:
                queue.put_nowait(self._emit("error", {"message": "Run cancelled."}))
            except Exception:  # noqa: BLE001 - the queue may already be closed
                pass
            return
        finally:
            self._answer_orphans("Not run: the turn ended before this tool could start.")
            self._queue = None
            self._known_calls = []
            self._trim_context()
            try:
                queue.put_nowait(None)
            except Exception:  # noqa: BLE001 - consumer already gone
                pass

    async def _stream_step(
        self, queue: asyncio.Queue
    ) -> tuple[str | None, list[ToolCall]]:
        """Run one LLM turn, retrying a transient failure that showed nothing.

        Returns ``(text, tool_calls)``, or ``(None, [])`` when the provider is
        genuinely unreachable and the turn must end. A stream that broke after
        it had already started showing output is not retried — replaying it
        would duplicate text the user has read.
        """
        last: Exception | None = None
        for attempt in range(TRANSIENT_STEP_RETRIES + 1):
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            try:
                async for kind, payload in self.llm.stream(self._context(), TOOL_SCHEMAS):
                    if kind == "text":
                        text_parts.append(payload)
                        await self._push(queue, self._emit("assistant_delta", {"text": payload}))
                    elif kind == "tool_calls":
                        tool_calls = payload
                    elif kind == "usage":
                        self._add_usage(payload.to_dict())
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - provider/network failure
                last = exc
                if text_parts or not is_transient(exc) or attempt == TRANSIENT_STEP_RETRIES:
                    break
                await self._push(
                    queue,
                    self._emit(
                        "warning",
                        {
                            "message": (
                                f"The model connection blipped ({type(exc).__name__}); "
                                f"retrying ({attempt + 1}/{TRANSIENT_STEP_RETRIES})."
                            )
                        },
                    ),
                )
                continue
            return "".join(text_parts).strip(), tool_calls

        await self._push(
            queue,
            self._emit(
                "error",
                {
                    "message": (
                        f"The model call was interrupted: {type(last).__name__}: {last}. "
                        "Everything computed so far is in the research state — send another "
                        "message and I will pick it up from there."
                    ),
                    "fatal": not text_parts,
                },
            ),
        )
        return None, []

    # -- history repair --------------------------------------------------
    def _answer_orphans(self, reason: str | None = None) -> None:
        """Guarantee the history can be sent again.

        An assistant message carrying ``tool_calls`` must be followed by one
        tool message per call. Several paths can end a turn between the two
        (cancellation, the step budget, a provider error), so rather than
        policing each one, the invariant is enforced once at the end of every
        turn.
        """
        answer_orphans(
            self.messages,
            {c.id: c for c in self._known_calls},
            reason or "Not run: the turn ended before this tool could start.",
        )

    # -- message plumbing -----------------------------------------------
    def _append_message(self, message: dict[str, Any]) -> None:
        """Append to the live history (stable across `_trim_context` rebinds)."""
        self.messages.append(message)

    def _record_call(self, call: ToolCall) -> None:
        """Track a call on the live list (stable across per-turn rebinds)."""
        self._known_calls.append(call)

    def _tool_message(self, call: ToolCall, text: str) -> dict[str, Any]:
        return tool_message(call, text)

    def _user_content(self, user_text: str) -> str:
        return user_message(user_text, [p.name for p in self.attachments])

    def _context(self) -> list[dict[str, Any]]:
        """System prompt (with a fresh state summary) plus the tail of the history."""
        system = build_system_prompt(
            self.state.summary_text(self.settings.max_state_summary_chars)
        )
        return window(self.messages, system, self.settings.context_window_messages)

    def _trim_context(self) -> None:
        """Keep the in-process history bounded; the full transcript stays in the log."""
        self.messages = trim(
            self.messages, self.settings.context_window_messages * 3, head=2
        )

    def _add_usage(self, usage: dict[str, int]) -> None:
        for key in self.usage_total:
            self.usage_total[key] += int(usage.get(key, 0) or 0)

    def _emit(self, type_: str, data: dict[str, Any]) -> Event:
        return self.logger.emit(type_, redact_data(data, self.settings.secrets))

    async def _push(self, queue: asyncio.Queue, event: Event) -> None:
        """Hand an event to the consumer, then give the loop a turn.

        ``Queue.put`` on an unbounded queue never suspends, so a turn made only
        of fast tools would run to the step budget without the UI ever getting
        a slot -- and the stop button would have nothing to interrupt.
        """
        await queue.put(event)
        await asyncio.sleep(0)

    async def _push_event(self, event: Event) -> None:
        """Put an event on the running turn's queue, if a turn is in flight."""
        if self._queue is None:
            return
        await self._queue.put(event)
        await asyncio.sleep(0)
