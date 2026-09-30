"""The agent loop.

UI-agnostic: it never imports Chainlit. It takes two async callbacks from the
UI (`request_approval` and `ask_user`) and yields :class:`~datalab.events.Event`
objects, each of which is also appended to the session's JSONL log.
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

from .approvals import ApprovalRequest, needs_approval
from .config import Settings, ensure_session_dirs
from .events import Event, EventLogger, redact_data
from .lab import LabSession
from .llm import LLMClient, SelfTestResult, ToolCall
from .prompts import build_system_prompt
from .state import ResearchState
from .tools import REGISTRY, TOOL_SCHEMAS, ToolContext, ToolResult
from .tools.files import copy_upload
from .tools.python_exec import PythonExecutor

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
        self._cancelled = False
        self._self_test: SelfTestResult | None = None
        self._task: asyncio.Task | None = None

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
        return await self.ask_user_cb(question, list(options or []))

    async def run(self, user_text: str) -> AsyncIterator[Event]:
        """Run one turn: stream, call tools, repeat until the model stops."""
        self._cancelled = False
        self._turn_steps = 0
        self._turn_tool_calls = 0
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
        try:
            repairs = 0
            for _step in range(self.settings.max_steps):
                self._turn_steps = _step + 1
                if self._cancelled:
                    await self._push(queue, self._emit("error", {"message": "Run cancelled."}))
                    return
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
                    await self._push(
                        queue,
                        self._emit(
                            "error",
                            {
                                "message": f"The model call failed: {type(exc).__name__}: {exc}",
                                "fatal": True,
                            },
                        ),
                    )
                    return

                text = "".join(text_parts).strip()
                if text:
                    self.messages.append({"role": "assistant", "content": text})
                    await self._push(queue, self._emit("assistant_message", {"text": text}))
                if not tool_calls:
                    return

                self.messages.append(
                    {
                        "role": "assistant",
                        "content": text or None,
                        "tool_calls": [c.to_message() for c in tool_calls],
                    }
                )
                for call in tool_calls:
                    outcome = await self._run_one_tool(queue, call)
                    if outcome is not None:
                        repairs = 0 if outcome else repairs + 1
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
            self._emit("error", {"message": "Run cancelled."})
            return
        finally:
            self._trim_context()
            await queue.put(None)

    # -- one tool call --------------------------------------------------
    async def _run_one_tool(self, queue: asyncio.Queue, call: ToolCall) -> bool | None:
        """Run one tool call. Returns whether it succeeded, or ``None`` if it was not run.

        Errors are never raised at the caller: they come back to the model as
        the tool result so the repair loop can act on the traceback.
        """
        spec = REGISTRY.get(call.name)
        if spec is None:
            text = (
                f"Unknown tool '{call.name}'. Available tools: {', '.join(sorted(REGISTRY))}. "
                "Call one of those instead."
            )
            self.messages.append(self._tool_message(call, text))
            await self._push(
                queue,
                self._emit(
                    "tool_result",
                    {"id": call.id, "name": call.name, "text": text, "error": True, "images": []},
                ),
            )
            return False

        request: ApprovalRequest | None = None
        try:
            request = needs_approval(
                call,
                self.state,
                session_dir=self.paths["root"],
                threshold_seconds=self.settings.approval_seconds_threshold,
            )
        except Exception:  # noqa: BLE001 - a broken heuristic must not block the run
            request = None

        if request is not None:
            await self._push(queue, self._emit("approval_request", request.to_dict()))
            try:
                approved = bool(await self.request_approval(request))
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - treat an errored prompt as "denied"
                approved = False
            await self._push(
                queue, self._emit("approval_response", {"id": call.id, "approved": approved})
            )
            if not approved:
                text = (
                    f"The user DENIED this action ({request.title}). Do not retry it as-is. "
                    "Either do something smaller, or explain what you would do differently and ask."
                )
                self.messages.append(self._tool_message(call, text))
                await self._push(
                    queue,
                    self._emit(
                        "tool_result",
                        {
                            "id": call.id,
                            "name": call.name,
                            "text": text,
                            "error": False,
                            "images": [],
                        },
                    ),
                )
                return True

        self.tool_call_count += 1
        self._turn_tool_calls += 1
        await self._push(queue, self._emit("tool_start", self._tool_start_data(call)))
        started = time.monotonic()
        try:
            result = await spec.handler(call.arguments, self.ctx)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a handler bug must not kill the turn
            result = ToolResult.fail(
                f"Tool '{call.name}' raised {type(exc).__name__}: {exc}\n"
                f"{traceback.format_exc(limit=8)}"
            )
        elapsed = time.monotonic() - started

        text = self.settings.redact(self._truncate(result.text))
        data = {
            "id": call.id,
            "name": call.name,
            "text": text,
            "error": result.error,
            "images": list(result.images),
            "data": _jsonable(result.data),
            "elapsed_s": round(elapsed, 2),
        }
        self.state.save()
        await self._push(queue, self._emit("tool_result", data))
        self.messages.append(self._tool_message(call, text))
        if self.state.plan:
            await self._push(queue, self._emit("state_update", self.state.summary_dict()))
        return not result.error

    def _tool_start_data(self, call: ToolCall) -> dict[str, Any]:
        data: dict[str, Any] = {"id": call.id, "name": call.name, "arguments": _jsonable(call.arguments)}
        if call.name == "python":
            data["code"] = str(call.arguments.get("code") or "")
            data["description"] = str(call.arguments.get("description") or "Run Python")
            data["language"] = "python"
        return data

    # -- message plumbing -----------------------------------------------
    def _tool_message(self, call: ToolCall, text: str) -> dict[str, Any]:
        return {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": text}

    def _user_content(self, user_text: str) -> str:
        """User text plus a note about which files are available in the session."""
        if not self.attachments:
            return user_text
        names = ", ".join(p.name for p in self.attachments)
        return (
            f"{user_text}\n\n[files available in DATA_DIR for this session: {names} — "
            "load one with lab.load(<name>) or pd.read_csv(DATA_DIR + '/<name>')]"
        )

    def _context(self) -> list[dict[str, Any]]:
        """System prompt (with a fresh state summary) plus the tail of the history."""
        system = build_system_prompt(self.state.summary_text(self.settings.max_state_summary_chars))
        window = self.messages[-self.settings.context_window_messages :]
        return [{"role": "system", "content": system}, *window]

    def _trim_context(self) -> None:
        """Keep the in-process history bounded; the full transcript stays in the log."""
        keep = self.settings.context_window_messages * 3
        if len(self.messages) > keep:
            head = self.messages[:2]  # the first user turn is useful context
            self.messages = head + self.messages[-(keep - len(head)) :]

    def _truncate(self, text: str) -> str:
        limit = self.settings.max_tool_output_chars
        if len(text) <= limit:
            return text
        head = int(limit * 0.6)
        tail = limit - head
        return (
            text[:head]
            + f"\n\n[... {len(text) - limit} characters truncated from the middle. "
            f"The full output is in the event log. ...]\n\n"
            + text[-tail:]
        )

    def _add_usage(self, usage: dict[str, int]) -> None:
        for key in self.usage_total:
            self.usage_total[key] += int(usage.get(key, 0) or 0)

    def _emit(self, type_: str, data: dict[str, Any]) -> Event:
        return self.logger.emit(type_, redact_data(data, self.settings.secrets))

    async def _push(self, queue: asyncio.Queue, event: Event) -> None:
        await queue.put(event)


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of tool payloads to something json.dumps accepts."""
    try:
        json.dumps(value, default=str)
        return value
    except (TypeError, ValueError):
        return json.loads(json.dumps(value, default=str))
