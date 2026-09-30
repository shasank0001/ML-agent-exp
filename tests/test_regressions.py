"""Regression tests for the bugs found in the correctness review.

Each test here corresponds to a specific defect that shipped and was fixed.
They are deliberately narrow: if one fails, the named bug is back.
"""

from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path
from typing import Any

import pytest

from datalab.agent import Agent
from datalab.conversation import safe_history_start
from datalab.config import Settings
from datalab.llm import LLMClient, ToolCall, Usage
from datalab.tools.python_exec import PythonExecutor, _restore_streams
from tests.fake_llm import RecordingApproval, RecordingAskUser, ScriptedLLM, call


def make(settings: Settings, llm: Any, approval=None, ask=None) -> Agent:
    return Agent(
        session_id="s1",
        settings=settings,
        llm=llm,
        request_approval=approval or RecordingApproval(default=True),
        ask_user=ask or RecordingAskUser([]),
    )


async def collect(agent: Agent, text: str = "go") -> list:
    return [e async for e in agent.run(text)]


def history_is_wellformed(agent: Agent) -> bool:
    """Every assistant tool_call must be followed by a matching tool response."""
    answered = {m.get("tool_call_id") for m in agent.messages if m.get("role") == "tool"}
    return all(
        c["id"] in answered
        for m in agent.messages
        if m.get("role") == "assistant"
        for c in (m.get("tool_calls") or [])
    )


# --- bug 2: cancelling during approval orphaned a tool_call -------------
async def test_cancel_during_approval_leaves_valid_history(settings: Settings) -> None:
    """Stopping the run while an approval dialog is open must not poison the API."""
    class BlockingApproval:
        """Stands in for a dialog the user walks away from, then hits stop."""

        def __init__(self) -> None:
            self.opened = asyncio.Event()

        async def __call__(self, request) -> bool:
            self.opened.set()
            await asyncio.sleep(3600)
            return True

    approval = BlockingApproval()
    slow = call("python", code="ran = True", description="train 40 models", est_seconds=900)
    agent = make(settings, ScriptedLLM(turns=[("t", [slow]), ("done", [])]), approval=approval)

    events = []
    async def drive():
        async for event in agent.run("go"):
            events.append(event)

    task = asyncio.create_task(drive())
    await asyncio.wait_for(approval.opened.wait(), timeout=5)
    agent.cancel()
    await asyncio.wait_for(task, timeout=5)

    assert history_is_wellformed(agent), agent.messages
    # the cell never ran, and the model was told so rather than being left hanging
    assert "ran" not in agent.executor.namespace
    tool_results = [e for e in events if e.type == "tool_result"]
    assert tool_results and "ancel" in tool_results[0].data["text"]
    assert [e for e in events if e.type == "done"]


async def test_cancel_mid_tool_sequence_answers_remaining_calls(settings: Settings) -> None:
    instant = call("record_finding", text="fast")
    agent = make(settings, ScriptedLLM(turns=[("t", [instant, instant, instant]), ("done", [])]))

    seen = 0
    async for event in agent.run("go"):
        seen += 1
        if event.type == "tool_result":
            agent.cancel()
    assert history_is_wellformed(agent), agent.messages


# --- bug 7: the stop button was ignored between instant tool calls -----
async def test_cancel_between_instant_tools_ends_the_turn_early(settings: Settings) -> None:
    """A turn of fast tools must still observe the stop button."""
    llm = ScriptedLLM(turns=[("t", [call("list_files", dir="outputs")])] * 200)
    agent = make(settings, llm)

    steps_seen = 0
    async for event in agent.run("go"):
        if event.type == "tool_result":
            steps_seen += 1
            if steps_seen == 1:
                agent.cancel()
    assert steps_seen <= 2, f"kept running after cancel ({steps_seen} tool results)"
    assert history_is_wellformed(agent)


# --- bug 6: trimming could split a tool_call / tool pair ---------------
def test_safe_history_start_avoids_orphaning_a_tool_call() -> None:
    messages = (
        [{"role": "user", "content": "a"}]
        + [{"role": "assistant", "tool_calls": [{"id": "1", "type": "function",
                                                 "function": {"name": "x", "arguments": "{}"}}]}]
        + [{"role": "tool", "tool_call_id": "1", "content": "ok"}]
        + [{"role": "assistant", "content": "done"}]
    )
    for wanted in range(len(messages) + 1):
        start = safe_history_start(messages, wanted)
        kept = messages[start:]
        answered = {m["tool_call_id"] for m in kept if m.get("role") == "tool"}
        for m in kept:
            for c in m.get("tool_calls") or []:
                assert c["id"] in answered, f"cut at {wanted} -> {start} orphaned {c['id']}"


async def test_long_session_keeps_history_valid(settings: Settings) -> None:
    llm = ScriptedLLM(turns=[("t", [call("record_finding", text="x")])] * 400)
    agent = make(settings, llm)
    for _ in range(6):
        await collect(agent, "go")
    assert history_is_wellformed(agent)
    assert len(agent.messages) <= settings.context_window_messages * 3 + 8


# --- bug 4: tool-call deltas without an index were merged --------------
class _Delta:
    def __init__(self, index=None, id=None, name=None, arguments=None):
        self.index, self.id = index, id
        self.function = type("F", (), {"name": name, "arguments": arguments})()


class _Chunk:
    def __init__(self, tool_calls):
        choice = type("C", (), {"delta": type("D", (), {"content": None, "tool_calls": tool_calls})()})()
        self.choices = [choice]
        self.usage = None


async def test_indexless_tool_call_deltas_stay_separate(settings: Settings) -> None:
    """Providers that omit `index` must not merge two calls into one."""
    chunks = [
        _Chunk([_Delta(id="call_a", name="first", arguments='{"x":')]),
        _Chunk([_Delta(id="call_a", arguments="1}")]),
        _Chunk([_Delta(id="call_b", name="second", arguments='{"y":2}')]),
    ]
    client = LLMClient(settings, client=_fake_client(chunks))
    calls: list[ToolCall] = []
    async for kind, payload in client.stream([{"role": "user", "content": "hi"}], None):
        if kind == "tool_calls":
            calls = payload
    assert len(calls) == 2, calls
    assert {c.name for c in calls} == {"first", "second"}
    first = next(c for c in calls if c.name == "first")
    assert first.arguments == {"x": 1}


async def test_indexed_tool_call_deltas_still_merge_by_index(settings: Settings) -> None:
    chunks = [
        _Chunk([_Delta(index=0, id="call_a", name="only", arguments='{"x":')]),
        _Chunk([_Delta(index=0, arguments="1}")]),
    ]
    client = LLMClient(settings, client=_fake_client(chunks))
    calls = []
    async for kind, payload in client.stream([{"role": "user", "content": "hi"}], None):
        if kind == "tool_calls":
            calls = payload
    assert len(calls) == 1
    assert calls[0].arguments == {"x": 1}


class _FakeCompletions:
    def __init__(self, chunks):
        self._chunks = chunks

    async def create(self, **kwargs):
        async def gen():
            for chunk in self._chunks:
                yield chunk

        return gen()


def _fake_client(chunks):
    completions = _FakeCompletions(chunks)
    chat = type("Chat", (), {"completions": completions})()
    return type("Client", (), {"chat": chat})()


# --- bug 5: a timed-out cell hijacked the process stdout ---------------
def test_restore_streams_recovers_a_hijacked_stdout() -> None:
    real = sys.stdout
    try:
        sys.stdout = io.StringIO()  # as a timed-out cell leaves it
        _restore_streams(real, sys.stderr)
        assert sys.stdout is real
    finally:
        sys.stdout = real


def test_restore_streams_leaves_healthy_streams_alone() -> None:
    real = sys.stdout
    _restore_streams(real, sys.stderr)
    assert sys.stdout is real


async def test_executor_restores_stdout_after_a_timeout(tmp_path: Path) -> None:
    from datalab.lab import LabSession
    from datalab.state import ResearchState

    state = ResearchState(root_dir=str(tmp_path / "s"))
    lab = LabSession(state, data_dir=tmp_path / "d", output_dir=tmp_path / "o")
    ex = PythonExecutor(
        session_dir=tmp_path / "s", data_dir=tmp_path / "d",
        output_dir=tmp_path / "o", figures_dir=tmp_path / "f", lab=lab,
    )
    before = sys.stdout
    result = await ex.run("import time\ntime.sleep(2)", timeout_s=1)
    assert result.error
    assert not isinstance(sys.stdout, io.StringIO)


# --- bug 1: bare relative writes escaped the session folder -------------
async def test_bare_relative_write_lands_in_the_session(tmp_path: Path) -> None:
    from datalab.lab import LabSession
    from datalab.state import ResearchState

    state = ResearchState(root_dir=str(tmp_path / "s"))
    lab = LabSession(state, data_dir=tmp_path / "d", output_dir=tmp_path / "o")
    ex = PythonExecutor(
        session_dir=tmp_path / "s", data_dir=tmp_path / "d",
        output_dir=tmp_path / "o", figures_dir=tmp_path / "f", lab=lab,
    )
    cwd = Path.cwd()
    await ex.run("open('escaped.txt', 'w').write('x')")
    assert (tmp_path / "s" / "escaped.txt").exists()
    assert not (cwd / "escaped.txt").exists()
    assert Path.cwd() == cwd


# --- bug 10: ask_user events were never emitted ------------------------
async def test_ask_user_emits_events(settings: Settings) -> None:
    llm = ScriptedLLM(
        turns=[("q", [call("ask_user", question="which target?", options=["a", "b"])]), ("ok", [])]
    )
    agent = make(settings, llm, ask=RecordingAskUser(["a"]))
    events = await collect(agent)
    asked = [e for e in events if e.type == "ask_user"]
    answered = [e for e in events if e.type == "ask_user_response"]
    assert asked and asked[0].data["question"] == "which target?"
    assert asked[0].data["options"] == ["a", "b"]
    assert answered and answered[0].data["answer"] == "a"
    logged = [e["type"] for e in agent.logger.read_all()]
    assert "ask_user" in logged and "ask_user_response" in logged


# --- bug 9: the cancellation event was emitted but never delivered ----
async def test_cancellation_error_event_reaches_the_consumer(settings: Settings) -> None:
    class Endless:
        model = "endless"

        async def stream(self, messages, tools=None):
            for _ in range(500):
                await asyncio.sleep(0.02)
                yield ("text", "tick")
            yield ("tool_calls", [])
            yield ("usage", Usage())

    agent = make(settings, Endless())
    events = []
    async for event in agent.run("go"):
        events.append(event)
        if len(events) == 4:
            agent.cancel()
    assert any(e.type == "done" for e in events)
    done = [e for e in events if e.type == "done"][-1]
    assert done.data["cancelled"] is True