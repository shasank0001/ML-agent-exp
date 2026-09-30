"""The agent loop: streaming, tool dispatch, approval, repair, budgets, cancellation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datalab.agent import Agent
from datalab.config import Settings
from tests.fake_llm import PolicyLLM, RecordingApproval, RecordingAskUser, ScriptedLLM, call


def make_agent(settings: Settings, llm, tmp_path: Path, **kw) -> Agent:
    return Agent(
        session_id="s1",
        settings=settings,
        llm=llm,
        request_approval=kw.get("approval", RecordingApproval(default=True)),
        ask_user=kw.get("ask_user", RecordingAskUser(["churned"])),
    )


async def collect(agent: Agent, text: str = "go") -> list:
    return [event async for event in agent.run(text)]


def types_of(events: list) -> list[str]:
    return [e.type for e in events]


# -- basics -------------------------------------------------------------
async def test_text_only_turn_produces_a_message_and_done(settings: Settings, tmp_path: Path) -> None:
    llm = ScriptedLLM(turns=[("Here is the answer.", [])])
    agent = make_agent(settings, llm, tmp_path)
    events = await collect(agent)
    assert "assistant_message" in types_of(events)
    assert events[-1].type == "done"
    assert events[0].type == "user_message"


async def test_deltas_are_streamed_before_the_message(settings: Settings, tmp_path: Path) -> None:
    llm = ScriptedLLM(turns=[("abcdefghij" * 5, [])])
    agent = make_agent(settings, llm, tmp_path)
    events = await collect(agent)
    deltas = [e for e in events if e.type == "assistant_delta"]
    assert len(deltas) > 1  # really streamed, not one blob
    joined = "".join(e.data["text"] for e in deltas)
    assert joined == "abcdefghij" * 5


async def test_tool_call_is_dispatched_and_its_result_returned(settings: Settings, tmp_path: Path) -> None:
    llm = ScriptedLLM(turns=[("Listing.", [call("list_files", dir="outputs")]), ("Done.", [])])
    agent = make_agent(settings, llm, tmp_path)
    events = await collect(agent)
    starts = [e for e in events if e.type == "tool_start"]
    results = [e for e in events if e.type == "tool_result"]
    assert len(starts) == len(results) == 1
    assert starts[0].data["name"] == "list_files"
    assert results[0].data["error"] is False
    # the tool result must be fed back to the model
    tool_messages = [m for m in agent.messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == starts[0].data["id"]


async def test_multiple_tool_calls_in_one_turn_all_run(settings: Settings, tmp_path: Path) -> None:
    llm = ScriptedLLM(
        turns=[
            (
                "Two things.",
                [call("record_finding", text="one"), call("record_finding", text="two")],
            ),
            ("Done.", []),
        ]
    )
    agent = make_agent(settings, llm, tmp_path)
    events = await collect(agent)
    assert len([e for e in events if e.type == "tool_start"]) == 2
    assert agent.state.findings == ["one", "two"]


async def test_unknown_tool_is_reported_not_raised(settings: Settings, tmp_path: Path) -> None:
    llm = ScriptedLLM(turns=[("hmm", [call("do_magic", x=1)]), ("ok", [])])
    agent = make_agent(settings, llm, tmp_path)
    events = await collect(agent)
    results = [e for e in events if e.type == "tool_result"]
    assert results[0].data["error"] is True
    assert "Unknown tool" in results[0].data["text"]
    assert events[-1].type == "done"


# -- approvals ----------------------------------------------------------
async def test_approval_is_requested_and_approved(settings: Settings, tmp_path: Path) -> None:
    approval = RecordingApproval(default=True)
    llm = ScriptedLLM(
        turns=[("Training.", [call("python", code="x = 1", description="train", est_seconds=600)]),
               ("Done.", [])]
    )
    agent = make_agent(settings, llm, tmp_path, approval=approval)
    events = await collect(agent)
    assert len(approval.requests) == 1
    assert any(e.type == "approval_request" for e in events)
    assert any(e.type == "approval_response" and e.data["approved"] for e in events)
    result = [e for e in events if e.type == "tool_result"][0]
    assert result.data["error"] is False  # it actually ran


async def test_denial_is_graceful(settings: Settings, tmp_path: Path) -> None:
    approval = RecordingApproval(default=False)
    llm = ScriptedLLM(
        turns=[("Training.", [call("python", code="x = 1", description="train", est_seconds=600)]),
               ("Understood, I will do something smaller.", [])]
    )
    agent = make_agent(settings, llm, tmp_path, approval=approval)
    events = await collect(agent)
    assert any(e.type == "approval_response" and not e.data["approved"] for e in events)
    result = [e for e in events if e.type == "tool_result"][0]
    assert "DENIED" in result.data["text"]
    # the cell never ran
    assert "x" not in agent.executor.namespace
    # and the model was told, so it can adapt
    assert any(m.get("role") == "tool" and "DENIED" in str(m.get("content")) for m in agent.messages)


async def test_approval_shows_the_code(settings: Settings, tmp_path: Path) -> None:
    approval = RecordingApproval(default=True)
    code = "import shutil\nshutil.rmtree('build')"
    llm = ScriptedLLM(turns=[("Cleanup.", [call("python", code=code, description="clean")]), ("ok", [])])
    agent = make_agent(settings, llm, tmp_path, approval=approval)
    events = await collect(agent)
    request = [e for e in events if e.type == "approval_request"][0]
    assert request.data["code"] == code
    assert "rmtree" in request.data["reason"]


async def test_no_approval_for_a_cheap_call(settings: Settings, tmp_path: Path) -> None:
    approval = RecordingApproval(default=False)
    llm = ScriptedLLM(turns=[("Quick.", [call("python", code="y = 2", description="quick")]), ("ok", [])])
    agent = make_agent(settings, llm, tmp_path, approval=approval)
    await collect(agent)
    assert approval.requests == []


# -- repair loop and budgets -------------------------------------------
async def test_repeated_failures_stop_the_turn(settings: Settings, tmp_path: Path) -> None:
    bad = call("python", code="raise ValueError('nope')", description="broken")
    llm = ScriptedLLM(turns=[("Trying.", [bad])] * 6)
    agent = make_agent(settings, llm, tmp_path)
    events = await collect(agent)
    errors = [e for e in events if e.type == "error"]
    assert errors, "the loop should give up rather than spin"
    assert "failed attempts" in errors[-1].data["message"]
    assert events[-1].type == "done"


async def test_a_successful_call_resets_the_repair_counter(settings: Settings, tmp_path: Path) -> None:
    bad = call("python", code="raise ValueError('nope')", description="broken")
    good = call("python", code="ok = 1", description="fine")
    # never three failures in a row, so the turn must survive to the end
    pattern = [("t", [bad]), ("t", [bad]), ("t", [good]), ("t", [bad]), ("t", [bad]), ("t", [good])]
    llm = ScriptedLLM(turns=pattern + [("done", [])])
    agent = make_agent(settings, llm, tmp_path)
    events = await collect(agent)
    assert not [e for e in events if e.type == "error"]
    assert len([e for e in events if e.type == "tool_start"]) == 6


async def test_step_budget_stops_a_runaway_loop(settings: Settings, tmp_path: Path) -> None:
    never_ends = call("list_files", dir="outputs")
    llm = ScriptedLLM(turns=[("again", [never_ends])] * 500)
    agent = make_agent(settings, llm, tmp_path)
    events = await collect(agent)
    errors = [e for e in events if e.type == "error"]
    assert errors and "Step budget reached" in errors[-1].data["message"]
    assert agent._turn_steps == settings.max_steps


async def test_a_long_tool_output_is_truncated(settings: Settings, tmp_path: Path) -> None:
    long_code = "print('z' * 50000)"
    llm = ScriptedLLM(turns=[("noisy", [call("python", code=long_code, description="noisy")]), ("ok", [])])
    agent = make_agent(settings, llm, tmp_path)
    events = await collect(agent)
    result = [e for e in events if e.type == "tool_result"][0]
    assert len(result.data["text"]) <= settings.max_tool_output_chars + 400
    assert "truncated" in result.data["text"]
    # the full text is still on disk
    log = (agent.paths["events"]).read_text(encoding="utf-8")
    assert "truncated" in log  # the log holds the same (truncated) record


# -- state and artifacts ------------------------------------------------
async def test_state_is_saved_after_every_tool_call(settings: Settings, tmp_path: Path) -> None:
    llm = ScriptedLLM(turns=[("note", [call("record_finding", text="durable")]), ("ok", [])])
    agent = make_agent(settings, llm, tmp_path)
    await collect(agent)
    state_file = agent.paths["state"]
    assert state_file.exists()
    assert json.loads(state_file.read_text(encoding="utf-8"))["findings"] == ["durable"]


async def test_state_update_is_emitted_when_a_plan_exists(settings: Settings, tmp_path: Path) -> None:
    llm = ScriptedLLM(
        turns=[
            ("plan", [call("todo", items=[{"id": "1", "text": "a", "status": "pending"}])]),
            ("ok", []),
        ]
    )
    agent = make_agent(settings, llm, tmp_path)
    events = await collect(agent)
    updates = [e for e in events if e.type == "state_update"]
    assert updates
    assert updates[0].data["plan"][0]["text"] == "a"


async def test_events_jsonl_is_written_and_replayable(settings: Settings, tmp_path: Path) -> None:
    llm = ScriptedLLM(turns=[("hello", [call("list_files")]), ("bye", [])])
    agent = make_agent(settings, llm, tmp_path)
    events = await collect(agent)
    lines = agent.logger.read_all()
    assert lines, "events.jsonl must not be empty"
    seqs = [line["seq"] for line in lines]
    assert seqs == sorted(seqs) and seqs[0] == 1
    types = [line["type"] for line in lines]
    assert "assistant_delta" not in types  # deltas are not persisted
    assert "user_message" in types and "done" in types
    assert all(line["session_id"] == "s1" for line in lines)


async def test_usage_is_reported_on_done(settings: Settings, tmp_path: Path) -> None:
    llm = ScriptedLLM(turns=[("hi", [])])
    agent = make_agent(settings, llm, tmp_path)
    events = await collect(agent)
    done = events[-1]
    assert done.data["usage"]["turn"]["total_tokens"] > 0
    assert done.data["usage"]["session"]["total_tokens"] == done.data["usage"]["turn"]["total_tokens"]


# -- provider failure ---------------------------------------------------
async def test_provider_error_is_surfaced_not_raised(settings: Settings, tmp_path: Path) -> None:
    class Broken:
        model = "broken"

        async def stream(self, messages, tools=None):
            raise RuntimeError("connection refused")
            yield  # pragma: no cover

    agent = make_agent(settings, Broken(), tmp_path)
    events = await collect(agent)
    errors = [e for e in events if e.type == "error"]
    assert errors and "connection refused" in errors[0].data["message"]
    assert events[-1].type == "done"


# -- attachments --------------------------------------------------------
async def test_attachments_are_copied_and_announced(settings: Settings, tmp_path: Path, messy_csv: Path) -> None:
    llm = ScriptedLLM(turns=[("ok", [])])
    agent = make_agent(settings, llm, tmp_path)
    copied = agent.add_attachment(messy_csv)
    assert copied.parent == agent.paths["data"]
    assert copied.exists()
    events = await collect(agent, "build a model")
    user = [e for e in events if e.type == "user_message"][0]
    assert "customers.csv" in user.data["attachments"][0]
    assert "DATA_DIR" in str(agent.messages[0]["content"])


async def test_repeated_upload_of_same_name_gets_a_suffix(settings: Settings, tmp_path: Path, messy_csv: Path) -> None:
    agent = make_agent(settings, ScriptedLLM(turns=[]), tmp_path)
    first = agent.add_attachment(messy_csv)
    second = agent.add_attachment(messy_csv)
    assert first.name == "customers.csv"
    assert second.name == "customers_1.csv"


# -- cancellation -------------------------------------------------------
async def test_cancel_stops_the_turn(settings: Settings, tmp_path: Path) -> None:
    import asyncio

    class Slow:
        model = "slow"

        async def stream(self, messages, tools=None):
            for _ in range(200):
                await asyncio.sleep(0.05)
                yield ("text", "tick")
            yield ("tool_calls", [])
            yield ("usage", None)

    agent = make_agent(settings, Slow(), tmp_path)
    events = []
    async def drive():
        async for event in agent.run("go"):
            events.append(event)
            if len(events) == 3:
                agent.cancel()

    await asyncio.wait_for(drive(), timeout=5)
    assert not agent._task or agent._task.done()


# -- the golden path, offline -------------------------------------------
async def test_golden_path_end_to_end(settings: Settings, tmp_path: Path, messy_csv: Path) -> None:
    """profile -> ask -> plan -> split -> baseline -> two models -> results."""
    llm = PolicyLLM()
    ask = RecordingAskUser(["churned"])
    agent = make_agent(settings, llm, tmp_path, ask_user=ask)
    agent.add_attachment(messy_csv)

    events = await collect(agent, "build the best model to predict `churned`")
    names = [e.data.get("name") for e in events if e.type == "tool_start"]
    assert "profile_dataset" in names
    assert "ask_user" in names
    assert "todo" in names
    assert names.count("python") >= 4
    assert events[-1].type == "done"
    assert not [e for e in events if e.type == "error"]

    # ground truth: the state file has the experiments
    state = json.loads(agent.paths["state"].read_text(encoding="utf-8"))
    assert state["task"]["target"] == "churned"
    assert state["task"]["task_type"] == "classification"
    assert state["dataset"]["n_rows"] == 603
    assert len(state["experiments"]) >= 4
    assert {e["name"] for e in state["experiments"]} >= {
        "baseline_dummy",
        "baseline_logreg",
        "random_forest",
        "lightgbm",
    }
    for experiment in state["experiments"]:
        assert "f1_macro" in experiment["metrics"]
        assert 0.0 <= experiment["metrics"]["f1_macro"] <= 1.0
    assert state["findings"]
    assert state["plan"]

    # the log agrees with the state
    log_types = [e["type"] for e in agent.logger.read_all()]
    assert log_types.count("tool_result") == len(names)
