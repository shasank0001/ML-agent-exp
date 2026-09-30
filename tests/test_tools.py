"""Tool registry, JSON-schema validity, and the file / state / todo handlers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from datalab.events import Event, EventLogger, redact_data
from datalab.state import ResearchState, TaskSpec
from datalab.tools import REGISTRY, TOOL_SCHEMAS, ToolContext
from datalab.tools import ask_user as ask_user_tool
from datalab.tools import files as files_tool
from datalab.tools import state_tools, todo


# -- schema validity ----------------------------------------------------
EXPECTED = {
    "python",
    "profile_dataset",
    "read_file",
    "write_file",
    "list_files",
    "todo",
    "ask_user",
    "record_finding",
    "query_state",
}


def test_every_expected_tool_is_registered() -> None:
    assert set(REGISTRY) == EXPECTED


def test_schemas_are_openai_function_shape() -> None:
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        fn = schema["function"]
        assert set(fn) == {"name", "description", "parameters"}
        assert fn["name"] in REGISTRY
        assert fn["description"].strip()
        assert fn["parameters"]["type"] == "object"
        assert isinstance(fn["parameters"].get("properties", {}), dict)


def test_schemas_are_json_serialisable_and_names_unique() -> None:
    json.dumps(TOOL_SCHEMAS)
    names = [s["function"]["name"] for s in TOOL_SCHEMAS]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_required_properties_exist_in_properties(name: str) -> None:
    params = REGISTRY[name].parameters
    for required in params.get("required", []):
        assert required in params.get("properties", {}), f"{name}.{required} missing"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_property_declares_a_type(name: str) -> None:
    params = REGISTRY[name].parameters
    for prop, spec in params.get("properties", {}).items():
        assert "type" in spec, f"{name}.{prop} has no type"
        assert "description" in spec, f"{name}.{prop} has no description"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_descriptions_are_actionable(name: str) -> None:
    description = REGISTRY[name].description
    assert len(description) > 60, f"{name} description is too thin to guide the model"


# -- fixtures -----------------------------------------------------------
@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    root = tmp_path / "run"
    for sub in ("data", "outputs", "figures"):
        (root / sub).mkdir(parents=True)
    return ToolContext(
        session_id="s1",
        paths={
            "root": root,
            "data": root / "data",
            "outputs": root / "outputs",
            "figures": root / "figures",
        },
        state=ResearchState(session_id="s1", root_dir=str(root)),
        settings=None,  # type: ignore[arg-type]
        logger=EventLogger("s1", root / "events.jsonl"),
    )


# -- files --------------------------------------------------------------
async def test_write_then_read(ctx: ToolContext) -> None:
    written = await files_tool.write_file({"path": "notes.md", "content": "hello"}, ctx)
    assert not written.error
    assert "Wrote" in written.text
    assert (ctx.output_dir / "notes.md").exists()

    read = await files_tool.read_file({"path": "notes.md"}, ctx)
    assert not read.error
    assert "hello" in read.text
    assert read.data["truncated"] is False


async def test_write_rejects_traversal_out_of_the_session(ctx: ToolContext) -> None:
    result = await files_tool.write_file({"path": "../../../../escape.md", "content": "x"}, ctx)
    assert result.error
    assert "outside the session folder" in result.text


async def test_write_allows_a_parent_step_that_stays_inside(ctx: ToolContext) -> None:
    result = await files_tool.write_file({"path": "../top-level.md", "content": "x"}, ctx)
    assert not result.error
    assert (ctx.root / "top-level.md").exists()


async def test_read_rejects_traversal(ctx: ToolContext) -> None:
    assert (await files_tool.read_file({"path": "../../../../etc/hostname"}, ctx)).error


async def test_read_missing_file(ctx: ToolContext) -> None:
    result = await files_tool.read_file({"path": "nope.md"}, ctx)
    assert result.error
    assert "No such file" in result.text


async def test_read_truncates_and_says_so(ctx: ToolContext) -> None:
    await files_tool.write_file({"path": "big.txt", "content": "y" * 5_000}, ctx)
    result = await files_tool.read_file({"path": "big.txt", "max_chars": 100}, ctx)
    assert result.data["truncated"] is True
    assert "truncated" in result.text


async def test_list_files_shows_outputs(ctx: ToolContext) -> None:
    await files_tool.write_file({"path": "a.csv", "content": "1,2"}, ctx)
    (ctx.figures_dir / "1_1.png").write_bytes(b"\x89PNG")
    listing = await files_tool.list_files({}, ctx)
    assert "a.csv" in listing.text
    assert "1_1.png" in listing.text
    assert "image" in listing.text


async def test_list_files_on_empty_dir(ctx: ToolContext) -> None:
    listing = await files_tool.list_files({"dir": "outputs"}, ctx)
    assert "empty" in listing.text.lower()


# -- todo ---------------------------------------------------------------
async def test_todo_replaces_the_list(ctx: ToolContext) -> None:
    first = await todo.handler(
        {"items": [{"id": "1", "text": "a", "status": "pending"}]}, ctx
    )
    assert not first.error
    assert len(ctx.state.plan) == 1
    await todo.handler(
        {
            "items": [
                {"id": "1", "text": "a", "status": "done"},
                {"id": "2", "text": "b", "status": "in_progress"},
            ]
        },
        ctx,
    )
    assert len(ctx.state.plan) == 2
    assert ctx.state.plan[0].status == "done"


async def test_todo_rejects_bad_input(ctx: ToolContext) -> None:
    assert (await todo.handler({"items": "nope"}, ctx)).error
    assert (await todo.handler({"items": [{"text": ""}]}, ctx)).error
    assert (await todo.handler({"items": [{"text": "ok"}, "junk"]}, ctx)).error


async def test_todo_result_carries_the_plan_for_the_ui(ctx: ToolContext) -> None:
    result = await todo.handler({"items": [{"id": "1", "text": "a", "status": "done"}]}, ctx)
    assert result.data["plan"][0]["status"] == "done"


# -- state tools --------------------------------------------------------
async def test_record_finding_dedupes_and_tags(ctx: ToolContext) -> None:
    a = await state_tools.record_finding({"text": "rf won", "tags": ["model"]}, ctx)
    assert "Finding recorded" in a.text
    assert "[tags: model]" in ctx.state.findings[0]
    b = await state_tools.record_finding({"text": "rf won", "tags": ["model"]}, ctx)
    assert "already recorded" in b.text
    assert len(ctx.state.findings) == 1


async def test_record_finding_rejects_empty(ctx: ToolContext) -> None:
    assert (await state_tools.record_finding({"text": "  "}, ctx)).error


async def test_query_state_sections(ctx: ToolContext) -> None:
    ctx.state.task = TaskSpec(target="y", task_type="classification", primary_metric="f1_macro")
    ctx.state.add_finding("something")
    for section in ("dataset", "task", "plan", "experiments", "findings", "all"):
        result = await state_tools.query_state({"section": section}, ctx)
        assert not result.error, section
        assert result.text.strip()
    assert (await state_tools.query_state({"section": "bogus"}, ctx)).error


async def test_query_state_findings_section_lists_them(ctx: ToolContext) -> None:
    ctx.state.add_finding("alpha")
    ctx.state.add_finding("beta")
    result = await state_tools.query_state({"section": "findings"}, ctx)
    assert "- alpha" in result.text and "- beta" in result.text


# -- ask_user -----------------------------------------------------------
async def test_ask_user_without_an_agent_explains_itself(ctx: ToolContext) -> None:
    result = await ask_user_tool.handler({"question": "which target?"}, ctx)
    assert result.error
    assert "not available" in result.text
    assert "which target?" in result.text


async def test_ask_user_uses_the_agent_callback(ctx: ToolContext) -> None:
    class FakeAgent:
        def __init__(self) -> None:
            self.asked: list[tuple[str, list[str]]] = []

        async def ask_user(self, question: str, options: list[str]) -> str:
            self.asked.append((question, options))
            return "churned"

    agent = FakeAgent()
    ctx.agent = agent
    result = await ask_user_tool.handler(
        {"question": "which target?", "options": ["a", "b"]}, ctx
    )
    assert not result.error
    assert result.data["answer"] == "churned"
    assert agent.asked == [("which target?", ["a", "b"])]


async def test_ask_user_handles_no_answer(ctx: ToolContext) -> None:
    class Silent:
        async def ask_user(self, question: str, options: list[str]) -> str:
            return ""

    ctx.agent = Silent()
    result = await ask_user_tool.handler({"question": "?"}, ctx)
    assert not result.error
    assert "did not answer" in result.text


# -- event log ----------------------------------------------------------
def test_event_logger_sequences_and_writes(tmp_path: Path) -> None:
    logger = EventLogger("s1", tmp_path / "events.jsonl")
    logger.emit("user_message", {"text": "hi"})
    logger.emit("assistant_message", {"text": "hello"})
    assert [e.seq for e in logger.events] == [1, 2]
    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["type"] == "user_message"


def test_assistant_deltas_are_neither_written_nor_buffered(tmp_path: Path) -> None:
    logger = EventLogger("s1", tmp_path / "events.jsonl")
    for _ in range(50):
        logger.emit("assistant_delta", {"text": "tok"})
    logger.emit("assistant_message", {"text": "full"})
    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    assert len(logger.events) == 1  # deltas are not retained either

def test_event_logger_redacts_secrets(tmp_path: Path) -> None:
    logger = EventLogger("s1", tmp_path / "events.jsonl", secrets=("sk-abcdefgh12345678",))
    logger.emit("tool_result", {"text": "key is sk-abcdefgh12345678"})
    content = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert "sk-abcdefgh12345678" not in content
    assert "redacted" in content


def test_redact_data_walks_nested_structures() -> None:
    out = redact_data({"a": ["sk-abcdefgh1234", {"b": "sk-abcdefgh1234"}]}, ("sk-abcdefgh1234",))
    assert "sk-abcdefgh1234" not in json.dumps(out)


def test_event_logger_round_trips(tmp_path: Path) -> None:
    logger = EventLogger("s1", tmp_path / "events.jsonl")
    logger.emit("user_message", {"text": "hi"})
    logger.emit("done", {"steps": 2})
    read = logger.read_all()
    assert [r["type"] for r in read] == ["user_message", "done"]
    Event.model_validate(read[1])  # the log is replayable
