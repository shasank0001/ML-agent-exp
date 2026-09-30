"""Tool registry: name -> (OpenAI schema, async handler).

One flat list so the model gets a single, coherent toolbelt. Handlers are
``async def handler(args: dict, ctx: ToolContext) -> ToolResult`` and never
raise: they return ``ToolResult.fail(...)`` so the agent loop can repair.
"""

from __future__ import annotations

from .base import Handler, ToolContext, ToolResult, ToolSpec
from . import ask_user as _ask_user
from . import files as _files
from . import profile as _profile
from . import python_exec as _python
from . import state_tools as _state
from . import todo as _todo

__all__ = [
    "REGISTRY",
    "TOOL_SCHEMAS",
    "get_tool",
    "has_tool",
    "ToolContext",
    "ToolResult",
    "ToolSpec",
    "Handler",
]

REGISTRY: dict[str, ToolSpec] = {
    "python": ToolSpec(
        name="python",
        description=_python.DESCRIPTION,
        parameters=_python.SCHEMA,
        handler=_python.handler,
    ),
    "profile_dataset": ToolSpec(
        name="profile_dataset",
        description=(
            "Deterministic pandas profile of a data file: shape, dtypes, missing counts, "
            "cardinality, numeric summary, top categories, duplicate rows, suggested target "
            "candidates and task type. Computed in Python, never guessed. Call this before any "
            "modelling work on a file you have not profiled."
        ),
        parameters=_profile.SCHEMA,
        handler=_profile.handler,
    ),
    "read_file": ToolSpec(
        name="read_file", description=_files.READ_DESCRIPTION, parameters=_files.READ_SCHEMA, handler=_files.read_file
    ),
    "write_file": ToolSpec(
        name="write_file", description=_files.WRITE_DESCRIPTION, parameters=_files.WRITE_SCHEMA, handler=_files.write_file
    ),
    "list_files": ToolSpec(
        name="list_files", description=_files.LIST_DESCRIPTION, parameters=_files.LIST_SCHEMA, handler=_files.list_files
    ),
    "todo": ToolSpec(name="todo", description=_todo.DESCRIPTION, parameters=_todo.SCHEMA, handler=_todo.handler),
    "ask_user": ToolSpec(
        name="ask_user", description=_ask_user.DESCRIPTION, parameters=_ask_user.SCHEMA, handler=_ask_user.handler
    ),
    "record_finding": ToolSpec(
        name="record_finding",
        description=_state.FINDING_DESCRIPTION,
        parameters=_state.FINDING_SCHEMA,
        handler=_state.record_finding,
    ),
    "query_state": ToolSpec(
        name="query_state",
        description=_state.QUERY_DESCRIPTION,
        parameters=_state.QUERY_SCHEMA,
        handler=_state.query_state,
    ),
}

#: OpenAI tool schemas handed to the model, in a stable order.
TOOL_SCHEMAS: list[dict] = [spec.schema() for spec in REGISTRY.values()]


def get_tool(name: str) -> ToolSpec | None:
    return REGISTRY.get(name)


def has_tool(name: str) -> bool:
    return name in REGISTRY
