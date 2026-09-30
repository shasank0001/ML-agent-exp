"""Shared tool plumbing: the result type, the per-session context and the spec type.

Kept separate from :mod:`datalab.tools` (the registry) so that handlers can
import these types without creating an import cycle.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings
from ..events import EventLogger
from ..state import ResearchState


@dataclass
class ToolResult:
    """What a tool hands back to the agent loop."""

    text: str = ""
    error: bool = False
    images: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, text: str, **kw: Any) -> "ToolResult":
        return cls(text=text, error=False, **kw)

    @classmethod
    def fail(cls, text: str, **kw: Any) -> "ToolResult":
        return cls(text=text, error=True, **kw)


@dataclass
class ToolContext:
    """Everything a handler may touch for one session."""

    session_id: str
    paths: dict[str, Path]
    state: ResearchState
    settings: Settings
    logger: EventLogger
    #: Set by the agent once its python executor exists.
    executor: Any = None
    #: The running agent; used by ``ask_user`` to reach the UI callback.
    agent: Any = None

    @property
    def root(self) -> Path:
        return self.paths["root"]

    @property
    def data_dir(self) -> Path:
        return self.paths["data"]

    @property
    def output_dir(self) -> Path:
        return self.paths["outputs"]

    @property
    def figures_dir(self) -> Path:
        return self.paths["figures"]


Handler = Callable[[dict[str, Any], ToolContext], Awaitable[ToolResult]]


@dataclass
class ToolSpec:
    """A tool's OpenAI-style schema plus its handler."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def new_id(prefix: str = "") -> str:
    raw = uuid.uuid4().hex[:12]
    return f"{prefix}{raw}" if prefix else raw
