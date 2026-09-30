"""OpenAI-compatible streaming client (LM Studio or OpenRouter) + tool-call self-test.

Both providers speak the chat-completions API, so a single :class:`LLMClient`
covers them; only the base URL, key and model id differ.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from .config import Settings

#: Emitted by :meth:`LLMClient.stream`.
StreamKind = str  # "text" | "tool_calls" | "usage"


@dataclass
class ToolCall:
    """A fully accumulated tool call."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""

    def to_message(self) -> dict[str, Any]:
        """Render as the ``tool_calls`` entry of an assistant message."""
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.raw_arguments or json.dumps(self.arguments)},
        }


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class SelfTestResult:
    """Outcome of the tool-calling capability probe."""

    ok: bool
    model: str
    detail: str = ""


class ToolCallNotSupported(RuntimeError):
    """Raised when the provider rejects a request that carries ``tools``."""


def _parse_arguments(raw: str) -> dict[str, Any]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Some models stream arguments as a bare string or emit trailing commas.
        try:
            parsed = json.loads(raw.replace(",}", "}").replace(",]", "]"))
        except json.JSONDecodeError:
            return {"__raw_arguments__": raw}
    return parsed if isinstance(parsed, dict) else {"__raw_arguments__": raw}


class LLMClient:
    """Thin streaming wrapper around ``AsyncOpenAI``."""

    def __init__(self, settings: Settings, client: AsyncOpenAI | None = None) -> None:
        self.settings = settings
        self._client = client
        self.last_usage = Usage()

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.settings.base_url,
                api_key=self.settings.api_key or "not-set",
                timeout=300.0,
                max_retries=2,
            )
        return self._client

    @property
    def model(self) -> str:
        return self.settings.model

    # -- streaming ------------------------------------------------------
    async def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[tuple[StreamKind, Any]]:
        """Stream one assistant turn.

        Yields ``("text", delta)`` for each content delta and, once the stream
        ends, ``("tool_calls", [ToolCall, ...])`` followed by
        ``("usage", Usage)``.
        """
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            stream = await self._create(kwargs, with_usage=True)
        except Exception as exc:  # noqa: BLE001 - re-raised below when it is a tools problem
            if tools and _is_stream_options_error(exc):
                # Some local servers reject stream_options; drop it and retry once.
                try:
                    stream = await self._create(kwargs, with_usage=False)
                except Exception as exc2:  # noqa: BLE001
                    raise _wrap(exc2, tools) from exc2
            else:
                raise _wrap(exc, tools) from exc

        text_parts: list[str] = []
        slots: dict[int, dict[str, Any]] = {}
        usage = Usage()

        async for chunk in stream:
            usage_chunk = getattr(chunk, "usage", None)
            if usage_chunk is not None:
                usage = Usage(
                    prompt_tokens=getattr(usage_chunk, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage_chunk, "completion_tokens", 0) or 0,
                    total_tokens=getattr(usage_chunk, "total_tokens", 0) or 0,
                )
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue
            content = getattr(delta, "content", None)
            if content:
                text_parts.append(content)
                yield ("text", content)
            for tc in getattr(delta, "tool_calls", None) or []:
                idx = getattr(tc, "index", 0) or 0
                slot = slots.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if getattr(tc, "id", None):
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["arguments"] += fn.arguments

        self.last_usage = usage
        tool_calls = [
            ToolCall(
                id=slot["id"] or f"call_{uuid.uuid4().hex[:10]}",
                name=slot["name"],
                raw_arguments=slot["arguments"],
                arguments=_parse_arguments(slot["arguments"]),
            )
            for _, slot in sorted(slots.items())
            if slot["name"]
        ]
        yield ("tool_calls", tool_calls)
        yield ("usage", usage)

    async def _create(self, kwargs: dict[str, Any], *, with_usage: bool):
        if with_usage:
            kwargs = {**kwargs, "stream_options": {"include_usage": True}}
        return await self.client.chat.completions.create(**kwargs)

    # -- one-shot helpers ----------------------------------------------
    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> tuple[str, list[ToolCall], Usage]:
        """Non-streaming convenience wrapper (used by the self-test)."""
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        usage = Usage()
        async for kind, payload in self.stream(messages, tools):
            if kind == "text":
                text_parts.append(payload)
            elif kind == "tool_calls":
                calls = payload
            elif kind == "usage":
                usage = payload
        return "".join(text_parts), calls, usage

    async def self_test(self) -> SelfTestResult:
        """Verify that the configured model can emit a function call at all.

        A tiny, cheap, side-effect-free probe: one tool, one message, the model
        is expected to call it. Many small local models silently ignore the
        ``tools`` parameter, which would otherwise surface as a mysterious
        failure much later in the run.
        """
        probe_tool = {
            "type": "function",
            "function": {
                "name": "self_test_probe",
                "description": "Report that the tool-calling self-test ran. Always call this.",
                "parameters": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean", "description": "Always true."}},
                    "required": ["ok"],
                },
            },
        }
        messages = [
            {
                "role": "user",
                "content": "Call the self_test_probe tool with ok=true. Do not answer in text.",
            }
        ]
        try:
            _, calls, _ = await self.complete(messages, [probe_tool])
        except Exception as exc:  # noqa: BLE001 - reported to the user, never raised
            return SelfTestResult(False, self.settings.model, _short(exc))
        if not calls:
            return SelfTestResult(
                False,
                self.settings.model,
                "the model replied without requesting any tool, so it likely ignores the "
                "`tools` parameter",
            )
        if calls[0].name != "self_test_probe":
            return SelfTestResult(
                False,
                self.settings.model,
                f"the model called `{calls[0].name}` instead of `self_test_probe`",
            )
        return SelfTestResult(True, self.settings.model, "tool calling works")


# -- helpers ------------------------------------------------------------
def _is_stream_options_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "stream_options" in text or "stream options" in text


def _wrap(exc: Exception, tools: list[dict[str, Any]] | None) -> Exception:
    """Turn provider errors into something the agent can show the user."""
    if tools and isinstance(exc, ToolCallNotSupported):
        return exc
    text = str(exc)
    if tools and ("tool" in text.lower() and ("not support" in text.lower() or "invalid" in text.lower())):
        return ToolCallNotSupported(text)
    return exc


def _short(exc: Exception, limit: int = 400) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text[:limit]
