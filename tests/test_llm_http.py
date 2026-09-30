"""The streaming client against a real HTTP server.

Everything else in the suite stubs the LLM out entirely, which leaves the parts
that only break over the wire untested: SSE framing, chunked tool-call deltas,
`stream_options` negotiation, and the retry when a server rejects that option.
A small threaded HTTP server here speaks enough of the chat-completions API to
cover those without a network.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator

import pytest

from datalab.config import Settings
from datalab.llm import LLMClient, ToolCall, ToolCallNotSupported

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "python",
            "description": "Run Python code.",
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}}},
        },
    }
]

MESSAGES = [{"role": "user", "content": "hi"}]


# -- chunk builders -----------------------------------------------------


def text_chunk(content: str) -> dict[str, Any]:
    return _chunk({"content": content})


def tool_chunk(index: int, call_id: str, name: str | None, arguments: str) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if call_id:
        delta["id"] = call_id
    function = {k: v for k, v in (("name", name), ("arguments", arguments)) if v}
    if function:
        delta["function"] = function
    return _chunk({"tool_calls": [{"index": index, **delta}]})


def finish_chunk(reason: str = "tool_calls") -> dict[str, Any]:
    chunk = _chunk({})
    chunk["choices"][0]["finish_reason"] = reason
    return chunk


def usage_chunk(prompt: int, completion: int) -> dict[str, Any]:
    return {
        "id": "c",
        "object": "chat.completion.chunk",
        "model": "mock-model",
        "choices": [],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


def _chunk(delta: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "c",
        "object": "chat.completion.chunk",
        "model": "mock-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }


# -- the mock server ----------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    """Answers `POST /v1/chat/completions` from the attached `MockOpenAIServer`."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:  # keep the test output clean
        pass

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        request: dict[str, Any] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        server: MockOpenAIServer = self.server.script  # type: ignore[attr-defined]
        server.calls.append(request)

        if "stream_options" in request and server.reject_stream_options:
            # Refused without consuming a scripted turn: the client is expected
            # to retry the same request without the option.
            self._json(400, {"error": {"message": "stream_options is not supported"}})
            return
        if request.get("tools") and server.refuse_tools:
            self._json(400, {"error": {"message": "'tools' is not supported by this model"}})
            return

        body = "".join(f"data: {json.dumps(c)}\n\n" for c in server.turn(request)).encode()
        body += b"data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class MockOpenAIServer:
    """A threaded server whose replies are supplied by the test."""

    def __init__(self) -> None:
        self.turns: list[list[dict[str, Any]]] = []
        self.calls: list[dict[str, Any]] = []
        self.reject_stream_options = False
        self.refuse_tools = False
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._httpd.script = self  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> "MockOpenAIServer":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def next(self, chunks: list[dict[str, Any]]) -> None:
        self.turns.append(chunks)

    def turn(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        if not self.turns:
            raise AssertionError("the client made more requests than the test scripted")
        return self.turns.pop(0)


# -- fixtures -----------------------------------------------------------


@pytest.fixture
def server() -> Iterator[MockOpenAIServer]:
    with MockOpenAIServer() as mock:
        yield mock


@pytest.fixture
def client(server: MockOpenAIServer) -> Settings:
    return Settings(
        provider="lmstudio",
        base_url=server.base_url,
        api_key="test-key",
        model="mock-model",
        max_steps=5,
        max_repairs=1,
        approval_seconds_threshold=60,
        python_soft_timeout_s=10,
    )


async def drive(client: Settings, messages=None, tools=None) -> tuple[str, list[ToolCall], Any]:
    text: list[str] = []
    calls: list[ToolCall] = []
    usage = None
    async for kind, payload in LLMClient(client).stream(messages or MESSAGES, tools):
        if kind == "text":
            text.append(payload)
        elif kind == "tool_calls":
            calls = payload
        else:
            usage = payload
    return "".join(text), calls, usage


# -- streaming ----------------------------------------------------------
async def test_plain_text_streams_in_pieces(server: MockOpenAIServer, client: Settings) -> None:
    server.next([text_chunk("Hello"), text_chunk(", world"), finish_chunk("stop")])
    text, calls, _ = await drive(client)
    assert text == "Hello, world"
    assert calls == []


async def test_usage_is_captured(server: MockOpenAIServer, client: Settings) -> None:
    server.next([text_chunk("hi"), finish_chunk("stop"), usage_chunk(120, 30)])
    _, _, usage = await drive(client)
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (120, 30, 150)


async def test_no_usage_chunk_leaves_zeroed_usage(server: MockOpenAIServer, client: Settings) -> None:
    server.next([text_chunk("hi"), finish_chunk("stop")])
    _, _, usage = await drive(client)
    assert usage.total_tokens == 0


# -- tool calls ---------------------------------------------------------
async def test_tool_arguments_arrive_in_fragments(server: MockOpenAIServer, client: Settings) -> None:
    server.next([
        tool_chunk(0, "call_1", "python", '{"co'),
        tool_chunk(0, "", None, 'de": "print(1)"}'),
        finish_chunk("tool_calls"),
    ])
    _, calls, _ = await drive(client, tools=TOOLS)
    assert len(calls) == 1
    assert calls[0].name == "python"
    assert calls[0].id == "call_1"
    assert calls[0].arguments == {"code": "print(1)"}
    assert json.loads(calls[0].to_message()["function"]["arguments"]) == {"code": "print(1)"}


async def test_two_tool_calls_are_kept_apart(server: MockOpenAIServer, client: Settings) -> None:
    server.next([
        tool_chunk(0, "call_1", "python", '{"code":'),
        tool_chunk(1, "call_2", "record_finding", '{"text": "ok"}'),
        tool_chunk(0, "", None, '"x"}'),
        finish_chunk("tool_calls"),
    ])
    _, calls, _ = await drive(client, tools=TOOLS)
    assert [c.id for c in calls] == ["call_1", "call_2"]
    assert calls[0].arguments == {"code": "x"}
    assert calls[1].arguments == {"text": "ok"}


async def test_indexless_deltas_do_not_merge(server: MockOpenAIServer, client: Settings) -> None:
    server.next([
        tool_chunk(0, "call_a", "todo", '{"items": []}'),
        tool_chunk(0, "call_b", "record_finding", '{"text": "b"}'),
        finish_chunk("tool_calls"),
    ])
    _, calls, _ = await drive(client, tools=TOOLS)
    assert [c.name for c in calls] == ["todo", "record_finding"]


async def test_malformed_arguments_come_back_verbatim(
    server: MockOpenAIServer, client: Settings
) -> None:
    server.next([tool_chunk(0, "call_1", "todo", "{not json"), finish_chunk("tool_calls")])
    _, calls, _ = await drive(client, tools=TOOLS)
    assert calls[0].arguments == {"__raw_arguments__": "{not json"}


async def test_tools_are_sent_with_auto_choice(server: MockOpenAIServer, client: Settings) -> None:
    server.next([finish_chunk("stop")])
    await drive(client, tools=TOOLS)
    sent = server.calls[-1]
    assert sent["tools"] == TOOLS
    assert sent["tool_choice"] == "auto"
    assert sent["stream"] is True


async def test_no_tools_field_when_there_are_none(server: MockOpenAIServer, client: Settings) -> None:
    server.next([text_chunk("hi"), finish_chunk("stop")])
    await drive(client)
    assert "tools" not in server.calls[-1]


# -- negotiation --------------------------------------------------------
async def test_stream_options_are_dropped_when_the_server_refuses(
    server: MockOpenAIServer, client: Settings
) -> None:
    server.reject_stream_options = True
    server.next([text_chunk("fine"), finish_chunk("stop")])
    text, _, _ = await drive(client)
    assert text == "fine"
    assert len(server.calls) == 2
    assert "stream_options" in server.calls[0]
    assert "stream_options" not in server.calls[1]


async def test_stream_options_are_dropped_even_without_tools(
    server: MockOpenAIServer, client: Settings
) -> None:
    """The option is sent on every request, so the retry must not depend on tools."""
    server.reject_stream_options = True
    server.next([text_chunk("fine"), finish_chunk("stop")])
    assert (await drive(client))[0] == "fine"
    assert len(server.calls) == 2


async def test_a_refused_tool_request_becomes_a_clear_error(
    server: MockOpenAIServer, client: Settings
) -> None:
    server.refuse_tools = True
    with pytest.raises(ToolCallNotSupported) as excinfo:
        await drive(client, tools=TOOLS)
    assert "not supported" in str(excinfo.value)


# -- the self-test ------------------------------------------------------
async def test_self_test_passes_against_a_well_behaved_server(
    server: MockOpenAIServer, client: Settings
) -> None:
    server.next([tool_chunk(0, "call_1", "self_test_probe", '{"ok": true}'),
                 finish_chunk("tool_calls")])
    result = await LLMClient(client).self_test()
    assert result.ok
    assert result.model == "mock-model"
    assert "works" in result.detail


async def test_self_test_fails_when_the_model_ignores_tools(
    server: MockOpenAIServer, client: Settings
) -> None:
    server.next([text_chunk("I would rather just answer."), finish_chunk("stop")])
    result = await LLMClient(client).self_test()
    assert not result.ok
    assert "tools" in result.detail


async def test_self_test_fails_when_the_wrong_tool_is_called(
    server: MockOpenAIServer, client: Settings
) -> None:
    server.next([tool_chunk(0, "call_1", "something_else", "{}"), finish_chunk("tool_calls")])
    result = await LLMClient(client).self_test()
    assert not result.ok
    assert "something_else" in result.detail


async def test_self_test_reports_a_transport_failure_instead_of_raising(
    server: MockOpenAIServer, client: Settings
) -> None:
    server.__exit__()  # the endpoint is gone
    result = await LLMClient(client).self_test()
    assert not result.ok
    assert result.detail
