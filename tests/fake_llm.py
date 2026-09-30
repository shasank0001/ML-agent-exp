"""Test doubles for the LLM and the UI callbacks.

`ScriptedLLM` replays a fixed list of turns so the agent loop can be tested
without a provider. `PolicyLLM` is a small rule-based agent that actually drives
the real tools — it is how the golden path is rehearsed offline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from datalab.llm import SelfTestResult, ToolCall, Usage

Turn = tuple[str, list[ToolCall]]


def call(name: str, **args: Any) -> ToolCall:
    """Build a ToolCall the way the OpenAI client would."""
    return ToolCall(id=f"call_{name}_{abs(hash(name + json.dumps(args, sort_keys=True))) % 10**8}",
                    name=name, arguments=args, raw_arguments=json.dumps(args))


@dataclass
class ScriptedLLM:
    """Replays ``turns`` in order; raises if the agent asks for more than it has."""

    turns: list[Turn]
    model: str = "scripted-model"
    index: int = 0
    seen: list[list[dict[str, Any]]] = field(default_factory=list)
    _repeat_last: bool = False

    def queue(self, text: str, *calls: ToolCall) -> None:
        self.turns.append((text, list(calls)))

    async def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[tuple[str, Any]]:
        self.seen.append(messages)
        if self.index >= len(self.turns):
            if not self._repeat_last and self.turns:
                # Running past the script is a normal "the model keeps going" case.
                self._repeat_last = True
                self.index = len(self.turns) - 1
            else:
                yield ("text", "")
                yield ("tool_calls", [])
                yield ("usage", Usage(1, 1, 2))
                return
        text, calls = self.turns[self.index]
        self.index += 1
        for i in range(0, len(text), 24):
            yield ("text", text[i : i + 24])
        yield ("tool_calls", list(calls))
        yield ("usage", Usage(100, 20, 120))

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ):
        async for kind, payload in self.stream(messages, tools):
            if kind == "tool_calls":
                return "", payload, Usage(1, 1, 2)
        return "", [], Usage(1, 1, 2)

    async def self_test(self) -> SelfTestResult:
        return SelfTestResult(True, self.model, "scripted")


class RecordingApproval:
    """Stands in for the Approve/Deny buttons."""

    def __init__(self, answers: list[bool] | None = None, default: bool = True) -> None:
        self.answers = list(answers or [])
        self.default = default
        self.requests: list[Any] = []

    async def __call__(self, request: Any) -> bool:
        self.requests.append(request)
        return self.answers.pop(0) if self.answers else self.default


class RecordingAskUser:
    """Stands in for the AskUserMessage round trip."""

    def __init__(self, answers: list[str] | None = None) -> None:
        self.answers = list(answers or [])
        self.questions: list[tuple[str, list[str]]] = []

    async def __call__(self, question: str, options: list[str]) -> str:
        self.questions.append((question, list(options)))
        return self.answers.pop(0) if self.answers else ""


# -- a small rule-based agent, for offline rehearsal --------------------
_TARGET_RE = re.compile(r"predict(?:s|ing)?\s+`?([A-Za-z_][A-Za-z0-9_ ]*)`?", re.I)


@dataclass
class PolicyLLM:
    """Drives the real tools through the golden path without any model.

    It reads the tool results that are already in its own message history, so it
    behaves like a competent-but-simple agent: profile -> ask for the target ->
    plan -> split -> baseline -> two models -> results -> finding.
    """

    model: str = "policy-model"
    target: str | None = None
    task_type: str = "classification"
    limit: int = 400
    index: int = 0
    self_test_ok: bool = True
    self_test_detail: str = "scripted policy"
    _state: dict[str, Any] = field(default_factory=dict)

    def _history(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return messages

    @staticmethod
    def _tool_outputs(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
        return [
            (str(m.get("name", "")), str(m.get("content", "")))
            for m in messages
            if m.get("role") == "tool"
        ]

    def _last(self, outputs: list[tuple[str, str]], name: str) -> str:
        for tool, text in reversed(outputs):
            if tool == name:
                return text
        return ""

    async def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[tuple[str, Any]]:
        self.index += 1
        text, calls = self._next_turn(messages)
        for i in range(0, len(text), 24):
            yield ("text", text[i : i + 24])
        yield ("tool_calls", calls)
        yield ("usage", Usage(200, 50, 250))

    async def complete(self, messages, tools=None):
        async for kind, payload in self.stream(messages, tools):
            if kind == "tool_calls":
                return "", payload, Usage(1, 1, 2)
        return "", [], Usage(1, 1, 2)

    async def self_test(self) -> SelfTestResult:
        return SelfTestResult(self.self_test_ok, self.model, self.self_test_detail)

    # -- the policy -----------------------------------------------------
    def _next_turn(self, messages: list[dict[str, Any]]) -> Turn:
        outputs = self._tool_outputs(messages)
        done = {name for name, _ in outputs}
        state_text = messages[0]["content"] if messages else ""
        if "Task: not defined yet" in state_text and not self.target:
            match = _TARGET_RE.search(state_text)
            if match:
                self.target = match.group(1).strip().strip("`")
        path = self._dataset_path(messages)

        if not path:
            return ("I need a dataset first. Please upload a CSV file.", [])
        if "profile_dataset" not in done:
            return (f"Profiling `{path}` first.", [call("profile_dataset", path=path)])
        if self.target is None:
            answered = re.search(r"User answered:\s*(.+)", self._last(outputs, "ask_user"))
            if answered:
                self.target = answered.group(1).strip()
        if self.target is None:
            return (
                "Which column should I predict?",
                [call("ask_user", question="Which column is the target to predict?",
                      options=["churned", "price"])],
            )
        if "todo" not in done:
            items = [
                {"id": "1", "text": f"Profile the data", "status": "done"},
                {"id": "2", "text": f"Confirm the target column ({self.target})", "status": "done"},
                {"id": "3", "text": "Create the train/test split", "status": "in_progress"},
                {"id": "4", "text": "Run the baseline models", "status": "pending"},
                {"id": "5", "text": "Train random forest and gradient boosting", "status": "pending"},
                {"id": "6", "text": "Compare results and explain the winner", "status": "pending"},
            ]
            return ("Here is the plan.", [call("todo", items=items)])
        if "python" not in done:
            return ("Creating the split.", [call("python", description="Create the split", code=(
                "df = lab.load(%r)\n"
                "X_train, X_test, y_train, y_test = lab.split(df, %r)\n"
                "X_train.shape\n" % (path, self.target)
            ))])
        if not any("BASELINES_DONE" in t for _, t in outputs):
            return ("Running the baselines.", [call("python", description="Run the baselines", code=(
                "lab.baseline()\n" "lab.results_table()\n" "print('BASELINES_DONE')\n"
            ))])
        lab_md = lab_code_markdown(messages)
        trained = self._state.get("trained", 0)
        if trained == 0:
            self._state["trained"] = 1
            return ("Training a random forest.", [call("python", description="Train a random forest",
                est_seconds=5, code=(
                    "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor\n"
                    "RF = (RandomForestClassifier if %r == 'classification' else RandomForestRegressor)"
                    "(n_estimators=120, random_state=0, n_jobs=2)\n"
                    "lab.evaluate(RF, name='random_forest')\n" % self.task_type))])
        if trained == 1:
            self._state["trained"] = 2
            return ("Training gradient boosting.", [call("python", description="Train gradient boosting",
                est_seconds=5, code=(
                    "from lightgbm import LGBMClassifier, LGBMRegressor\n"
                    "GBM = (LGBMClassifier if %r == 'classification' else LGBMRegressor)"
                    "(n_estimators=200, random_state=0, verbose=-1)\n"
                    "lab.evaluate(GBM, name='lightgbm')\n" % self.task_type))])
        if "record_finding" not in done:
            self._state["table"] = lab_md
            return ("Recording the outcome.", [call("record_finding", text=(
                f"Trained random forest and LightGBM on `{self.target}`; "
                "compared via lab.results_table()."))])
        return (
            "Here is the comparison:\n\n```\n"
            + lab_md
            + "\n```\n\n"
            + "The winner is whichever model is ranked first in the table above; the numbers "
            "come straight from `lab.results_table()`, which is backed by `state.json`.",
            [],
        )

    def _dataset_path(self, messages: list[dict[str, Any]]) -> str | None:
        match = re.search(r"DATA_DIR for this session: ([^\s]+)", " ".join(
            str(m.get("content", "")) for m in messages if m.get("role") == "user"
        ))
        if match:
            return match.group(1)
        state_text = messages[0]["content"] if messages else ""
        match = re.search(r"Dataset: (\S+) —", state_text)
        return match.group(1) if match else None


def lab_code_markdown(messages: list[dict[str, Any]]) -> str:
    """Recover the last results table the policy printed inside a python cell."""
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        text = str(message.get("content", ""))
        if "| name |" in text or "results" in text.lower() and "|" in text:
            start = text.find("|")
            if start >= 0:
                return text[start:]
    return "(no table captured)"


PolicyFactory = Callable[[], Any]
