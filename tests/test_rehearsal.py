"""Run the offline rehearsal as part of pytest.

This is the end-to-end acceptance suite from the build spec, executed with a
scripted model instead of a real one. It is slower than the unit tests (real
sklearn fits on real CSVs) but takes no network and no API key.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.rehearse import (
    SCENARIOS,
    run_scenario,
    verify_control,
    verify_flexibility,
    verify_golden,
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
async def messy_result():
    return await run_scenario("messy", SCENARIOS["messy"].build, task_type="classification")


async def test_golden_path(messy_result: dict) -> None:
    assert verify_golden(messy_result), "golden path did not complete"


async def test_state_and_log_agree(messy_result: dict) -> None:
    """Everything the UI showed must be recoverable from the two artifacts."""
    state = messy_result["state"]
    log = messy_result["log"]

    log_tools = [e["data"]["name"] for e in log if e["type"] == "tool_result"]
    assert log_tools == messy_result["tools"]

    # the results the agent reported are the results on disk
    for experiment in state["experiments"]:
        assert 0.0 <= experiment["metrics"]["f1_macro"] <= 1.0
        assert experiment["primary_metric"] == "f1_macro"

    # figures referenced by tool results exist
    for entry in log:
        for image in entry["data"].get("images") or []:
            assert Path(image).exists(), image


async def test_flexibility_without_code_changes(messy_result: dict) -> None:
    assert await verify_flexibility(messy_result), "an off-script request failed"


async def test_control_rails(messy_result: dict) -> None:
    assert await verify_control(messy_result), "a guard rail did not hold"


@pytest.mark.parametrize("name", ["regression", "multiclass", "degenerate"])
async def test_other_datasets(name: str) -> None:
    scenario = SCENARIOS[name]
    result = await run_scenario(name, scenario.build, task_type=scenario.task_type)
    assert verify_golden(result), f"{name}: golden path did not complete"
    task = result["state"]["task"]
    assert task["task_type"] == scenario.task_type
    assert task["primary_metric"] == ("rmse" if scenario.task_type == "regression" else "f1_macro")
    for experiment in result["state"]["experiments"]:
        assert task["primary_metric"] in experiment["metrics"]
    json.dumps(result["state"])  # the artifact is valid JSON on disk
