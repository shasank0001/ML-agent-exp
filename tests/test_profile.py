"""Dataset profiler: frame stats, JSON safety, handler state handling, edge cases."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datalab.events import EventLogger
from datalab.state import ResearchState, TaskSpec
from datalab.tools import profile as profile_tool
from datalab.tools.base import ToolContext
from datalab.tools.profile import profile_dataset, profile_frame


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


def messy_frame(n: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:04d}" for i in range(n)],
            "age": rng.integers(18, 80, n).astype(float),
            "plan": rng.choice(["basic", "plus", "pro"], n),
            "spend": rng.gamma(2.0, 30.0, n).round(2),
            "churn": rng.choice(["yes", "no"], n, p=[0.25, 0.75]),
        }
    )
    df.loc[rng.choice(n, 20, replace=False), "age"] = np.nan
    df.loc[rng.choice(n, 12, replace=False), "spend"] = np.nan
    return pd.concat([df, df.iloc[:2]], ignore_index=True)  # 2 duplicate rows


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False)
    return path


# -- messy classification -----------------------------------------------
def test_messy_csv_suggests_classification_with_string_target(tmp_path: Path) -> None:
    info = profile_dataset(write_csv(messy_frame(), tmp_path / "customers.csv"))
    assert info.suggested_task_type == "classification"
    assert info.target_candidates[0] == "churn"


def test_notes_call_out_missing_values_and_the_id_column(tmp_path: Path) -> None:
    info = profile_dataset(write_csv(messy_frame(), tmp_path / "customers.csv"))
    notes = "\n".join(info.profile_notes)
    assert "age" in notes and "missing" in notes
    assert "customer_id" in notes and "identifier" in notes


def test_duplicates_and_missing_counts_are_exact(tmp_path: Path) -> None:
    df = messy_frame()
    info = profile_dataset(write_csv(df, tmp_path / "customers.csv"))
    assert info.duplicates == 2
    by_name = {c["name"]: c for c in info.columns}
    assert by_name["age"]["n_missing"] == 20
    assert by_name["spend"]["n_missing"] == 12
    assert by_name["plan"]["n_missing"] == 0


def test_conftest_messy_csv_is_classification(messy_csv: Path) -> None:
    info = profile_dataset(messy_csv)
    assert info.suggested_task_type == "classification"
    assert info.target_candidates[0] == "churned"


def test_conftest_regression_csv_is_regression(regression_csv: Path) -> None:
    info = profile_dataset(regression_csv)
    assert info.suggested_task_type == "regression"
    assert "price_k" in info.target_candidates


# -- JSON safety ----------------------------------------------------------
def test_messy_profile_is_json_serialisable(messy_csv: Path) -> None:
    json.dumps(profile_dataset(messy_csv).model_dump())  # numpy scalars must not leak


def test_regression_profile_is_json_serialisable(regression_csv: Path) -> None:
    json.dumps(profile_dataset(regression_csv).model_dump())


# -- handler --------------------------------------------------------------
async def test_handler_records_dataset_but_keeps_the_task(ctx: ToolContext) -> None:
    write_csv(messy_frame(), ctx.data_dir / "customers.csv")
    ctx.state.task = TaskSpec(target="churn", task_type="classification", primary_metric="f1_macro")
    result = await profile_tool.handler({"path": "customers.csv"}, ctx)
    assert not result.error
    assert ctx.state.dataset is not None and ctx.state.dataset.n_rows == 202
    assert ctx.state.task is not None and ctx.state.task.target == "churn"
    assert ctx.state.task.primary_metric == "f1_macro"


async def test_handler_failure_lists_what_is_available(ctx: ToolContext) -> None:
    (ctx.data_dir / "customers.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    result = await profile_tool.handler({"path": "ghost.csv"}, ctx)
    assert result.error
    assert "not found" in result.text
    assert "customers.csv" in result.text


# -- edge cases -----------------------------------------------------------
def test_empty_frame_does_not_crash() -> None:
    info = profile_frame(pd.DataFrame(), "empty.csv")
    assert info.n_rows == 0 and info.n_cols == 0
    json.dumps(info.model_dump())


def test_single_row_frame_does_not_crash() -> None:
    info = profile_frame(pd.DataFrame({"a": [1], "b": ["x"]}), "one.csv")
    assert info.n_rows == 1
    json.dumps(info.model_dump())


def test_all_nan_column_does_not_crash() -> None:
    df = pd.DataFrame({"ghost": [np.nan] * 5, "y": [0, 1, 0, 1, 0]})
    info = profile_frame(df, "gaps.csv")
    by_name = {c["name"]: c for c in info.columns}
    assert by_name["ghost"]["n_missing"] == 5
    json.dumps(info.model_dump())
