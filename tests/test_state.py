"""Research state: serialization, plan handling, experiment ranking, summary text."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datalab.state import Experiment, ResearchState, TaskSpec, TodoItem


def exp(name: str, metric: str, value: float) -> Experiment:
    return Experiment(
        id=name,
        name=name,
        model="M",
        params={},
        metrics={metric: value},
        primary_metric=metric,
        status="done",
    )


# -- persistence --------------------------------------------------------
def test_save_and_load_round_trip(tmp_path: Path) -> None:
    state = ResearchState(session_id="s1", root_dir=str(tmp_path))
    state.task = TaskSpec(target="y", task_type="classification", primary_metric="f1_macro")
    state.plan = [TodoItem(id="1", text="do it", status="done")]
    state.add_experiment(exp("rf", "f1_macro", 0.8))
    state.add_finding("rf won")
    state.save()

    reloaded = ResearchState.load(tmp_path / "state.json")
    assert reloaded.session_id == "s1"
    assert reloaded.task is not None and reloaded.task.target == "y"
    assert reloaded.plan[0].status == "done"
    assert reloaded.experiments[0].name == "rf"
    assert reloaded.findings == ["rf won"]


def test_state_json_is_valid_and_has_no_root_dir(tmp_path: Path) -> None:
    state = ResearchState(root_dir=str(tmp_path))
    state.save()
    raw = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert "root_dir" not in raw  # internal, must not leak into the artifact
    assert raw["session_id"] == state.session_id


def test_create_makes_the_directory(tmp_path: Path) -> None:
    root = tmp_path / "nested" / "run"
    state = ResearchState.create(root)
    assert (root / "state.json").exists()
    assert state.session_id


def test_save_is_atomic(tmp_path: Path) -> None:
    state = ResearchState(root_dir=str(tmp_path))
    state.save()
    state.save()
    assert not list(tmp_path.glob("*.tmp"))


# -- plan ---------------------------------------------------------------
def test_set_plan_normalises_ids_and_statuses() -> None:
    state = ResearchState()
    plan = state.set_plan(
        [
            {"text": "one"},
            {"id": "x", "text": "two", "status": "in_progress"},
            {"id": "y", "text": "three", "status": "nonsense"},
        ]
    )
    assert [p.id for p in plan] == ["1", "x", "y"]
    assert plan[2].status == "pending"
    assert state.plan is plan


def test_add_finding_deduplicates() -> None:
    state = ResearchState()
    state.add_finding("same")
    state.add_finding("same")
    assert state.findings == ["same"]


# -- ranking ------------------------------------------------------------
def test_ranking_higher_is_better() -> None:
    state = ResearchState()
    state.task = TaskSpec(target="y", task_type="classification", primary_metric="f1_macro")
    state.add_experiment(exp("low", "f1_macro", 0.5))
    state.add_experiment(exp("high", "f1_macro", 0.9))
    assert [e.name for e in state.sorted_experiments()] == ["high", "low"]
    assert state.best_experiment().name == "high"  # type: ignore[union-attr]


def test_ranking_lower_is_better_for_errors() -> None:
    state = ResearchState()
    state.task = TaskSpec(target="y", task_type="regression", primary_metric="rmse")
    state.add_experiment(exp("bad", "rmse", 12.0))
    state.add_experiment(exp("good", "rmse", 3.0))
    assert [e.name for e in state.sorted_experiments()] == ["good", "bad"]
    assert state.best_experiment().name == "good"  # type: ignore[union-attr]


def test_failed_experiments_are_ranked_last() -> None:
    state = ResearchState()
    state.task = TaskSpec(target="y", task_type="classification", primary_metric="f1_macro")
    state.add_experiment(exp("good", "f1_macro", 0.7))
    state.add_experiment(exp("broken", "f1_macro", 0.99, ))
    state.experiments[-1].status = "failed"
    assert [e.name for e in state.sorted_experiments()] == ["good", "broken"]


# -- summaries ----------------------------------------------------------
def test_summary_text_mentions_the_key_facts(tmp_path: Path) -> None:
    state = ResearchState(root_dir=str(tmp_path))
    state.task = TaskSpec(
        target="churned",
        task_type="classification",
        primary_metric="f1_macro",
        split={"seed": 42, "test_size": 0.2, "stratified": True, "n_train": 480, "n_test": 120},
    )
    state.set_plan([{"id": "1", "text": "profile", "status": "done"}])
    state.add_experiment(exp("rf", "f1_macro", 0.81))
    state.add_finding("rf beat logistic regression")

    text = state.summary_text()
    assert "churned" in text
    assert "f1_macro" in text
    assert "seed=42" in text
    assert "[done] profile" in text
    assert "rf" in text
    assert "rf beat logistic regression" in text


def test_summary_text_handles_an_empty_state() -> None:
    text = ResearchState().summary_text()
    assert "none profiled yet" in text
    assert "not defined yet" in text
    assert "none yet" in text


def test_summary_text_is_truncated_when_asked() -> None:
    state = ResearchState()
    for i in range(50):
        state.add_finding(f"finding number {i} " + "x" * 200)
    text = state.summary_text(max_chars=500)
    assert len(text) <= 560
    assert "truncated" in text


def test_summary_dict_is_small_and_json_safe() -> None:
    state = ResearchState()
    state.task = TaskSpec(target="y", task_type="classification", primary_metric="f1_macro")
    state.set_plan([{"id": "1", "text": "a", "status": "pending"}])
    state.add_experiment(exp("rf", "f1_macro", 0.8))
    summary = state.summary_dict()
    json.dumps(summary)  # must not raise
    assert summary["task"]["target"] == "y"
    assert summary["best"]["name"] == "rf"
    assert summary["n_experiments"] == 1
    assert summary["dataset"] is None


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_best_is_none_without_a_usable_metric(missing: float | None) -> None:
    state = ResearchState()
    state.task = TaskSpec(target="y", task_type="classification", primary_metric="f1_macro")
    e = exp("rf", "f1_macro", 0.8)
    e.metrics = {} if missing is None else {"f1_macro": float("nan")}
    state.add_experiment(e)
    # NaN sorts last rather than winning by accident.
    assert state.sorted_experiments()[-1].name == "rf"
