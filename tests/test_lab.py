"""LabSession ground truth: task inference, metrics, split, evaluate, baselines, results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVC

from datalab.lab import (
    LabSession,
    compute_metrics,
    default_primary_metric,
    infer_task_type,
    is_higher_better,
)
from datalab.state import Experiment, ResearchState


def make_lab(tmp_path: Path) -> LabSession:
    return LabSession(ResearchState(), data_dir=tmp_path, output_dir=tmp_path / "out")


def classification_frame(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.normal(40, 12, n).round(1),
            "spend": rng.gamma(2.0, 30.0, n).round(2),
            "plan": rng.choice(["basic", "plus", "pro"], n),
            "target": rng.choice([0, 1], n, p=[0.75, 0.25]),
        }
    )
    df.loc[rng.choice(n, 15, replace=False), "age"] = np.nan
    df.loc[rng.choice(n, 10, replace=False), "plan"] = np.nan
    return df


def regression_frame(n: int = 160, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "price": rng.normal(size=n) * 10 + 50,
        }
    )


# -- infer_task_type ----------------------------------------------------
def test_string_target_is_classification() -> None:
    assert infer_task_type(pd.Series(["yes", "no", "yes"])) == "classification"


def test_bool_target_is_classification() -> None:
    assert infer_task_type(pd.Series([True, False, True])) == "classification"


def test_categorical_target_is_classification() -> None:
    assert infer_task_type(pd.Series(pd.Categorical(["a", "b", "a"]))) == "classification"


def test_binary_int_target_is_classification() -> None:
    assert infer_task_type(pd.Series([0, 1, 0, 1, 1])) == "classification"


def test_small_int_range_is_classification() -> None:
    assert infer_task_type(pd.Series(range(11))) == "classification"


def test_continuous_float_is_regression() -> None:
    series = pd.Series(np.random.default_rng(0).normal(size=300))
    assert infer_task_type(series) == "regression"


def test_float_zero_one_is_classification() -> None:
    assert infer_task_type(pd.Series([0.0, 1.0, 0.0, 1.0])) == "classification"


def test_constant_column_does_not_crash() -> None:
    assert infer_task_type(pd.Series([5] * 20)) == "classification"
    assert infer_task_type(pd.Series([5.0] * 20)) == "classification"
    assert infer_task_type(pd.Series(["same"] * 20)) == "classification"


# -- default_primary_metric / is_higher_better --------------------------
def test_default_primary_metric_for_each_task() -> None:
    assert default_primary_metric("classification") == "f1_macro"
    assert default_primary_metric("regression") == "rmse"


def test_default_primary_metric_rejects_unknown_tasks() -> None:
    with pytest.raises(ValueError, match="unknown task_type"):
        default_primary_metric("clustering")


@pytest.mark.parametrize("metric", ["accuracy", "f1_macro", "roc_auc", "r2"])
def test_higher_is_better_for_scores(metric: str) -> None:
    assert is_higher_better(metric) is True


@pytest.mark.parametrize("metric", ["rmse", "mae", "log_loss"])
def test_lower_is_better_for_errors(metric: str) -> None:
    assert is_higher_better(metric) is False


# -- compute_metrics ----------------------------------------------------
def test_perfect_classification_scores_one() -> None:
    y = [0, 1, 1, 0, 1]
    proba = np.array([[0.9, 0.1], [0.1, 0.9], [0.2, 0.8], [0.8, 0.2], [0.05, 0.95]])
    metrics = compute_metrics(y, y, task_type="classification", y_proba=proba)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1_macro"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_degenerate_single_class_does_not_raise() -> None:
    proba = np.array([[0.1, 0.9]] * 4)
    metrics = compute_metrics([1, 1, 1, 1], [1, 1, 1, 1], task_type="classification", y_proba=proba)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert "roc_auc" not in metrics  # single-class roc_auc is omitted, never raised


def test_regression_metrics_match_hand_computation() -> None:
    y = np.array([3.0, -0.5, 2.0, 7.0])
    pred = np.array([2.5, 0.0, 2.0, 8.0])
    metrics = compute_metrics(y, pred, task_type="regression")
    assert metrics["rmse"] == pytest.approx(float(np.sqrt(mean_squared_error(y, pred))))
    assert metrics["rmse"] == pytest.approx(float(np.sqrt(np.mean((y - pred) ** 2))))
    assert metrics["mae"] == pytest.approx(float(mean_absolute_error(y, pred)))
    assert metrics["r2"] == pytest.approx(float(r2_score(y, pred)))


def test_compute_metrics_rejects_unknown_tasks() -> None:
    with pytest.raises(ValueError, match="unknown task_type"):
        compute_metrics([0, 1], [0, 1], task_type="clustering")


# -- split: classification ----------------------------------------------
def test_classification_split_is_stratified_and_consistent(tmp_path: Path) -> None:
    lab = make_lab(tmp_path)
    df = classification_frame()
    X_train, X_test, y_train, y_test = lab.split(df, "target")
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)
    assert lab.state.task is not None and lab.state.task.task_type == "classification"
    assert lab.state.task.split["stratified"] is True
    overall, trained = float(df["target"].mean()), float(y_train.mean())
    assert abs(overall - trained) < 0.05


def test_split_frames_are_numeric_without_missing_values(tmp_path: Path) -> None:
    lab = make_lab(tmp_path)
    X_train, X_test, _, _ = lab.split(classification_frame(), "target")
    assert all(pd.api.types.is_numeric_dtype(dt) for dt in X_train.dtypes)
    assert all(pd.api.types.is_numeric_dtype(dt) for dt in X_test.dtypes)
    assert not bool(X_train.isna().any().any())
    assert not bool(X_test.isna().any().any())


def test_split_does_not_leak_target_or_index(tmp_path: Path) -> None:
    lab = make_lab(tmp_path)
    X_train, _, _, _ = lab.split(classification_frame(), "target")
    assert "target" not in X_train.columns
    assert "index" not in X_train.columns
    assert not any(str(c).startswith(("index", "level_0", "unnamed")) for c in X_train.columns)


def test_regression_split_records_task_and_shapes(tmp_path: Path) -> None:
    lab = make_lab(tmp_path)
    df = regression_frame()
    X_train, X_test, y_train, y_test = lab.split(df, "price")
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)
    assert lab.state.task is not None and lab.state.task.task_type == "regression"
    assert lab.state.task.primary_metric == "rmse"


def test_split_config_lands_in_state(tmp_path: Path) -> None:
    lab = make_lab(tmp_path)
    lab.split(classification_frame(), "target", test_size=0.2, seed=42)
    split = lab.state.task.split  # type: ignore[union-attr]
    assert split["seed"] == 42
    assert split["test_size"] == 0.2
    assert split["stratified"] is True
    assert split["n_train"] == len(lab.X_train)  # type: ignore[arg-type]
    assert split["n_test"] == len(lab.X_test)  # type: ignore[arg-type]
    assert split["n_train"] + split["n_test"] == 240


def test_split_with_same_args_is_idempotent(tmp_path: Path) -> None:
    lab = make_lab(tmp_path)
    df = classification_frame()
    first = lab.split(df, "target")
    before = dict(lab.state.task.split)  # type: ignore[union-attr]
    second = lab.split(df, "target")
    assert first[0] is second[0]  # cached frames are reused
    assert lab.state.task.split == before  # type: ignore[union-attr]
    assert lab.state.experiments == []  # splitting never logs experiments


def test_split_with_a_new_seed_recreates_the_split(tmp_path: Path) -> None:
    lab = make_lab(tmp_path)
    df = classification_frame()
    old_X, _, _, _ = lab.split(df, "target", seed=42)
    new_X, _, _, _ = lab.split(df, "target", seed=7)
    assert lab.state.task.split["seed"] == 7  # type: ignore[union-attr]
    assert new_X is not old_X
    assert not new_X.equals(old_X)


def test_single_member_class_does_not_crash_stratify(tmp_path: Path) -> None:
    lab = make_lab(tmp_path)
    df = pd.DataFrame({"a": range(20), "t": [0] * 19 + [1]})
    X_train, X_test, y_train, y_test = lab.split(df, "t")
    assert len(X_train) + len(X_test) == 20
    assert len(y_train) + len(y_test) == 20
    assert lab.state.task.split["stratified"] is False  # type: ignore[union-attr]


def test_hostile_frame_stays_numeric_and_compact(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    n = 200
    df = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c", None], n),
            "when": pd.date_range("2020-01-01", periods=n).astype(str),
            "hc": [f"level_{i}" for i in rng.integers(0, 80, n)],
            "target": rng.integers(0, 2, n),
        }
    )
    df.loc[5, "num"] = np.nan
    lab = make_lab(tmp_path)
    X_train, X_test, _, _ = lab.split(df, "target")
    assert all(pd.api.types.is_numeric_dtype(dt) for dt in X_train.dtypes)
    assert not bool(X_train.isna().any().any())
    assert not bool(X_test.isna().any().any())
    assert "target" not in X_train.columns
    assert X_train.shape[1] <= 12  # high-cardinality stays one ordinal column


def test_split_rejects_a_missing_target(tmp_path: Path) -> None:
    lab = make_lab(tmp_path)
    with pytest.raises(ValueError, match="not in columns"):
        lab.split(classification_frame(), "nope")


# -- evaluate -----------------------------------------------------------
def test_evaluate_logs_an_experiment_with_metrics(tmp_path: Path) -> None:
    lab = make_lab(tmp_path)
    lab.split(classification_frame(), "target")
    metrics = lab.evaluate(LogisticRegression(max_iter=2000), "logreg")
    assert lab.state.task is not None and lab.state.task.primary_metric in metrics
    assert len(lab.state.experiments) == 1
    logged = lab.state.experiments[0]
    assert logged.name == "logreg"
    assert logged.model == "LogisticRegression"
    assert logged.primary_metric == lab.state.task.primary_metric
    assert logged.status == "done"
    assert logged.created_at.strip() != ""


def test_evaluating_the_same_model_twice_refits_cleanly(tmp_path: Path) -> None:
    lab = make_lab(tmp_path)
    lab.split(classification_frame(), "target")
    model = DummyClassifier(strategy="most_frequent")
    first = lab.evaluate(model, "dummy-1")
    second = lab.evaluate(model, "dummy-2")  # already fitted: reuse path
    assert first == second
    assert [e.name for e in lab.state.experiments] == ["dummy-1", "dummy-2"]


def test_failing_fit_propagates_and_logs_nothing(tmp_path: Path) -> None:
    lab = make_lab(tmp_path)
    lab.split(classification_frame(), "target")

    class BadFit:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            raise NotFittedError("not fitted yet")

        def fit(self, X: pd.DataFrame, y: pd.Series) -> "BadFit":
            raise ValueError("incompatible data boom")

    with pytest.raises(ValueError, match="incompatible data boom"):
        lab.evaluate(BadFit(), "badfit")
    assert lab.state.experiments == []  # real behaviour: the error propagates, state untouched


def test_evaluate_without_a_split_raises_clearly(tmp_path: Path) -> None:
    lab = make_lab(tmp_path)
    with pytest.raises(RuntimeError, match="no split yet"):
        lab.evaluate(DummyClassifier(strategy="most_frequent"), "x")


def test_model_without_probabilities_omits_roc_auc(tmp_path: Path) -> None:
    lab = make_lab(tmp_path)
    lab.split(classification_frame(), "target")
    metrics = lab.evaluate(SVC(), "svc")
    assert "accuracy" in metrics and "f1_macro" in metrics
    assert "roc_auc" not in metrics


def test_multiclass_reports_ovr_roc_auc(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    df = pd.DataFrame({"a": rng.normal(size=150), "t": rng.choice(["x", "y", "z"], 150)})
    lab = make_lab(tmp_path)
    lab.split(df, "t")
    metrics = lab.evaluate(LogisticRegression(max_iter=2000), "logreg-mc")
    assert "roc_auc" in metrics  # real behaviour: macro ovr roc_auc is reported
    assert 0.0 <= metrics["roc_auc"] <= 1.0


# -- baseline / results -------------------------------------------------
def test_baseline_logs_exactly_two_experiments(tmp_path: Path) -> None:
    lab = make_lab(tmp_path)
    lab.split(classification_frame(), "target")
    out = lab.baseline()
    assert sorted(out) == ["baseline_dummy", "baseline_logreg"]
    assert len(lab.state.experiments) == 2
    assert all(lab.state.task.primary_metric in m for m in out.values())  # type: ignore[union-attr]


def test_regression_baseline_uses_rmse_pair(tmp_path: Path) -> None:
    lab = make_lab(tmp_path)
    lab.split(regression_frame(), "price")
    out = lab.baseline()
    assert sorted(out) == ["baseline_dummy", "baseline_ridge"]
    assert all("rmse" in m for m in out.values())


def test_results_table_is_best_first(tmp_path: Path) -> None:
    lab = make_lab(tmp_path)
    lab.split(classification_frame(), "target")
    for name, value in (("low", 0.4), ("high", 0.9), ("mid", 0.7)):
        lab.state.add_experiment(
            Experiment(id=name, name=name, model="M", metrics={"f1_macro": value}, primary_metric="f1_macro")
        )
    table = lab.results_table()
    assert list(table["name"]) == ["high", "mid", "low"]


def test_results_table_prefers_low_rmse_for_regression(tmp_path: Path) -> None:
    lab = make_lab(tmp_path)
    lab.split(regression_frame(), "price")
    for name, value in (("bad", 12.0), ("good", 3.0)):
        lab.state.add_experiment(
            Experiment(id=name, name=name, model="M", metrics={"rmse": value}, primary_metric="rmse")
        )
    assert list(lab.results_table()["name"]) == ["good", "bad"]


def test_empty_results_table_has_the_right_columns(tmp_path: Path) -> None:
    lab = make_lab(tmp_path)
    lab.split(classification_frame(), "target")
    table = lab.results_table()
    assert table.empty
    assert list(table.columns) == ["name", "model", "f1_macro", "notes", "created_at"]


def test_results_markdown_names_experiments_and_scores(tmp_path: Path) -> None:
    lab = make_lab(tmp_path)
    lab.split(classification_frame(), "target")
    lab.baseline()
    markdown = lab.results_markdown()
    assert markdown.startswith("| name |")
    assert "baseline_dummy" in markdown and "baseline_logreg" in markdown
    assert f"{lab.state.experiments[0].metrics['f1_macro']:.4f}" in markdown


def test_empty_results_markdown_says_so(tmp_path: Path) -> None:
    lab = make_lab(tmp_path)
    lab.split(classification_frame(), "target")
    assert lab.results_markdown() == "No experiments logged yet."


def test_set_primary_metric_reranks_and_rewrites_experiments(tmp_path: Path) -> None:
    lab = make_lab(tmp_path)
    lab.split(classification_frame(), "target")
    lab.state.add_experiment(
        Experiment(id="a", name="a", model="M", metrics={"f1_macro": 0.9, "roc_auc": 0.5}, primary_metric="f1_macro")
    )
    lab.state.add_experiment(
        Experiment(id="b", name="b", model="M", metrics={"f1_macro": 0.6, "roc_auc": 0.95}, primary_metric="f1_macro")
    )
    assert list(lab.results_table()["name"]) == ["a", "b"]
    assert lab.set_primary_metric("roc_auc") == "roc_auc"
    assert lab.state.task is not None and lab.state.task.primary_metric == "roc_auc"
    assert all(e.primary_metric == "roc_auc" for e in lab.state.experiments)
    assert list(lab.results_table()["name"]) == ["b", "a"]


def test_set_primary_metric_rejects_unknown_metrics(tmp_path: Path) -> None:
    lab = make_lab(tmp_path)
    lab.split(classification_frame(), "target")
    with pytest.raises(ValueError, match="unknown metric"):
        lab.set_primary_metric("bogus")


# -- load / feature_names -----------------------------------------------
def test_load_reads_csv_and_tsv(tmp_path: Path) -> None:
    df = classification_frame(n=20)
    (tmp_path / "a.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    (tmp_path / "b.tsv").write_text(df.to_csv(index=False, sep="\t"), encoding="utf-8")
    lab = make_lab(tmp_path)
    assert lab.load("a.csv").shape == df.shape
    assert lab.load("b.tsv").shape == df.shape


def test_load_rejects_bad_extensions_and_missing_files(tmp_path: Path) -> None:
    lab = make_lab(tmp_path)
    with pytest.raises(ValueError, match="unsupported extension"):
        lab.load("notes.txt")
    with pytest.raises(FileNotFoundError):
        lab.load("missing.csv")


def test_feature_names_match_the_encoded_frame(tmp_path: Path) -> None:
    lab = make_lab(tmp_path)
    lab.split(classification_frame(), "target")
    assert lab.X_train is not None
    assert len(lab.feature_names) == lab.X_train.shape[1]
    assert lab.feature_names == list(lab.X_train.columns)
