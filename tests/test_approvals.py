"""Approval heuristics: what must ask, and what must not."""

from __future__ import annotations

from pathlib import Path

import pytest

from datalab.approvals import (
    find_destructive_issues,
    find_heavy_issues,
    needs_approval,
)
from datalab.llm import ToolCall
from datalab.state import DatasetInfo, ResearchState


def py_call(code: str, **args: object) -> ToolCall:
    return ToolCall(id="c1", name="python", arguments={"code": code, "description": "test", **args})


@pytest.fixture
def state(tmp_path: Path) -> ResearchState:
    s = ResearchState(root_dir=str(tmp_path))
    s.dataset = DatasetInfo(path=str(tmp_path / "data" / "raw.csv"))
    return s


@pytest.fixture
def session(tmp_path: Path) -> Path:
    root = tmp_path / "runs" / "s1"
    (root / "data").mkdir(parents=True)
    (root / "outputs").mkdir(parents=True)
    return root


def ask(state: ResearchState, session: Path, code: str, threshold: int = 60, **args: object):
    return needs_approval(py_call(code, **args), state, session_dir=session, threshold_seconds=threshold)


# -- rule 1: est_seconds ------------------------------------------------
def test_no_approval_for_a_quick_cell(state: ResearchState, session: Path) -> None:
    assert ask(state, session, "print(1)", est_seconds=5) is None


def test_approval_when_est_seconds_exceeds_threshold(state: ResearchState, session: Path) -> None:
    request = ask(state, session, "model.fit(X, y)", est_seconds=180)
    assert request is not None
    assert "180s" in request.reason
    assert request.code == "model.fit(X, y)"
    assert request.details["est_seconds"] == 180


def test_exactly_at_threshold_does_not_ask(state: ResearchState, session: Path) -> None:
    assert ask(state, session, "model.fit(X, y)", est_seconds=60) is None


def test_approval_threshold_is_configurable(state: ResearchState, session: Path) -> None:
    assert ask(state, session, "fit()", est_seconds=20, threshold=10) is not None


# -- rule 2: destructive ------------------------------------------------
@pytest.mark.parametrize(
    "code",
    [
        "import os\nos.remove('x.txt')",
        "os.unlink('x.txt')",
        "import shutil\nshutil.rmtree('build')",
        "Path('x').unlink()",
        "shutil.rmtree('build', ignore_errors=True)",
    ],
)
def test_destructive_calls_ask(state: ResearchState, session: Path, code: str) -> None:
    request = ask(state, session, code)
    assert request is not None
    assert "destructive" in request.title.lower()


def test_writing_to_the_uploaded_dataset_asks(state: ResearchState, session: Path) -> None:
    code = "df.to_csv('/tmp/whatever/runs/s1/data/raw.csv', index=False)"
    state.dataset = DatasetInfo(path=str(session / "data" / "raw.csv"))
    request = ask(state, session, code)
    assert request is not None
    assert "uploaded dataset" in request.reason


def test_writing_by_basename_to_the_dataset_asks(state: ResearchState, session: Path) -> None:
    code = "df.to_csv('raw.csv')"
    request = ask(state, session, code)
    assert request is not None
    assert "uploaded dataset" in request.reason


def test_writing_outside_the_session_folder_asks(state: ResearchState, session: Path) -> None:
    request = ask(state, session, "df.to_csv('/home/me/important.csv', index=False)")
    assert request is not None
    assert "outside the session folder" in request.reason


def test_writing_inside_outputs_does_not_ask(state: ResearchState, session: Path) -> None:
    assert ask(state, session, "df.to_csv(OUTPUT_DIR + '/clean.csv', index=False)") is None
    assert ask(state, session, "fig.savefig('plots/hist.png')") is None
    assert ask(state, session, "df.to_parquet('cleaned.parquet')") is None


def test_open_in_write_mode_outside_asks(state: ResearchState, session: Path) -> None:
    assert ask(state, session, "open('/etc/hosts', 'w').write('x')") is not None


def test_reading_is_fine(state: ResearchState, session: Path) -> None:
    assert ask(state, session, "df = pd.read_csv(DATA_DIR + '/raw.csv')") is None
    assert ask(state, session, "text = open('notes.txt').read()") is None


def test_find_destructive_issues_reports_nothing_for_plain_code(session: Path) -> None:
    assert find_destructive_issues("df.head()\nx = 1 + 1", session, []) == []


# -- rule 3: write_file overwrite ---------------------------------------
def test_write_file_new_path_is_fine(state: ResearchState, session: Path) -> None:
    call = ToolCall(id="c", name="write_file", arguments={"path": "report.md", "content": "hi"})
    assert needs_approval(call, state, session_dir=session) is None


def test_write_file_overwrite_asks(state: ResearchState, session: Path) -> None:
    (session / "outputs").mkdir(exist_ok=True)
    (session / "outputs" / "report.md").write_text("old content", encoding="utf-8")
    call = ToolCall(id="c", name="write_file", arguments={"path": "report.md", "content": "new"})
    request = needs_approval(call, state, session_dir=session)
    assert request is not None
    assert "already exists" in request.reason


def test_write_file_outside_session_asks(state: ResearchState, session: Path) -> None:
    call = ToolCall(id="c", name="write_file", arguments={"path": "/tmp/evil.py", "content": "x"})
    request = needs_approval(call, state, session_dir=session)
    assert request is not None
    assert "outside the session folder" in request.reason


# -- rule 4: heavy compute ----------------------------------------------
def test_grid_search_without_estimate_asks(state: ResearchState, session: Path) -> None:
    code = "from sklearn.model_selection import GridSearchCV\ngs = GridSearchCV(model, {'n_estimators':[10,20,30]}, cv=5)\ngs.fit(X, y)"
    request = ask(state, session, code)
    assert request is not None
    assert "Heavy compute" in request.title
    assert "est_seconds" in request.reason


def test_grid_search_with_estimate_uses_the_estimate(state: ResearchState, session: Path) -> None:
    code = "GridSearchCV(model, {'n_estimators':[10,20]}, cv=3)"
    request = ask(state, session, code, est_seconds=5)
    assert request is not None
    assert "Estimated runtime" in request.reason


def test_small_random_forest_does_not_ask(state: ResearchState, session: Path) -> None:
    assert ask(state, session, "RandomForestClassifier(n_estimators=100).fit(X, y)") is None


def test_thousands_of_estimators_asks(state: ResearchState, session: Path) -> None:
    request = ask(state, session, "RandomForestClassifier(n_estimators=5000).fit(X, y)")
    assert request is not None
    assert "thousands" in request.reason


def test_grid_size_is_estimated_for_the_prompt(state: ResearchState, session: Path) -> None:
    code = (
        "from sklearn.model_selection import RandomizedSearchCV\n"
        "gs = RandomizedSearchCV(model, {'n_estimators': [10,20,30,40], 'depth': [2,3,4,5]}, cv=3)"
    )
    request = ask(state, session, code)
    assert request is not None
    assert request.details["grid_size"] == 16
    assert "16 combinations" in request.reason


def test_find_heavy_issues_is_empty_for_cheap_code() -> None:
    assert find_heavy_issues("LinearRegression().fit(X, y)") == []


# -- other tools --------------------------------------------------------
def test_other_tools_never_ask(state: ResearchState, session: Path) -> None:
    for name in ("profile_dataset", "read_file", "list_files", "todo", "query_state", "ask_user"):
        call = ToolCall(id="c", name=name, arguments={"path": "x"})
        assert needs_approval(call, state, session_dir=session) is None


# -- new destructive patterns: process execution ------------------------
@pytest.mark.parametrize(
    "code",
    [
        'os.system("rm -rf x")',
        'subprocess.run(["rm", "-rf", "x"])',
        'subprocess.call(["rm", "-rf", "x"])',
        'subprocess.Popen(["rm", "-rf", "x"])',
        'subprocess.check_output(["ls"])',
        'os.popen("ls")',
    ],
)
def test_process_execution_asks(state: ResearchState, session: Path, code: str) -> None:
    request = ask(state, session, code)
    assert request is not None
    assert "destructive" in request.title.lower()


# -- new destructive patterns: rename / mkdir / chmod -------------------
@pytest.mark.parametrize(
    "code",
    [
        "os.rename(a, b)",
        "os.replace(a, b)",
        "os.renames(a, b)",
        'os.mkdir("newdir")',
        'os.makedirs("a/b")',
        'os.chmod("f", 0o644)',
        'os.chown("f", 0, 0)',
        'os.truncate("f", 0)',
        "os._exit(0)",
    ],
)
def test_filesystem_mutation_asks(state: ResearchState, session: Path, code: str) -> None:
    request = ask(state, session, code)
    assert request is not None
    assert "destructive" in request.title.lower()


# -- new destructive patterns: obfuscated calls --------------------------
@pytest.mark.parametrize(
    "code",
    [
        '__import__("os").remove("x")',
        'getattr(os, "remove")(p)',
        'getattr(__builtins__, "eval")("1+1")',
        'eval("os.remove(\'x\')")',
        'exec("os.remove(\'x\')")',
    ],
)
def test_obfuscated_destructive_calls_ask(
    state: ResearchState, session: Path, code: str
) -> None:
    request = ask(state, session, code)
    assert request is not None
    assert "destructive" in request.title.lower()


def test_path_division_traversal_asks(state: ResearchState, session: Path) -> None:
    request = ask(state, session, 'Path(DATA_DIR) / ".." / ".." / "x"')
    assert request is not None
    assert "destructive" in request.title.lower()


def test_fstring_write_outside_asks(state: ResearchState, session: Path) -> None:
    request = ask(state, session, 'df.to_csv(f"{DATA_DIR}/../data/raw.csv")')
    assert request is not None
    assert "destructive" in request.title.lower()


def test_open_without_visible_literal_asks(state: ResearchState, session: Path) -> None:
    request = ask(state, session, 'm = "w"; open(p, mode=m)')
    assert request is not None
    assert "destructive" in request.title.lower()


def test_open_variable_in_read_mode_does_not_ask(
    state: ResearchState, session: Path
) -> None:
    assert ask(state, session, "data = open(p).read()") is None
    assert ask(state, session, "data = open(p, 'r')") is None


# -- new destructive patterns: shutil copies outside ----------------------
@pytest.mark.parametrize(
    "code",
    [
        'shutil.copy(src, "/etc/x")',
        'shutil.move(src, "/etc/y")',
        'shutil.copytree(src, "/etc/z")',
    ],
)
def test_shutil_copy_outside_asks(state: ResearchState, session: Path, code: str) -> None:
    request = ask(state, session, code)
    assert request is not None
    assert "destructive" in request.title.lower()


def test_shutil_copy_with_bare_variables_does_not_ask(
    state: ResearchState, session: Path
) -> None:
    assert ask(state, session, "shutil.copy(a, b)") is None


def test_write_with_bare_variable_does_not_crash_or_ask(
    state: ResearchState, session: Path
) -> None:
    assert ask(state, session, "df.to_csv(path_variable)") is None


# -- new heavy-compute patterns --------------------------------------------
@pytest.mark.parametrize(
    "code",
    [
        "RandomForestClassifier(n_estimators=10**5).fit(X, y)",
        "RandomForestClassifier(n_estimators=int(1e5)).fit(X, y)",
        "RandomForestClassifier(n_estimators=5000).fit(X, y)",
        'params = {"n_estimators": 100000}\nClf(**params)',
    ],
)
def test_large_n_estimators_asks(state: ResearchState, session: Path, code: str) -> None:
    request = ask(state, session, code)
    assert request is not None
    assert "Heavy compute" in request.title


# -- safe code from the Task 2 list does not ask -----------------------------
def test_safe_code_does_not_ask(state: ResearchState, session: Path) -> None:
    safe = [
        "df.to_csv('out.csv')",
        "df.to_csv('cleaned.parquet')",
        "df.to_csv(OUTPUT_DIR + '/clean.csv', index=False)",
        "df.to_csv(os.path.join(OUTPUT_DIR, 'clean.csv'))",
        "pd.read_csv(DATA_DIR + '/data.csv')",
        "open('notes.txt').read()",
        "open('x.csv').read()",
        "plt.savefig('plots/hist.png')",
        "plt.savefig(FIGURES_DIR + '/h.png')",
        "df.to_parquet('cleaned.parquet')",
        "open(OUTPUT_DIR + '/report.md', 'w').write(text)",
        "RandomForestClassifier(n_estimators=100).fit(X, y)",
        "LogisticRegression(max_iter=2000)",
        "cross_val_score(model, X, y, cv=5)",
        "json.dump(results, open(OUTPUT_DIR + '/r.json', 'w'))",
        "pd.read_excel(DATA_DIR + '/d.xlsx')",
        "print('hello')\nx = 1 + 1\nplt.plot([1, 2])",
    ]
    for code in safe:
        assert ask(state, session, code) is None, f"false positive for: {code!r}"


# -- dialog quality: the reason must name the risk ------------------------------
def test_reason_names_the_risk(state: ResearchState, session: Path) -> None:
    request = ask(state, session, 'os.system("rm -rf x")')
    assert request is not None
    assert "os.system" in request.reason
    heavy = ask(state, session, "RandomForestClassifier(n_estimators=5000).fit(X, y)")
    assert heavy is not None
    assert "n_estimators" in heavy.reason


# -- totality: broken cells never block the agent ----------------------------------
@pytest.mark.parametrize("code", ["def broken(:", "((((", ""])
def test_broken_cell_returns_none(
    state: ResearchState, session: Path, code: str
) -> None:
    assert ask(state, session, code) is None
