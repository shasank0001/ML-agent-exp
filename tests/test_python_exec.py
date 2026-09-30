"""Executor semantics: persistence, notebook return values, errors, figures, timeouts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pytest

from datalab.config import Settings
from datalab.lab import LabSession
from datalab.state import ResearchState
from datalab.tools import python_exec
from datalab.tools.base import ToolContext
from datalab.tools.python_exec import PythonExecutor, split_code


@pytest.fixture
def executor(tmp_path: Path) -> PythonExecutor:
    state = ResearchState(root_dir=str(tmp_path / "s"))
    lab = LabSession(state, data_dir=tmp_path / "data", output_dir=tmp_path / "out")
    lab.data_dir.mkdir(parents=True, exist_ok=True)
    lab.output_dir.mkdir(parents=True, exist_ok=True)
    return PythonExecutor(
        session_dir=tmp_path / "s",
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "out",
        figures_dir=tmp_path / "fig",
        lab=lab,
    )


def _context_for(executor: PythonExecutor) -> ToolContext:
    return ToolContext(
        session_id="s",
        paths={
            "root": executor.session_dir,
            "data": executor.data_dir,
            "outputs": executor.output_dir,
            "figures": executor.figures_dir,
        },
        state=ResearchState(root_dir=str(executor.session_dir)),
        settings=Settings(
            provider="lmstudio",
            base_url="http://x/v1",
            api_key="k",
            model="m",
            max_steps=5,
            max_repairs=2,
            approval_seconds_threshold=60,
            python_soft_timeout_s=30,
        ),
        logger=None,  # type: ignore[arg-type]
        executor=executor,
    )


# -- split_code ---------------------------------------------------------
def test_split_code_extracts_trailing_expression() -> None:
    body, expr = split_code("a = 1\na + 1\n")
    assert expr == "a + 1"
    assert body.strip() == "a = 1"


def test_split_code_keeps_assignment_as_body() -> None:
    body, expr = split_code("a = 1\n")
    assert expr is None
    assert "a = 1" in body


def test_split_code_empty() -> None:
    assert split_code("") == ("", None)


def test_split_code_syntax_error_passes_through() -> None:
    body, expr = split_code("def (:")
    assert body == "def (:"
    assert expr is None


# -- execution ----------------------------------------------------------
async def test_variables_persist_between_cells(executor: PythonExecutor) -> None:
    first = await executor.run("total = 21 * 2")
    assert not first.error
    second = await executor.run("total + 19")
    assert second.text.strip() == "61"
    assert first.text == "Cell finished with no output."


async def test_stdout_is_captured(executor: PythonExecutor) -> None:
    result = await executor.run("print('hello')\nprint('world')")
    assert "hello" in result.text and "world" in result.text
    assert not result.error


async def test_trailing_dataframe_value_is_a_preview(executor: PythonExecutor) -> None:
    result = await executor.run("import pandas as pd\ndf = pd.DataFrame({'a': range(50)})\ndf")
    assert not result.error
    assert "50 rows x 1 cols" in result.text
    assert len(result.text) < 3_000


async def test_trailing_series_value(executor: PythonExecutor) -> None:
    result = await executor.run("import pandas as pd\ns = pd.Series(range(5), name='v')\ns")
    assert "v" in result.text


async def test_error_returns_traceback_not_exception(executor: PythonExecutor) -> None:
    result = await executor.run("raise ValueError('boom')")
    assert result.error
    assert "ValueError: boom" in result.text
    assert "Traceback" in result.text
    assert "raised ValueError" not in result.text  # it is a cell, not a tool


async def test_handler_tells_the_model_to_fix_the_cause(executor: PythonExecutor) -> None:
    ctx = _context_for(executor)
    result = await python_exec.handler(
        {"code": "raise ValueError('boom')", "description": "try something"}, ctx
    )
    assert result.error
    assert "fix the cause" in result.text
    assert result.data["description"] == "try something"


async def test_handler_rejects_empty_code(executor: PythonExecutor) -> None:
    result = await python_exec.handler({"code": "   "}, _context_for(executor))
    assert result.error
    assert "non-empty" in result.text


async def test_namespace_survives_an_error(executor: PythonExecutor) -> None:
    await executor.run("kept = 5")
    result = await executor.run("1/0")
    assert result.error
    after = await executor.run("kept * 3")
    assert after.text.strip() == "15"


async def test_preloaded_names_exist(executor: PythonExecutor) -> None:
    for name in ("pd", "np", "plt", "lab", "SESSION_DIR", "DATA_DIR", "OUTPUT_DIR", "FIGURES_DIR"):
        assert name in executor.namespace
    assert executor.namespace["OUTPUT_DIR"] == str(executor.output_dir)


async def test_figures_are_saved_and_closed(executor: PythonExecutor) -> None:
    result = await executor.run(
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\n"
        "ax.plot([1, 2, 3], [1, 4, 9])\n"
    )
    assert not result.error
    assert len(result.images) == 1
    assert Path(result.images[0]).exists()
    assert result.images[0].startswith(str(executor.figures_dir))
    import matplotlib.pyplot as plt

    assert plt.get_fignums() == []


async def test_two_figures_get_distinct_files(executor: PythonExecutor) -> None:
    result = await executor.run(
        "import matplotlib.pyplot as plt\nplt.plot([1,2]); plt.figure(); plt.plot([3,4])"
    )
    assert len(result.images) == 2
    assert result.images[0] != result.images[1]
    assert all(Path(p).exists() for p in result.images)


async def test_soft_timeout_returns_an_explanation(tmp_path: Path) -> None:
    state = ResearchState(root_dir=str(tmp_path / "s"))
    lab = LabSession(state, data_dir=tmp_path / "d", output_dir=tmp_path / "o")
    ex = PythonExecutor(
        session_dir=tmp_path / "s",
        data_dir=tmp_path / "d",
        output_dir=tmp_path / "o",
        figures_dir=tmp_path / "f",
        lab=lab,
    )
    result = await ex.run("import time\ntime.sleep(5)", timeout_s=1)
    assert result.error
    assert "timed out" in result.text.lower()
    assert "could not be killed" in result.text


async def test_stderr_is_reported_separately(executor: PythonExecutor) -> None:
    result = await executor.run("import sys\nsys.stderr.write('a warning\\n')")
    assert "[stderr]" in result.text
    assert "a warning" in result.text


async def test_variable_names_lists_user_names(executor: PythonExecutor) -> None:
    await executor.run("my_frame = 1")
    names = executor.variable_names()
    assert "my_frame" in names
    assert "pd" not in names
