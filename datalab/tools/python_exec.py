"""Persistent-namespace Python executor (notebook semantics).

One namespace per session: variables survive between cells, exactly like a
notebook. This runs LLM-generated code **in the app process, unsandboxed** — see
NOTES.md. The soft timeout cannot kill a running thread, so a timed-out cell
leaves its thread running; the next cell must cope with that.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import io
import traceback
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

from .base import ToolResult  # noqa: E402

MAX_VALUE_CHARS = 2_000
MAX_FIGURES_PER_CELL = 8
_RESERVED = {
    "__name__",
    "pd",
    "np",
    "plt",
    "lab",
    "SESSION_DIR",
    "DATA_DIR",
    "OUTPUT_DIR",
    "FIGURES_DIR",
    "__builtins__",
}


def _short_repr(value: Any) -> str:
    """A compact, notebook-like repr: previews for pandas, truncates everything else."""
    try:
        if isinstance(value, pd.DataFrame):
            with pd.option_context("display.max_columns", 12, "display.width", 160):
                body = value.head(10).to_string()
            if len(value) > 10:
                body += f"\n[{len(value):,} rows x {value.shape[1]} cols]"
            return body
        if isinstance(value, pd.Series):
            body = value.head(20).to_string()
            if len(value) > 20:
                body += f"\n[... {len(value):,} values]"
            header = f"{value.name}\n" if value.name is not None else ""
            return header + body
        if isinstance(value, np.ndarray) and value.size > 50:
            return f"ndarray shape={value.shape} dtype={value.dtype}\n{value.reshape(-1)[:20]}"
        text = repr(value)
    except Exception as exc:  # noqa: BLE001 - repr must never break a cell
        return f"<unreprable {type(value).__name__}: {exc}>"
    if len(text) > MAX_VALUE_CHARS:
        text = text[:MAX_VALUE_CHARS] + f" ... (truncated, {len(text)} chars total)"
    return text


def split_code(code: str) -> tuple[str, str | None]:
    """Return ``(code_to_exec, trailing_expression)`` for notebook semantics.

    A cell whose last statement is a bare expression behaves like a notebook
    cell: the expression is evaluated separately so its value can be reported.
    A cell that fails to parse is executed as-is and will raise on compile.
    """
    try:
        module = ast.parse(code)
    except SyntaxError:
        return code, None
    if not module.body:
        return code, None
    last = module.body[-1]
    if isinstance(last, ast.Expr):
        head = ast.Module(body=module.body[:-1], type_ignores=[])
        prefix = ast.unparse(head) if head.body else ""
        try:
            return prefix, ast.unparse(last.value)
        except Exception:  # noqa: BLE001 - fall back to plain exec
            return code, None
    return code, None


class PythonExecutor:
    """Executes cells in one persistent namespace for the life of a session."""

    def __init__(
        self,
        *,
        session_dir: Path,
        data_dir: Path,
        output_dir: Path,
        figures_dir: Path,
        lab: Any,
    ) -> None:
        self.session_dir = Path(session_dir)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self._cell_index = 0
        self.last_result: ToolResult | None = None
        self.namespace: dict[str, Any] = {
            "__name__": "__main__",
            "pd": pd,
            "np": np,
            "plt": plt,
            "lab": lab,
            "SESSION_DIR": str(self.session_dir),
            "DATA_DIR": str(self.data_dir),
            "OUTPUT_DIR": str(self.output_dir),
            "FIGURES_DIR": str(self.figures_dir),
        }

    # -- introspection --------------------------------------------------
    def variable_names(self) -> list[str]:
        """Public names currently in the namespace (used by `query_state`)."""
        return sorted(
            k for k in self.namespace if not k.startswith("_") and k not in _RESERVED
        )

    def history_note(self) -> str:
        return f"{self._cell_index} cell(s) executed so far in this session."

    # -- execution ------------------------------------------------------
    async def run(self, code: str, *, timeout_s: int = 300) -> ToolResult:
        """Run one cell. Never raises; failures come back as ``ToolResult(error=True)``."""
        self._cell_index += 1
        index = self._cell_index
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._run_sync, code, index), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            # A thread cannot be killed in-process; the cell may keep running.
            return ToolResult.fail(
                f"Cell timed out after {timeout_s}s and was abandoned. The worker thread could "
                "not be killed, so it may still be running in the background. Use a smaller data "
                "sample, fewer models, or split the work into smaller cells, then try again.",
                data={"cell": index, "timed_out": True},
            )
        except Exception as exc:  # noqa: BLE001 - defensive
            return ToolResult.fail(
                f"Executor error: {type(exc).__name__}: {exc}", data={"cell": index}
            )
        self.last_result = result
        return result

    # -- the blocking half (runs in a worker thread) --------------------
    def _run_sync(self, code: str, index: int) -> ToolResult:
        out, err = io.StringIO(), io.StringIO()
        figures_before = set(plt.get_fignums())
        value: Any = None
        tb = ""
        failed = False
        # NOTE: redirect_stdout/redirect_stderr are process-global. Acceptable for
        # a single-user local demo; not safe for concurrent sessions. See NOTES.md.
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                body, expr = split_code(code)
                if body.strip():
                    exec(compile(body, "<cell>", "exec"), self.namespace)  # noqa: S102
                if expr is not None:
                    value = eval(expr, self.namespace)  # noqa: S307
            except BaseException:  # noqa: BLE001 - every failure goes back to the model
                failed = True
                tb = traceback.format_exc(limit=12)
        images = self._save_figures(figures_before, index)
        text = self._compose(tb, out.getvalue(), err.getvalue(), value, failed=failed)
        result = ToolResult(
            text=text,
            error=failed,
            images=images,
            data={"cell": index, "timeout_s": None},
        )
        if tb:
            result.data["traceback"] = tb
        return result

    def _compose(self, tb: str, stdout: str, stderr: str, value: Any, *, failed: bool) -> str:
        parts: list[str] = []
        if stdout.strip():
            parts.append(stdout.rstrip())
        if not failed and value is not None:
            parts.append(_short_repr(value))
        if stderr.strip():
            parts.append("[stderr]\n" + stderr.rstrip())
        if tb:
            parts.append(tb.rstrip())
        if not parts:
            return "Cell finished with no output."
        return "\n\n".join(parts)

    def _save_figures(self, figures_before: set[int], index: int) -> list[str]:
        """Save and close every figure opened during the cell."""
        new = [n for n in sorted(set(plt.get_fignums()) - figures_before)]
        paths: list[str] = []
        for offset, num in enumerate(new[:MAX_FIGURES_PER_CELL], start=1):
            path = self.figures_dir / f"{index}_{offset}.png"
            try:
                plt.figure(num).savefig(path, dpi=110, bbox_inches="tight")
                paths.append(str(path))
            except Exception:  # noqa: BLE001 - a bad figure must not kill the cell
                continue
        for num in new:
            plt.close(num)
        return paths


# -- the `python` tool --------------------------------------------------
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "The Python code for this cell."},
        "description": {
            "type": "string",
            "description": "One short line shown as the step title, e.g. 'Train a random forest'.",
        },
        "est_seconds": {
            "type": "integer",
            "description": (
                "Your honest estimate of the runtime. REQUIRED for anything above ~30 seconds — "
                "the user is asked to approve slow cells, and the estimate is shown to them."
            ),
        },
    },
    "required": ["code", "description"],
}

DESCRIPTION = (
    "Run Python code in a persistent, notebook-style namespace shared across cells: variables "
    "defined in one cell are available in the next. Pre-loaded: pd, np, plt (matplotlib, Agg), lab, "
    "SESSION_DIR, DATA_DIR, OUTPUT_DIR, FIGURES_DIR. A trailing expression is evaluated and its "
    "value returned. Matplotlib figures are saved automatically and shown inline. Returns stdout, "
    "stderr, the trailing value and the traceback on error."
)


async def handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    """Run a cell through the session's executor."""
    code = args.get("code")
    if not isinstance(code, str) or not code.strip():
        return ToolResult.fail("python needs a non-empty 'code' string.")
    if ctx.executor is None:
        return ToolResult.fail("The Python executor is not available in this session.")
    result = await ctx.executor.run(code, timeout_s=ctx.settings.python_soft_timeout_s)
    result.data.setdefault("description", str(args.get("description") or ""))
    if result.error:
        result.text = (
            f"{result.text}\n\n[cell failed — read the traceback above, fix the cause, then rerun a corrected cell]"
        )
    return result
