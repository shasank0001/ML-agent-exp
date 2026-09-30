"""Rule-based approval gate for costly or destructive actions.

Local demo, no sandbox: anything the heuristics flag is shown to the user as an
Approve / Deny question before it runs. The rules are deliberately conservative
and readable — they are heuristics on the code text, not a static analysis.
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .llm import ToolCall
from .state import ResearchState

# -- regex heuristics ---------------------------------------------------
DESTRUCTIVE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bos\s*\.\s*remove\s*\(", "calls os.remove()"),
    (r"\bos\s*\.\s*unlink\s*\(", "calls os.unlink()"),
    (r"\bshutil\s*\.\s*rmtree\s*\(", "calls shutil.rmtree()"),
    (r"\bos\s*\.\s*rmdir\s*\(", "calls os.rmdir()"),
    (r"\bos\s*\.\s*removedirs\s*\(", "calls os.removedirs()"),
    (r"\.unlink\s*\(", "calls .unlink() on a file"),
    (r"(?<![\w.])rmtree\s*\(", "calls rmtree()"),
    (r"\bpathlib\s*\.\s*Path\s*\([^)]*\)\s*\.\s*unlink\b", "unlinks a path"),
)

WRITE_CALLS: tuple[str, ...] = (
    "to_csv",
    "to_parquet",
    "to_excel",
    "to_json",
    "to_pickle",
    "to_hdf",
    "to_sql",
    "to_feather",
    "savefig",
    "savetxt",
    "write_text",
    "write_bytes",
)

HEAVY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bGridSearchCV\s*\(", "a GridSearchCV grid search"),
    (r"\bRandomizedSearchCV\s*\(", "a RandomizedSearchCV random search"),
    (r"\bHalvingRandomSearchCV\s*\(", "a HalvingRandomSearchCV search"),
    (r"\bOptuna\b", "an Optuna study"),
    (r"\bn_estimators\s*=\s*\d{4,}", "n_estimators in the thousands"),
    (r"\bn_estimators\s*=\s*\[[^\]]*,\s*\d{3,}[^\]]*\]", "a large n_estimators list"),
    (r"\bn_jobs\s*=\s*-\s*1", "n_jobs=-1 (uses every core)"),
    (r"\bdeepcopy\s*\(.*cross_val", "deep cross-validation"),
)

_STRING_LITERAL = re.compile(r"""["']([^"'\n]{1,300})["']""")
_DATA_SUFFIXES = (".csv", ".tsv", ".parquet", ".xlsx", ".xls", ".feather", ".h5", ".pkl", ".json")


@dataclass
class ApprovalRequest:
    """A question for the user, with enough context to decide."""

    id: str
    tool: str
    title: str
    reason: str
    code: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tool": self.tool,
            "title": self.title,
            "reason": self.reason,
            "code": self.code,
            "details": self.details,
        }


# -- small helpers ------------------------------------------------------
def _strings_in(text: str) -> list[str]:
    return _STRING_LITERAL.findall(text)


def _looks_like_path(value: str) -> bool:
    if not value:
        return False
    if value.startswith(("/", "~", "\\\\")) or re.match(r"^[A-Za-z]:[\\/]", value):
        return True
    if ".." in value.replace("\\", "/").split("/"):
        return True
    if "/" in value or "\\" in value:
        tail = re.split(r"[\\/]", value)[-1]
        return "." in tail
    return value.lower().endswith(_DATA_SUFFIXES)


def _outside_session(value: str, session_dir: Path) -> bool:
    """True when a literal path escapes the session directory."""
    if not value:
        return False
    norm = value.replace("\\", "/")
    if not _looks_like_path(norm):
        return False
    if re.match(r"^[a-zA-Z]+://", norm):  # URLs are not filesystem writes
        return False
    try:
        resolved = (session_dir / norm).resolve()
    except OSError:
        return True
    try:
        return not resolved.is_relative_to(session_dir.resolve())
    except (OSError, ValueError):
        return True


def _matches_dataset(value: str, dataset_paths: list[Path]) -> bool:
    if not value:
        return False
    norm = value.replace("\\", "/")
    for ds in dataset_paths:
        d = str(ds).replace("\\", "/")
        if norm == d or norm.endswith("/" + Path(d).name) or d.endswith("/" + norm):
            return True
    return False


def _arg_window(code: str, start: int, span: int = 260) -> str:
    """Rough source window following a call site, enough to see its arguments."""
    return code[start : start + span]


def _count_grid_combinations(code: str) -> int | None:
    """Best-effort size of a ``param_grid`` dict literal, for the prompt text."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = ast.unparse(node.func)
        if not fn.endswith(("Grid", "GridSearchCV", "RandomizedSearchCV", "param_grid")):
            continue
        # The grid may be positional (2nd arg after the estimator) or keyword.
        candidates = [*node.args, *(kw.value for kw in node.keywords if kw.arg == "param_grid")]
        for candidate in candidates:
            if not isinstance(candidate, ast.Dict):
                continue
            total = 1
            for value in candidate.values:
                if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
                    total *= max(len(value.elts), 1)
                elif isinstance(value, ast.Call):
                    total *= 3  # np.arange / np.linspace / similar
                else:
                    total = -1
                    break
            if total > 0:
                return total
    return None


# -- the public heuristic ----------------------------------------------
#: Prefixes that pin a path inside the session folder, so a relative literal
#: appended to one of them is known-safe without resolving the expression.
SAFE_ROOT_TOKENS = ("OUTPUT_DIR", "DATA_DIR", "SESSION_DIR", "FIGURES_DIR", "os.path.join", "Path(")


def find_destructive_issues(code: str, session_dir: Path, dataset_paths: list[Path]) -> list[str]:
    """Describe destructive-looking operations in ``code``."""
    issues: list[str] = []
    for pattern, label in DESTRUCTIVE_PATTERNS:
        if re.search(pattern, code):
            issues.append(f"it {label}")

    # Write-ish calls: flag only the ones whose target escapes the session folder
    # or lands on the uploaded dataset.
    for call in WRITE_CALLS:
        for m in re.finditer(rf"\.{call}\s*\(", code):
            issues.extend(_write_issues_in(_arg_window(code, m.end()), session_dir, dataset_paths))
    for m in re.finditer(r"\bopen\s*\(", code):
        window = _arg_window(code, m.end(), 200)
        if re.search(r"""["'][wa]|\.write_bytes|\.write\s*\(""", window):
            issues.extend(_write_issues_in(window, session_dir, dataset_paths))
    # de-duplicate while keeping order
    return list(dict.fromkeys(issues))


def _write_issues_in(window: str, session_dir: Path, dataset_paths: list[Path]) -> list[str]:
    """Check the string literals in one write call's argument text."""
    head = window.split(",", 1)[0]
    literals = _strings_in(head)
    if not literals:
        return []
    # An expression anchored on a known session folder (OUTPUT_DIR + '/x.csv',
    # os.path.join(DATA_DIR, 'x.csv'), ...) is inside the session by construction,
    # as long as the literal cannot climb out of it.
    anchored = any(token in head for token in SAFE_ROOT_TOKENS)
    out: list[str] = []
    for lit in literals:
        if anchored:
            issue = _suffix_issue(lit, session_dir, dataset_paths)
        else:
            issue = _write_issue(lit, session_dir, dataset_paths)
        if issue:
            out.append(issue)
    return out


def _suffix_issue(literal: str, session_dir: Path, dataset_paths: list[Path]) -> str | None:
    """Check a path literal that is appended to a known-safe session folder."""
    if not literal:
        return None
    if _matches_dataset(literal, dataset_paths):
        return f"it writes to the uploaded dataset path `{literal}` (your data may be overwritten)"
    # Joined onto a safe root, the literal behaves as a relative component.
    return _write_issue(literal.lstrip("/\\"), session_dir, dataset_paths)


def _write_issue(literal: str, session_dir: Path, dataset_paths: list[Path]) -> str | None:
    if not literal:
        return None
    if _matches_dataset(literal, dataset_paths):
        return f"it writes to the uploaded dataset path `{literal}` (your data may be overwritten)"
    if _outside_session(literal, session_dir):
        return (
            f"it writes to `{literal}`, which is outside the session folder {session_dir.name}/"
        )
    return None


def find_heavy_issues(code: str) -> list[str]:
    """Describe likely-slow compute patterns in ``code``."""
    return [label for pattern, label in HEAVY_PATTERNS if re.search(pattern, code)]


def needs_approval(
    call: ToolCall, state: ResearchState, *, session_dir: Path, threshold_seconds: int = 60
) -> ApprovalRequest | None:
    """Return an :class:`ApprovalRequest` when the user must confirm, else ``None``."""
    args = call.arguments or {}
    if call.name == "python":
        return _needs_approval_python(call, args, state, session_dir, threshold_seconds)
    if call.name == "write_file":
        return _needs_approval_write_file(call, args, session_dir)
    return None


def _needs_approval_python(
    call: ToolCall,
    args: dict[str, Any],
    state: ResearchState,
    session_dir: Path,
    threshold_seconds: int,
) -> ApprovalRequest | None:
    code = str(args.get("code") or "")
    if not code.strip():
        return None
    description = str(args.get("description") or "Run a Python cell")
    est_raw = args.get("est_seconds")
    est = int(est_raw) if isinstance(est_raw, (int, float)) or (
        isinstance(est_raw, str) and est_raw.strip().isdigit()
    ) else None

    dataset_paths: list[Path] = []
    if state.dataset and state.dataset.path:
        dataset_paths.append(Path(state.dataset.path))

    destructive = find_destructive_issues(code, session_dir, dataset_paths)
    if destructive:
        return ApprovalRequest(
            id=call.id,
            tool=call.name,
            title="Destructive or out-of-bounds action",
            reason=(
                f"**{description}** — this cell looks destructive: " + "; ".join(destructive) + ". "
                "The agent's code runs directly on this machine with no sandbox."
            ),
            code=code,
            details={"issues": destructive, "dataset_paths": [str(p) for p in dataset_paths]},
        )

    if est is not None and est > threshold_seconds:
        return ApprovalRequest(
            id=call.id,
            tool=call.name,
            title="Long-running cell",
            reason=(
                f"**{description}** — the agent estimated ~{est}s of runtime "
                f"(above the {threshold_seconds}s approval threshold). Approve?"
            ),
            code=code,
            details={"est_seconds": est, "threshold": threshold_seconds},
        )

    heavy = find_heavy_issues(code)
    if heavy:
        hint = (
            "Ask for this work with a realistic `est_seconds` value on the `python` tool "
            "next time so the estimate can be shown here."
            if est is None
            else f"Estimated runtime supplied by the agent: ~{est}s."
        )
        grid = _count_grid_combinations(code)
        size = f" The parameter grid looks like ~{grid} combinations." if grid else ""
        return ApprovalRequest(
            id=call.id,
            tool=call.name,
            title="Heavy compute",
            reason=(
                f"**{description}** — this cell looks expensive: "
                + "; ".join(heavy)
                + f".{size} {hint}"
            ),
            code=code,
            details={"issues": heavy, "grid_size": grid, "est_seconds": est},
        )
    return None


def _needs_approval_write_file(
    call: ToolCall, args: dict[str, Any], session_dir: Path
) -> ApprovalRequest | None:
    rel = str(args.get("path") or "")
    content = str(args.get("content") or "")
    if not rel:
        return None
    # Mirror files.write_file exactly: relative paths land in outputs/.
    raw = Path(rel.strip()).expanduser()
    candidate = raw if raw.is_absolute() else session_dir / "outputs" / raw
    target = Path(os.path.normpath(str(candidate)))
    if _outside_session(str(target), session_dir) or target == session_dir:
        return ApprovalRequest(
            id=call.id,
            tool=call.name,
            title="Write outside the session folder",
            reason=(
                f"This file write targets `{target}`, which is outside the session folder "
                f"{session_dir.name}/. Approve?"
            ),
            code=None,
            details={"path": str(target)},
        )
    if target.exists():
        return ApprovalRequest(
            id=call.id,
            tool=call.name,
            title="Overwrite an existing file",
            reason=(
                f"**`{rel}` already exists** ({target.stat().st_size} bytes) and would be "
                f"overwritten with {len(content)} characters. Approve?"
            ),
            code=None,
            details={"path": str(target), "existing_bytes": target.stat().st_size},
        )
    return None
