"""Code-shape heuristics behind the approval gate.

Everything here is a heuristic over the *text* of a cell: regexes for the
obvious shapes, plus a light :mod:`ast` pass that is immune to string-splitting
tricks such as f-strings, ``getattr(os, "remove")`` and computed paths.

This is a speed bump for a single-user local demo, not a security boundary —
see NOTES.md. :mod:`datalab.approvals` owns the policy; this module only decides
what a given piece of code looks like.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

#: An estimator with at least this many trees is a multi-minute cell.
N_ESTIMATORS_THRESHOLD = 1000

#: Prefixes that pin a path inside the session folder, so a literal appended to
#: one of them cannot escape it even when it looks absolute.
SAFE_ROOT_TOKENS = (
    "OUTPUT_DIR",
    "DATA_DIR",
    "SESSION_DIR",
    "FIGURES_DIR",
    "os.path.join",
    "Path(",
)

# -- regex heuristics ---------------------------------------------------
DESTRUCTIVE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bos\s*\.\s*remove\s*\(", "calls os.remove()"),
    (r"\bos\s*\.\s*unlink\s*\(", "calls os.unlink()"),
    (r"\bshutil\s*\.\s*rmtree\s*\(", "calls shutil.rmtree()"),
    (r"\bos\s*\.\s*rmdir\s*\(", "calls os.rmdir()"),
    (r"\bos\s*\.\s*removedirs\s*\(", "calls os.removedirs()"),
    (r"\bos\s*\.\s*rename\s*\(", "calls os.rename()"),
    (r"\bos\s*\.\s*replace\s*\(", "calls os.replace()"),
    (r"\bos\s*\.\s*renames\s*\(", "calls os.renames()"),
    (r"\bos\s*\.\s*mkdir\s*\(", "calls os.mkdir()"),
    (r"\bos\s*\.\s*makedirs\s*\(", "calls os.makedirs()"),
    (r"\bos\s*\.\s*chmod\s*\(", "calls os.chmod()"),
    (r"\bos\s*\.\s*chown\s*\(", "calls os.chown()"),
    (r"\bos\s*\.\s*truncate\s*\(", "calls os.truncate()"),
    (r"\bos\s*\.\s*_exit\s*\(", "calls os._exit(), which would kill the app process"),
    (r"\bos\s*\.\s*system\s*\(", "shells out via os.system()"),
    (r"\bos\s*\.\s*popen\s*\(", "shells out via os.popen()"),
    (r"\bsubprocess\s*\.", "shells out via subprocess"),
    (r"\b__import__\s*\(", "reaches a module through __import__()"),
    (r"\.unlink\s*\(", "calls .unlink() on a file"),
    (r"\.rmdir\s*\(", "calls .rmdir() on a directory"),
    (r"(?<![\w.])rmtree\s*\(", "calls rmtree()"),
    (r"\bpathlib\s*\.\s*Path\s*\([^)]*\)\s*\.\s*unlink\b", "unlinks a path"),
)

#: Calls whose *first* argument names the file being written.
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

#: Calls that move an existing file; the destination is the second argument.
COPY_CALLS: tuple[str, ...] = ("copy", "copy2", "copyfile", "copytree", "move")

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

#: Attribute access that is destructive however it is spelled.
_METHOD_EVIDENCE = (".unlink(", ".rmdir(")

_STRING_LITERAL = re.compile(r"""["']([^"'\n]{1,300})["']""")
_DATA_SUFFIXES = (".csv", ".tsv", ".parquet", ".xlsx", ".xls", ".feather", ".h5", ".pkl", ".json")

#: Module names that make ``getattr`` worth a second look.
_SENSITIVE_MODULES = frozenset(
    {"os", "shutil", "subprocess", "sys", "builtins", "__builtins__", "pathlib", "io", "sysconfig"}
)

#: Dotted call names that are destructive no matter what they are called on.
_DESTRUCTIVE_CALLS = frozenset(
    {
        "os.remove", "os.unlink", "os.rmdir", "os.removedirs", "os.rename", "os.replace",
        "os.renames", "os.mkdir", "os.makedirs", "os.chmod", "os.chown", "os.truncate",
        "os._exit", "os.system", "os.popen", "os.removeprefix",
        "shutil.rmtree", "shutil.move",
        "subprocess.run", "subprocess.call", "subprocess.Popen", "subprocess.check_output",
        "subprocess.check_call", "subprocess.getoutput", "subprocess.getstatusoutput",
        "__import__",
    }
)

#: Short fragments that make a string literal worth re-reading.
_DESTRUCTIVE_FRAGMENTS = (
    "os.remove", "os.unlink", "os.rmdir", "os.rename", "os.replace", "os.system",
    "os.mkdir", "os.makedirs", "os.chmod", "os._exit", "rm -rf", "shutil.rmtree",
    "subprocess.", "unlink()",
)


# -- text helpers -------------------------------------------------------
def strings_in(text: str) -> list[str]:
    """Every double- or single-quoted run in ``text``."""
    return _STRING_LITERAL.findall(text)


def looks_like_path(value: str) -> bool:
    """True when ``value`` is shaped like a filesystem path rather than prose."""
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


def outside_session(value: str, session_dir: Path) -> bool:
    """True when a path literal escapes the session directory."""
    if not value:
        return False
    norm = value.replace("\\", "/")
    if not looks_like_path(norm):
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


def matches_dataset(value: str, dataset_paths: list[Path]) -> bool:
    """True when a literal names one of the uploaded files, however it is spelled."""
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


def count_grid_combinations(code: str) -> int | None:
    """Best-effort size of a ``param_grid`` dict literal, for the prompt text."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = _dotted_name(node.func)
        if not fn.endswith(("Grid", "GridSearchCV", "RandomizedSearchCV", "param_grid")):
            continue
        # The grid may be positional (2nd arg, after the estimator) or keyword.
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


# -- ast helpers --------------------------------------------------------
def _dotted_name(func: ast.expr) -> str:
    """``os.path.join`` for ``os.path.join(...)``, ``''`` for anything computed."""
    parts: list[str] = []
    node: ast.AST = func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _dotted_args(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _strings_in_node(node: ast.AST) -> list[str]:
    """Every string constant reachable from ``node``, including f-string parts."""
    out: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            out.append(child.value)
    return out


def _string_looks_destructive(text: str) -> bool:
    return any(fragment in text for fragment in _DESTRUCTIVE_FRAGMENTS)


def _int_value(node: ast.AST) -> int | float | None:
    """Best-effort numeric value of a constant or simple arithmetic expression."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Call) and _dotted_name(node.func) in {"int", "float"}:
        return _int_value(node.args[0]) if node.args else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        inner = _int_value(node.operand)
        if isinstance(inner, (int, float)):
            return -inner if isinstance(node.op, ast.USub) else inner
        return None
    if isinstance(node, ast.BinOp):
        left, right = _int_value(node.left), _int_value(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)) and right <= 8:
            try:
                return left**right
            except OverflowError:
                return None
        if isinstance(node.op, ast.Mult):
            return left * right
    return None


def _parse(code: str) -> ast.AST | None:
    try:
        return ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return None


# -- ast analysis -------------------------------------------------------
def _ast_destructive_issues(tree: ast.AST) -> list[str]:
    """Destructive calls, including the obfuscated spellings regexes miss."""
    issues: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = _dotted_name(node.func)
        if fn in _DESTRUCTIVE_CALLS:
            issues.append(f"it calls {fn}()")
        elif fn == "getattr" and _dotted_args(node) & _SENSITIVE_MODULES:
            issues.append("it reaches a destructive helper through getattr()")
        elif fn == "getattr" and any(
            isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value in
            {"remove", "unlink", "rmdir", "system", "_exit", "popen", "rmtree", "run", "Popen"}
            for a in node.args
        ):
            issues.append("it reaches a destructive helper through getattr()")
        elif fn in {"eval", "exec"} and any(
            _string_looks_destructive(s) for s in _strings_in_node(node)
        ):
            issues.append(f"it calls {fn}() on a string containing a destructive call")
    return issues


def _ast_traversal_issues(tree: ast.AST) -> list[str]:
    """``Path(x) / ".." / ".." / "y"`` climbs out of the session folder."""
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if any(".." in s for s in _strings_in_node(node)):
                issues = ["it walks a path out of the session folder with '..'"]
                return issues
    return []


def _tail_after(node: ast.AST, lines: list[str]) -> str:
    """The source text immediately following ``node`` on its last line."""
    line_no, col = getattr(node, "end_lineno", None), getattr(node, "end_col_offset", None)
    if not line_no or col is None or line_no > len(lines):
        return ""
    return lines[line_no - 1][col : col + 40]


#: Methods chained onto an ``open()`` handle that make its intent clear.
_HANDLE_METHODS = ("read", "readlines", "read_text", "read_bytes", "write", "write_text")


def _ast_write_issues(
    tree: ast.AST, session_dir: Path, dataset_paths: list[Path], lines: list[str]
) -> list[str]:
    """Write/copy targets that escape the session or hit the uploaded data."""
    issues: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = _dotted_name(node.func)
        short = fn.rsplit(".", 1)[-1]

        if short in WRITE_CALLS and node.args:
            issues += _target_issues(node.args[0], session_dir, dataset_paths)
        elif short in COPY_CALLS and len(node.args) >= 2:
            issues += _target_issues(node.args[1], session_dir, dataset_paths)
        elif short == "open":
            window = _strings_in_node(node)
            tail = _tail_after(node, lines)
            chained = any(f".{m}(" in tail for m in _HANDLE_METHODS)
            if not window and not chained:
                issues.append("it calls open() without a visible mode, so it may write")
            elif any(m in {"w", "a", "x", "wb", "ab", "+"} for m in window):
                mode_arg = node.args[1] if len(node.args) > 1 else next(
                    (kw.value for kw in node.keywords if kw.arg == "mode"), None
                )
                if mode_arg is None:
                    issues.append("it calls open() without a visible mode, so it may write")
                else:
                    for lit in _strings_in_node(mode_arg):
                        if _looks_like_write_mode(lit):
                            issues += _target_issues(
                                node.args[0] if node.args else None, session_dir, dataset_paths
                            )
    return issues


def _looks_like_write_mode(literal: str) -> bool:
    mode = literal.strip().lower()
    return any(ch in mode for ch in "wax+") and not mode.startswith("r")


def _target_issues(
    node: ast.AST | None, session_dir: Path, dataset_paths: list[Path]
) -> list[str]:
    """Check the path literals inside one target expression."""
    if node is None:
        return []
    literals = _strings_in_node(node)
    if not literals:
        return []
    source = ast.unparse(node)
    anchored = any(token in source for token in SAFE_ROOT_TOKENS)
    out: list[str] = []
    for lit in literals:
        issue = (
            _suffix_issue(lit, session_dir, dataset_paths)
            if anchored
            else _write_issue(lit, session_dir, dataset_paths)
        )
        if issue:
            out.append(issue)
    return out


def _ast_heavy_issues(tree: ast.AST) -> list[str]:
    """Slow-looking estimator sizes, including computed ones."""
    issues: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "n_estimators":
                    value = _int_value(kw.value)
                    if value is not None and value >= N_ESTIMATORS_THRESHOLD:
                        issues.append("n_estimators in the thousands")
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "n_estimators":
                    number = _int_value(value)
                    if number is not None and number >= N_ESTIMATORS_THRESHOLD:
                        issues.append("n_estimators in the thousands")
    return issues


# -- public analysis ----------------------------------------------------
def find_destructive_issues(code: str, session_dir: Path, dataset_paths: list[Path]) -> list[str]:
    """Describe destructive-looking operations in ``code``."""
    issues = [label for pattern, label in DESTRUCTIVE_PATTERNS if re.search(pattern, code)]

    tree = _parse(code)
    if tree is not None:
        issues += _ast_destructive_issues(tree)
        issues += _ast_traversal_issues(tree)
        issues += _ast_write_issues(tree, session_dir, dataset_paths, code.split("\n"))

    # Text scan, as a backstop for anything the parse could not reach.
    for call in (*WRITE_CALLS, *COPY_CALLS):
        for m in re.finditer(rf"\.{call}\s*\(", code):
            window = _arg_window(code, m.end())
            issues += _write_issues_in(window, session_dir, dataset_paths, skip_first=call in COPY_CALLS)
    for m in re.finditer(r"\bopen\s*\(", code):
        window = _arg_window(code, m.end(), 200)
        if re.search(r"""["'][wax][bt+]?\+?["']|\.write_bytes|\.write\s*\(""", window):
            issues += _write_issues_in(window, session_dir, dataset_paths)

    return list(dict.fromkeys(issues))


def _write_issues_in(
    window: str, session_dir: Path, dataset_paths: list[Path], *, skip_first: bool = False
) -> list[str]:
    """Check the string literals in one write call's argument text."""
    head = window.split(",", 1)[0]
    literals = strings_in(head)
    if not literals:
        return []
    # An expression anchored on a known session folder (OUTPUT_DIR + '/x.csv',
    # os.path.join(DATA_DIR, 'x.csv'), ...) is inside the session by construction,
    # as long as the literal cannot climb out of it.
    anchored = any(token in head for token in SAFE_ROOT_TOKENS)
    issues: list[str] = []
    for lit in literals:
        issue = (
            _suffix_issue(lit, session_dir, dataset_paths)
            if anchored
            else _write_issue(lit, session_dir, dataset_paths)
        )
        if issue:
            issues.append(issue)
    return issues


def _suffix_issue(literal: str, session_dir: Path, dataset_paths: list[Path]) -> str | None:
    """Check a path literal that is appended to a known-safe session folder."""
    if not literal:
        return None
    if matches_dataset(literal, dataset_paths):
        return f"it writes to the uploaded dataset path `{literal}` (your data may be overwritten)"
    # Joined onto a safe root, the literal behaves as a relative component.
    return _write_issue(literal.lstrip("/\\"), session_dir, dataset_paths)


def _write_issue(literal: str, session_dir: Path, dataset_paths: list[Path]) -> str | None:
    if not literal:
        return None
    if matches_dataset(literal, dataset_paths):
        return f"it writes to the uploaded dataset path `{literal}` (your data may be overwritten)"
    if outside_session(literal, session_dir):
        return f"it writes to `{literal}`, which is outside the session folder {session_dir.name}/"
    return None


def find_heavy_issues(code: str) -> list[str]:
    """Describe likely-slow compute patterns in ``code``."""
    issues = [label for pattern, label in HEAVY_PATTERNS if re.search(pattern, code)]
    tree = _parse(code)
    if tree is not None:
        issues += _ast_heavy_issues(tree)
    return list(dict.fromkeys(issues))
