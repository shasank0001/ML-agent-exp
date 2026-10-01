"""Deterministic pandas profiler: the `profile_dataset` tool.

Numbers are computed in plain Python (never LLM-written). The result fills
in ``state.dataset``; a compact text report goes back to the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..state import DatasetInfo, TaskType
from .base import ToolContext, ToolResult

DESCRIPTION = (
    "Profile a tabular dataset (CSV/TSV/Parquet/Excel) and record deterministic "
    "shape, column and target-candidate statistics in session state. "
    "Give the file path (absolute, relative to the data dir, or bare filename)."
)
SCHEMA: dict[str, Any] = {"type": "object", "properties": {
    "path": {"type": "string", "description": "Dataset file to profile."}},
    "required": ["path"], "additionalProperties": False}
_READERS = frozenset({".csv", ".tsv", ".parquet", ".xlsx", ".xls"})
_TARGET_HINTS = ("target", "label", "outcome", "class", "category", "response",
                 "sales", "price", "revenue", "churn", "survived")
_ID_LIKE = frozenset({"id", "index", "uuid", "key", "timestamp", "date", "time", "created_at"})
_GENERIC = frozenset({"unnamed", "column", "col", "feature", "value",
                      "variable", "var", "field", "data", "x"})


def _py(value: Any) -> Any:  # numpy/pandas scalars -> JSON-safe (NaN -> None)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if f != f else f
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _is_numeric(s: pd.Series) -> bool:
    return not pd.api.types.is_bool_dtype(s.dtype) and bool(pd.api.types.is_numeric_dtype(s.dtype))


def _is_id_like(name: str) -> bool:
    return (low := name.strip().lower()) in _ID_LIKE or low.endswith("_id")


def _column_summary(name: str, s: pd.Series, n_rows: int) -> dict[str, Any]:
    n_missing = int(s.isna().sum())
    try:
        n_unique = int(s.nunique(dropna=True))
    except Exception:  # noqa: BLE001 - exotic dtypes must not break profiling
        n_unique = 0
    col: dict[str, Any] = {"name": name, "dtype": str(s.dtype), "n_missing": n_missing,
                           "n_unique": n_unique,
                           "missing_pct": round(n_missing / n_rows * 100, 1) if n_rows else 0.0}
    if _is_numeric(s):
        col["numeric"] = True
        try:
            desc = s.describe()
        except Exception:  # noqa: BLE001
            desc = pd.Series(dtype=float)
        for key, stat in (("mean", "mean"), ("std", "std"), ("min", "min"),
                          ("q25", "25%"), ("median", "50%"), ("q75", "75%"), ("max", "max")):
            col[key] = _py(desc.get(stat))
        try:
            col["n_zeros"] = int((s == 0).sum())
        except Exception:  # noqa: BLE001
            col["n_zeros"] = 0
    else:
        col["numeric"] = False
        try:
            vc = s.value_counts(dropna=True).head(10)
            col["top_categories"] = [{"value": str(v)[:60], "count": int(c)}
                                     for v, c in vc.items()]
        except Exception:  # noqa: BLE001
            col["top_categories"] = []
    return col


def _target_candidates(summaries: list[dict[str, Any]], n_rows: int) -> list[str]:
    """Rank columns that look like a supervised target, best first (max 8)."""
    scored: list[tuple[float, int, str]] = []
    for col in summaries:
        name = str(col.get("name", ""))
        low = name.strip().lower()
        n_unique, n_missing = int(col.get("n_unique", 0)), int(col.get("n_missing", 0))
        if _is_id_like(low) or n_unique < 2:
            continue
        if n_rows - n_missing > 0 and n_unique / (n_rows - n_missing) > 0.5:
            continue
        score = 0.0
        if low == "y" or low.startswith("is_") or any(h in low for h in _TARGET_HINTS):
            score += 3.0
        if not col.get("numeric", False):
            score += 2.0
        elif 2 <= n_unique <= 20 and low not in _GENERIC and not low.startswith("unnamed"):
            score += 2.0
        if score > 0:
            scored.append((score, n_unique, name))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    return list(dict.fromkeys(n for _, _, n in scored))[:8]


def _infer_task_type(candidates: list[str], by_name: dict[str, dict[str, Any]]) -> TaskType | None:
    """Same rules as ``datalab.lab.infer_task_type`` (local copy: no lab import)."""
    if not candidates:
        return None
    best = by_name.get(candidates[0], {})
    if not best.get("numeric", False) or 2 <= int(best.get("n_unique", 0)) <= 20:
        return "classification"
    return "regression"


def _looks_like_datetime(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return False
    try:
        sample = s.dropna().astype(str).head(20)
        if len(sample) < 5:
            return False
        return bool(pd.to_datetime(sample, errors="coerce", format="mixed").notna().mean() >= 0.8)
    except Exception:  # noqa: BLE001
        return False


def profile_frame(df: pd.DataFrame, path: str) -> DatasetInfo:
    """Profile an in-memory frame. Pure and deterministic (no sampling)."""
    n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
    try:
        duplicates = int(df.duplicated().sum())
    except Exception:  # noqa: BLE001
        duplicates = 0
    summaries: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    for i in range(n_cols):
        try:
            name = str(df.columns[i])
        except Exception:  # noqa: BLE001
            name = f"column_{i}"
        positions.setdefault(name, i)
        try:
            summaries.append(_column_summary(name, df.iloc[:, i], n_rows))
        except Exception:  # noqa: BLE001 - one bad column must not kill the run
            summaries.append({"name": name, "dtype": "unknown", "n_missing": 0,
                              "n_unique": 0, "missing_pct": 0.0, "numeric": False,
                              "top_categories": []})
    by_name = {c["name"]: c for c in summaries}
    candidates = _target_candidates(summaries, n_rows)
    task_type = _infer_task_type(candidates, by_name)
    missing = [f"column '{c['name']}' has {c['missing_pct']:.1f}% missing values"
               for c in sorted(summaries, key=lambda c: float(c.get("missing_pct", 0.0)),
                               reverse=True) if int(c.get("n_missing", 0)) > 0]
    ids, id_noted = [], set()
    for c in summaries:
        name, u = str(c["name"]), int(c.get("n_unique", 0))
        near_unique = n_rows > 0 and u > 10 and u / n_rows > 0.9
        if u >= 2 and (_is_id_like(name) or (near_unique and not c.get("numeric", False))) \
                and not _looks_like_datetime(df.iloc[:, positions[name]]):
            ids.append(f"column '{name}' is a near-unique identifier ({u}/{n_rows} unique)"
                       " and is probably not a feature")
            id_noted.add(name)
    imb: list[str] = []
    if task_type == "classification" and candidates:
        best, u = candidates[0], int(by_name[candidates[0]].get("n_unique", 0))
        if 2 <= u <= 10:
            try:
                vc = df.iloc[:, positions[best]].value_counts(dropna=False).head(4)
                parts = ", ".join(f"{str(v)[:20]}: {int(n)}" for v, n in vc.items())
                counts = [int(n) for n in vc.values if int(n) > 0]
                word = "is imbalanced" if len(counts) >= 2 and max(counts) / min(counts) >= 1.5 \
                    else "distribution"
                imb.append(f"target column '{best}' {word} ({parts})")
            except Exception:  # noqa: BLE001
                pass
    other = ([f"{duplicates} duplicate row{'s' if duplicates != 1 else ''}"]
             if duplicates else [])
    for c in summaries:
        name = str(c["name"])
        if name in id_noted or c.get("numeric", False):
            continue
        if _looks_like_datetime(df.iloc[:, positions[name]]):
            other.append(f"column '{name}' looks like a datetime but is stored as text")
        elif int(c.get("n_unique", 0)) > 50:
            other.append(f"column '{name}' is high-cardinality ({c['n_unique']} unique)")
    notes = [(n[:140]) for n in (missing + ids + imb + other)[:12]]
    return DatasetInfo(path=path, n_rows=n_rows, n_cols=n_cols, columns=summaries,
                       profile_notes=notes, duplicates=duplicates,
                       target_candidates=candidates, suggested_task_type=task_type)


def profile_dataset(path: Path | str) -> DatasetInfo:
    """Read a file by extension and profile it (ValueError on bad ext/unreadable)."""
    p = Path(path).expanduser()
    ext = p.suffix.lower()
    if ext not in _READERS:
        raise ValueError(f"Unsupported extension '{p.suffix}' for '{p}'. "
                         "Supported: .csv, .tsv, .parquet, .xlsx, .xls.")
    try:
        if ext == ".csv":
            df = pd.read_csv(p, low_memory=False)
        elif ext == ".tsv":
            df = pd.read_csv(p, sep="\t", low_memory=False)
        elif ext == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_excel(p)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not read '{p}': {exc}") from exc
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return profile_frame(df, str(p))


def format_report(info: DatasetInfo, *, max_categories: int = 6) -> str:
    """Compact model-friendly text block (kept under ~4000 chars)."""
    lines = [f"Dataset: {info.path}", f"Shape: {info.n_rows} rows x {info.n_cols} cols",
             f"Duplicates: {info.duplicates}", "Columns (name | dtype | missing | unique):"]
    for c in info.columns[:40]:
        lines.append(f"  - {c.get('name')} | {c.get('dtype')} | "
                     f"{c.get('n_missing', 0)} missing | {c.get('n_unique', 0)} unique")
    if len(info.columns) > 40:
        lines.append(f"  ... and {len(info.columns) - 40} more columns")
    if info.target_candidates:
        lines.append(f"Target candidates: {', '.join(info.target_candidates)}")
        best = next((c for c in info.columns if c.get("name") == info.target_candidates[0]), {})
        cats = best.get("top_categories", [])[:max_categories]
        if cats:
            lines.append("Top values of '{}': {}".format(
                info.target_candidates[0], ", ".join(f"{t['value']}: {t['count']}" for t in cats)))
    else:
        lines.append("Target candidates: none found")
    lines.append(f"Suggested task type: {info.suggested_task_type or 'unknown'}")
    if info.profile_notes:
        lines.append("Notes:")
        lines.extend(f"  - {n}" for n in info.profile_notes)
    text = "\n".join(lines)
    return text if len(text) <= 4000 else text[:3940] + "\n... (report truncated)"


def _resolve_in_session(raw: str, ctx) -> Path:
    """Session-contained resolver: absolute paths must live under the session root."""
    from .files import resolve_in_session

    path = resolve_in_session(ctx, raw, bases=("data", "outputs", "figures", ""))
    if not path.exists() or not path.is_file():
        try:
            present = sorted(p.name for p in ctx.data_dir.iterdir() if p.is_file())[:20]
        except OSError:
            present = []
        have = f"Files in data dir: {', '.join(present)}" if present else "Data dir is empty"
        raise FileNotFoundError(f"File '{raw}' not found. {have}.")
    return path


async def handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    try:
        raw = str((args or {}).get("path", "")).strip()
        if not raw:
            raise ValueError("Missing required argument 'path'.")
        info = profile_dataset(_resolve_in_session(raw, ctx))
        ctx.state.dataset = info
        return ToolResult.ok(format_report(info), data={"profile": info.model_dump()})
    except Exception as exc:  # noqa: BLE001 - handler never raises
        return ToolResult.fail(f"profile_dataset failed: {exc}")
