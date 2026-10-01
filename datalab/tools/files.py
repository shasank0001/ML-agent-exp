"""Session-scoped file tools: ``read_file``, ``write_file``, ``list_files``.

Every path is resolved inside ``runs/<session_id>/``. Anything that escapes the
session folder is an error — the approval gate covers the *overwrite* case, this
covers the *containment* case.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from .base import ToolContext, ToolResult

MAX_READ_CHARS = 40_000
MAX_WRITE_CHARS = 2_000_000

#: Where a bare relative name is looked for, most specific first. `""` is the
#: session root itself.
READ_BASES = ("outputs", "data", "figures", "")
WRITE_BASES = ("outputs",)
LIST_BASES = ("",)


def resolve_in_session(
    ctx: ToolContext, raw: str, *, bases: tuple[str, ...] = READ_BASES
) -> Path:
    """Resolve ``raw`` inside the session root, refusing to escape it.

    A relative name is looked up under each folder in ``bases`` in order and the
    first existing match wins; when nothing exists, ``bases[0]`` is used so the
    caller can create it. Absolute paths are accepted only when they already
    live under the session root (that is how uploaded data files are addressed).
    """
    root = ctx.root.resolve()
    candidate = Path(raw.strip()).expanduser()
    if not candidate.is_absolute():
        for base in bases:
            probe = Path(os.path.normpath(str((ctx.paths[base] if base else ctx.root) / candidate)))
            if probe.exists():
                candidate = probe
                break
        else:
            first = ctx.paths[bases[0]] if bases[0] else ctx.root
            candidate = first / candidate
    resolved = Path(os.path.normpath(str(candidate)))
    try:
        resolved.resolve().relative_to(root)
    except ValueError:
        raise PermissionError(
            f"'{raw}' resolves outside the session folder ({root.name}/). "
            "Only paths inside the session directory are allowed."
        ) from None
    return resolved.resolve()


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _display_path(ctx: ToolContext, path: Path) -> str:
    """Session-relative display path (never leak absolute server paths)."""
    try:
        return str(path.resolve().relative_to(ctx.root.resolve()))
    except ValueError:
        return path.name


#: Files the agent must never overwrite via write_file (ground truth).
_PROTECTED_NAMES = {"state.json", "events.jsonl"}


def _is_protected(ctx: ToolContext, path: Path) -> bool:
    try:
        resolved = path.resolve()
        root = ctx.root.resolve()
        rel = resolved.relative_to(root)
    except ValueError:
        return True
    if resolved == (root / "state.json") or resolved == (root / "events.jsonl"):
        return True
    if path.name in _PROTECTED_NAMES or resolved.suffix == ".tmp":
        return True
    try:
        if resolved.is_relative_to((root / "data").resolve()):
            return True
    except (OSError, ValueError):
        pass
    if len(rel.parts) > 1 and rel.parts[0] == "data":
        return True
    return False


# -- read_file ----------------------------------------------------------
READ_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "File path inside the session folder."},
        "max_chars": {
            "type": "integer",
            "description": "Maximum characters to return (default 20000).",
        },
    },
    "required": ["path"],
}
READ_DESCRIPTION = "Read a text file from the session folder (outputs/, data/, or any file the agent saved)."


async def read_file(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    try:
        path = resolve_in_session(ctx, str(args["path"]))
    except KeyError:
        return ToolResult.fail("read_file needs a 'path' argument.")
    except PermissionError as exc:
        return ToolResult.fail(str(exc))
    except Exception as exc:  # noqa: BLE001 - bad path type must not raise
        return ToolResult.fail(f"read_file needs a valid 'path' string: {exc}")
    display = _display_path(ctx, path)
    if not path.exists():
        return ToolResult.fail(f"No such file: {display}")
    if path.is_dir():
        return ToolResult.fail(f"{display} is a directory. Use list_files instead.")
    try:
        limit = int(args.get("max_chars") or 20_000)
    except (TypeError, ValueError):
        limit = 20_000
    limit = max(1, min(limit, MAX_READ_CHARS))
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        size = path.stat().st_size
    except OSError as exc:
        return ToolResult.fail(f"Could not read {path.name}: {exc}")
    truncated = len(text) > limit
    body = text[:limit]
    suffix = f"\n\n[truncated: showing {limit} of {len(text)} characters]" if truncated else ""
    return ToolResult.ok(
        f"{display} ({_human_size(size)})\n\n{body}{suffix}",
        data={"path": display, "chars": len(text), "truncated": truncated},
    )


# -- write_file ---------------------------------------------------------
WRITE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path relative to the session outputs/ folder, e.g. 'report.md'.",
        },
        "content": {"type": "string", "description": "Full file content."},
    },
    "required": ["path", "content"],
}
WRITE_DESCRIPTION = (
    "Write a text file into the session outputs/ folder (reports, notes, scripts). "
    "Overwriting an existing file asks the user for approval first."
)


async def write_file(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path_raw = str(args.get("path") or "").strip()
    content = args.get("content")
    if not path_raw:
        return ToolResult.fail("write_file needs a 'path' argument.")
    if not isinstance(content, str):
        return ToolResult.fail("write_file needs a 'content' string argument.")
    if len(content) > MAX_WRITE_CHARS:
        return ToolResult.fail(
            f"Content is {len(content):,} characters, over the {MAX_WRITE_CHARS:,} limit. "
            "Write the file in pieces or trim it."
        )
    try:
        # Default target is outputs/ so the agent's deliverables stay together.
        path = resolve_in_session(ctx, path_raw, bases=WRITE_BASES)
    except PermissionError as exc:
        return ToolResult.fail(str(exc))
    if _is_protected(ctx, path):
        return ToolResult.fail(
            f"Refusing to write {_display_path(ctx, path)}: state.json, events.jsonl and "
            "data/ are harness-owned. Write deliverables under outputs/ instead."
        )
    if path.is_dir():
        return ToolResult.fail(f"{_display_path(ctx, path)} is a directory; give a file name instead.")
    existed = path.exists()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        return ToolResult.fail(f"Could not write {path.name}: {exc}")
    return ToolResult.ok(
        f"{'Overwrote' if existed else 'Wrote'} {_display_path(ctx, path)} ({_human_size(len(content.encode('utf-8')))}).",
        data={"path": _display_path(ctx, path), "bytes": len(content.encode("utf-8")), "existed": existed},
    )


# -- list_files ---------------------------------------------------------
LIST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "dir": {
            "type": "string",
            "description": "Folder inside the session, e.g. 'data', 'outputs', 'figures'. Default: the session root.",
        },
        "recursive": {"type": "boolean", "description": "Walk sub-folders (default true)."},
    },
}
LIST_DESCRIPTION = "List files under a folder inside the session (data/, outputs/, figures/)."


async def list_files(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    raw = str(args.get("dir") or "").strip() or "."
    try:
        base = resolve_in_session(ctx, raw, bases=LIST_BASES)
    except PermissionError as exc:
        return ToolResult.fail(str(exc))
    display_base = _display_path(ctx, base)
    if not base.exists():
        return ToolResult.fail(f"No such folder: {display_base}")
    if not base.is_dir():
        return ToolResult.fail(f"{display_base} is a file, not a folder. Use read_file to see its contents.")
    recursive = bool(args.get("recursive", True))
    entries: list[str] = []
    truncated_walk = False
    for i, path in enumerate(sorted(base.rglob("*") if recursive else base.glob("*"))):
        if i >= 2000:
            truncated_walk = True
            break
        if path.is_symlink() or not path.is_file():
            continue
        try:
            rel = path.relative_to(ctx.root)
            size = path.stat().st_size
        except (OSError, ValueError):
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}:
            entries.append(f"  {rel}  ({_human_size(size)}, image)")
        else:
            entries.append(f"  {rel}  ({_human_size(size)})")
        if len(entries) >= 2000:
            truncated_walk = True
            break
    if not entries:
        return ToolResult.ok(f"{display_base} is empty.")
    shown = entries[:200]
    suffix = (
        f"\n... showing 200 of {len(entries)} files" + (" (walk capped)" if truncated_walk else "")
        if len(entries) > 200 or truncated_walk else ""
    )
    header = f"{len(entries)} file(s) under {display_base}/:"
    return ToolResult.ok(header + "\n" + "\n".join(shown) + suffix, data={"dir": display_base, "count": len(entries)})


def copy_upload(src: Path, ctx: ToolContext) -> Path:
    """Copy an uploaded file into the session's ``data/`` folder."""
    ctx.data_dir.mkdir(parents=True, exist_ok=True)
    dest = ctx.data_dir / Path(src).name
    counter = 1
    while dest.exists() and dest.resolve() != Path(src).resolve():
        dest = ctx.data_dir / f"{Path(src).stem}_{counter}{Path(src).suffix}"
        counter += 1
    shutil.copy2(src, dest)
    return dest
