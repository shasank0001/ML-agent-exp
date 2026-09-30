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
    return resolved


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


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
    if not path.exists():
        return ToolResult.fail(f"No such file: {path}")
    if path.is_dir():
        return ToolResult.fail(f"{path} is a directory. Use list_files instead.")
    limit = int(args.get("max_chars") or 20_000)
    limit = max(1, min(limit, MAX_READ_CHARS))
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return ToolResult.fail(f"Could not read {path.name}: {exc}")
    truncated = len(text) > limit
    body = text[:limit]
    suffix = f"\n\n[truncated: showing {limit} of {len(text)} characters]" if truncated else ""
    return ToolResult.ok(
        f"{path} ({_human_size(path.stat().st_size)})\n\n{body}{suffix}",
        data={"path": str(path), "chars": len(text), "truncated": truncated},
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
    if path.is_dir():
        return ToolResult.fail(f"{path} is a directory; give a file name instead.")
    existed = path.exists()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        return ToolResult.fail(f"Could not write {path.name}: {exc}")
    return ToolResult.ok(
        f"{'Overwrote' if existed else 'Wrote'} {path} ({_human_size(len(content.encode('utf-8')))}).",
        data={"path": str(path), "bytes": len(content.encode("utf-8")), "existed": existed},
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
    if not base.exists():
        return ToolResult.fail(f"No such folder: {base}")
    if not base.is_dir():
        return ToolResult.fail(f"{base} is a file, not a folder. Use read_file to see its contents.")
    recursive = bool(args.get("recursive", True))
    entries: list[str] = []
    for path in sorted(base.rglob("*") if recursive else base.glob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ctx.root)
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}:
            entries.append(f"  {rel}  ({_human_size(path.stat().st_size)}, image)")
        else:
            entries.append(f"  {rel}  ({_human_size(path.stat().st_size)})")
    if not entries:
        return ToolResult.ok(f"{base} is empty.")
    header = f"{len(entries)} file(s) under {base.relative_to(ctx.root) if base != ctx.root else ctx.root.name}/:"
    return ToolResult.ok(header + "\n" + "\n".join(entries[:200]), data={"dir": str(base), "count": len(entries)})


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
