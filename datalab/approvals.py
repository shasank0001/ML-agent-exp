"""Rule-based approval gate for costly or destructive actions.

Local demo, no sandbox: anything the heuristics flag is shown to the user as an
Approve / Deny question before it runs. The shape analysis lives in
:mod:`datalab.approval_heuristics`; this module owns the policy — which rule
fires, what the dialog says, and what happens on each answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .approval_heuristics import (
    count_grid_combinations,
    find_destructive_issues,
    find_heavy_issues,
    outside_session,
)
from .llm import ToolCall
from .state import ResearchState


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


def needs_approval(
    call: ToolCall,
    state: ResearchState,
    *,
    session_dir: Path,
    threshold_seconds: int = 180,
    python_soft_timeout_s: int | None = None,
) -> ApprovalRequest | None:
    """Return an :class:`ApprovalRequest` when the user must confirm, else ``None``.

    Total by construction: a cell that will not parse, or an argument shape this
    module has never seen, returns ``None`` rather than raising. Blocking the
    agent on a broken heuristic would be worse than missing a case.
    """
    args = call.arguments or {}
    try:
        if call.name == "python":
            return _needs_approval_python(
                call, args, state, session_dir, threshold_seconds,
                python_soft_timeout_s=python_soft_timeout_s,
            )
        if call.name == "write_file":
            return _needs_approval_write_file(call, args, session_dir)
    except Exception:  # noqa: BLE001 - the gate must never break a run
        return None
    return None


def coerce_est_seconds(raw: Any) -> int | None:
    """Lenient runtime-estimate parser (accepts 90, 90.5, '90s', '5m', '1.5h')."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        value = int(raw)
        return value if 0 <= value <= 7200 else None
    if isinstance(raw, str):
        import re

        match = re.fullmatch(
            r"\s*(\d+(?:\.\d+)?)\s*(s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours)?\s*",
            raw,
            re.IGNORECASE,
        )
        if not match:
            return None
        amount, unit = float(match.group(1)), (match.group(2) or "s").lower()
        multiplier = 3600 if unit.startswith("h") else 60 if unit.startswith("m") else 1
        value = int(amount * multiplier)
        return value if 0 <= value <= 7200 else None
    return None


def _estimated_seconds(args: dict[str, Any]) -> int | None:
    return coerce_est_seconds((args or {}).get("est_seconds"))


def _needs_approval_python(
    call: ToolCall,
    args: dict[str, Any],
    state: ResearchState,
    session_dir: Path,
    threshold_seconds: int,
    *,
    python_soft_timeout_s: int | None = None,
) -> ApprovalRequest | None:
    code = str(args.get("code") or "")
    if not code.strip():
        return None
    description = str(args.get("description") or "Run a Python cell")
    est = _estimated_seconds(args)

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
                f"**{description}** — this cell looks destructive: "
                + "; ".join(destructive)
                + ". The agent's code runs directly on this machine with no sandbox."
            ),
            code=code,
            details={"issues": destructive, "dataset_paths": [str(p) for p in dataset_paths]},
        )

    if est is not None and est > threshold_seconds:
        note = ""
        if python_soft_timeout_s is not None and est + 180 > python_soft_timeout_s:
            effective = min(3600, max(python_soft_timeout_s, est + 180))
            note = f" NOTE: the cell timeout will be ~{effective}s (estimate + headroom)."
        return ApprovalRequest(
            id=call.id,
            tool=call.name,
            title="Long-running cell",
            reason=(
                f"**{description}** — the agent estimated ~{est}s of runtime "
                f"(above the {threshold_seconds}s approval threshold). Approve?{note}"
            ),
            code=code,
            details={"est_seconds": est, "threshold": threshold_seconds},
        )

    heavy = find_heavy_issues(code)
    if heavy:
        if est is not None and est <= threshold_seconds:
            # The model gave a fast estimate under the threshold: trust it and
            # run without asking. (n_jobs=-1 on a small frame is seconds, not
            # a cluster job.) The estimate stays visible in the step title.
            return None
        hint = (
            "Ask for this work with a realistic `est_seconds` value on the `python` tool "
            "next time so the estimate can be shown here."
            if est is None
            else f"Estimated runtime supplied by the agent: ~{est}s."
        )
        grid = count_grid_combinations(code)
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
    # Mirror files.write_file exactly: relative paths land in outputs/, and both
    # sides resolve symlinks the same way so the two cannot disagree.
    raw = Path(rel.strip()).expanduser()
    candidate = raw if raw.is_absolute() else session_dir / "outputs" / raw
    try:
        target = candidate.resolve()
        root = session_dir.resolve()
        inside = target.is_relative_to(root)
    except (OSError, ValueError, RuntimeError):
        inside, target = False, candidate
    if not inside or target == session_dir:
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


__all__ = [
    "ApprovalRequest",
    "needs_approval",
    "find_destructive_issues",
    "find_heavy_issues",
    "outside_session",
]
