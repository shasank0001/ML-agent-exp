"""Persistent research state.

The agent reasons about previous work ("I already tried random forest, it
scored X") through this structure, which is dumped to
``runs/<session_id>/state.json`` after every tool call.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

TaskType = Literal["classification", "regression"]


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class DatasetInfo(BaseModel):
    """Deterministic summary of an uploaded file (never LLM-written)."""

    path: str
    n_rows: int = 0
    n_cols: int = 0
    columns: list[dict[str, Any]] = Field(default_factory=list)
    profile_notes: list[str] = Field(default_factory=list)
    duplicates: int = 0
    target_candidates: list[str] = Field(default_factory=list)
    suggested_task_type: TaskType | None = None
    created_at: str = Field(default_factory=_now)


class TaskSpec(BaseModel):
    """The prediction task the user asked for."""

    target: str
    task_type: TaskType
    primary_metric: str
    split: dict[str, Any] = Field(default_factory=dict)


class TodoItem(BaseModel):
    id: str
    text: str
    status: Literal["pending", "in_progress", "done"] = "pending"


class Experiment(BaseModel):
    """One model evaluated on the held-out test set by :mod:`datalab.lab`."""

    id: str
    name: str
    model: str
    params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    primary_metric: str = ""
    status: Literal["done", "failed"] = "done"
    notes: str = ""
    created_at: str = Field(default_factory=_now)

    @property
    def primary_value(self) -> float | None:
        v = self.metrics.get(self.primary_metric)
        return float(v) if isinstance(v, (int, float)) else None


class ResearchState(BaseModel):
    """Everything the agent knows about this session's work."""

    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    root_dir: str = Field(default="", exclude=True)
    dataset: DatasetInfo | None = None
    task: TaskSpec | None = None
    plan: list[TodoItem] = Field(default_factory=list)
    experiments: list[Experiment] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)

    # -- persistence ----------------------------------------------------
    @property
    def state_path(self) -> Path:
        return Path(self.root_dir) / "state.json"

    def save(self, path: Path | None = None) -> Path:
        """Write the state to disk. Returns the path written."""
        target = Path(path) if path else self.state_path
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(target)
        return target

    @classmethod
    def load(cls, path: Path) -> "ResearchState":
        """Read state back from disk."""
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def create(cls, root_dir: Path) -> "ResearchState":
        state = cls(root_dir=str(root_dir))
        state.save()
        return state

    # -- mutation helpers ----------------------------------------------
    def add_finding(self, text: str) -> str:
        text = text.strip()
        if text and text not in self.findings:
            self.findings.append(text)
        return text

    def set_plan(self, items: list[dict[str, Any]]) -> list[TodoItem]:
        """Replace the whole to-do list. Malformed entries are normalised, not fatal."""
        statuses = {"pending", "in_progress", "done"}
        plan: list[TodoItem] = []
        for raw in items:
            if not isinstance(raw, dict):
                continue
            text = str(raw.get("text", "")).strip()
            if not text:
                continue
            status = str(raw.get("status", "pending")).strip().lower()
            plan.append(
                TodoItem(
                    id=str(raw.get("id") or len(plan) + 1),
                    text=text,
                    status=status if status in statuses else "pending",  # type: ignore[arg-type]
                )
            )
        self.plan = plan
        return self.plan

    def add_experiment(self, exp: Experiment) -> Experiment:
        self.experiments.append(exp)
        return exp

    # -- views ----------------------------------------------------------
    @property
    def primary_metric(self) -> str | None:
        return self.task.primary_metric if self.task else None

    def done_experiments(self) -> list[Experiment]:
        return [e for e in self.experiments if e.status == "done"]

    def sorted_experiments(self) -> list[Experiment]:
        """Experiments ordered best-first for the primary metric (lower is better for errors)."""
        metric = self.primary_metric or ""
        lower_is_better = metric.lower() in {"rmse", "mae", "mse", "log_loss", "mape", "rmsle"}

        def key(e: Experiment) -> tuple[int, int, float]:
            value = e.primary_value
            if e.status != "done" or value is None:
                return (2, 0, 0.0)
            return (0, 0, value if lower_is_better else -value)

        return sorted(self.experiments, key=key)

    def best_experiment(self) -> Experiment | None:
        ranked = [e for e in self.sorted_experiments() if e.status == "done"]
        return ranked[0] if ranked else None

    def summary_dict(self) -> dict[str, Any]:
        """Small dict sent to the UI on every ``state_update`` event."""
        best = self.best_experiment()
        return {
            "plan": [{"id": t.id, "text": t.text, "status": t.status} for t in self.plan],
            "dataset": None
            if self.dataset is None
            else {
                "path": self.dataset.path,
                "name": Path(self.dataset.path).name,
                "n_rows": self.dataset.n_rows,
                "n_cols": self.dataset.n_cols,
            },
            "task": None
            if self.task is None
            else {
                "target": self.task.target,
                "task_type": self.task.task_type,
                "primary_metric": self.task.primary_metric,
            },
            "n_experiments": len(self.experiments),
            "best": None if best is None else {"name": best.name, "metrics": best.metrics},
            "n_findings": len(self.findings),
        }

    def summary_text(self, max_chars: int = 6_000) -> str:
        """Compact text block injected into the system prompt every turn."""
        lines: list[str] = ["## Current research state"]

        if self.dataset is None:
            lines.append("- Dataset: none profiled yet.")
        else:
            ds = self.dataset
            lines.append(
                f"- Dataset: {Path(ds.path).name} — {ds.n_rows} rows x {ds.n_cols} cols, "
                f"{ds.duplicates} duplicate rows."
            )
            missing = [c["name"] for c in ds.columns if c.get("n_missing", 0)]
            if missing:
                lines.append(f"  - Columns with missing values: {', '.join(missing[:12])}")
            if ds.target_candidates:
                lines.append(f"  - Suggested target candidates: {', '.join(ds.target_candidates[:8])}")
            if ds.suggested_task_type:
                lines.append(f"  - Suggested task type: {ds.suggested_task_type}")

        if self.task is None:
            lines.append("- Task: not defined yet (ask the user for the target column).")
        else:
            t = self.task
            split = t.split or {}
            lines.append(
                f"- Task: {t.task_type} on target `{t.target}`, primary metric `{t.primary_metric}`."
            )
            if split:
                lines.append(
                    f"  - Split: seed={split.get('seed')}, test_size={split.get('test_size')}, "
                    f"stratified={split.get('stratified')}, "
                    f"n_train={split.get('n_train')}, n_test={split.get('n_test')}"
                )
            else:
                lines.append("  - Split: not created yet — call `lab.split(...)` before training.")

        if self.plan:
            lines.append("- Plan:")
            lines.extend(f"  {i + 1}. [{t.status}] {t.text}" for i, t in enumerate(self.plan))
        else:
            lines.append("- Plan: none yet (use the `todo` tool to write one).")

        if self.experiments:
            lines.append(f"- Experiments logged ({len(self.experiments)}), best first:")
            for e in self.sorted_experiments()[:8]:
                metrics = ", ".join(f"{k}={_fmt(v)}" for k, v in e.metrics.items())
                lines.append(f"  - {e.name} [{e.model}] {metrics}")
        else:
            lines.append("- Experiments: none yet (run `lab.baseline()` first).")

        if self.findings:
            lines.append("- Findings:")
            lines.extend(f"  - {f}" for f in self.findings[-6:])

        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[: max_chars - 40] + "\n... (state summary truncated)"
        return text


def _fmt(v: Any) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{f:.4f}"
