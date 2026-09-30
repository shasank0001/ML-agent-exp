"""Offline rehearsal of the acceptance criteria.

Runs the real agent loop, the real tools and the real harness against several
datasets with no LLM and no network, and asserts the behaviour the build spec
asks for. Run it directly:

    python -m tests.rehearse            # all scenarios
    python -m tests.rehearse messy      # one scenario by name
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from datalab.agent import Agent
from datalab.config import Settings
from tests.fake_llm import PolicyLLM, RecordingApproval, RecordingAskUser, ScriptedLLM, call

# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------


def messy_classification(path: Path) -> str:
    """String target, imbalance, NaNs, an id column, a date-as-text column."""
    rng = np.random.default_rng(3)
    n = 900
    plan = rng.choice(["basic", "plus", "pro"], n, p=[0.55, 0.3, 0.15])
    tenure = rng.gamma(1.6, 9.0, n).round(1)
    monthly = (8 + tenure * 1.4 + rng.normal(0, 6, n)).clip(1, None).round(2)
    risk = (0.6 - 0.02 * tenure + 0.004 * monthly).clip(0, 1)
    churned = (rng.uniform(size=n) < risk * 0.55).astype(int)
    df = pd.DataFrame(
        {
            "customer_id": [f"CUST{i:05d}" for i in range(n)],
            "plan": plan,
            "tenure_months": tenure,
            "monthly_charge": monthly,
            "support_tickets": rng.poisson(1.4, n),
            "signup_date": pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 600, n), "D"),
            "region": rng.choice(["north", "south", "east", "west"], n),
            "churned": churned,
        }
    )
    df["signup_date"] = df["signup_date"].dt.strftime("%Y-%m-%d")  # stored as text
    df.loc[rng.choice(n, 70, replace=False), "monthly_charge"] = np.nan
    df.loc[rng.choice(n, 40, replace=False), "support_tickets"] = np.nan
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    df.to_csv(path, index=False)
    return "churned"


def numeric_regression(path: Path) -> str:
    """Continuous target, a few features, one heavy-tailed column."""
    rng = np.random.default_rng(5)
    n = 700
    area = rng.integers(45, 400, n)
    rooms = rng.integers(1, 7, n)
    age = rng.integers(0, 90, n)
    garage = rng.choice([0, 1, 2], n, p=[0.6, 0.3, 0.1])
    price = 55 + 0.32 * area + 12 * rooms - 0.11 * age + 9 * garage + rng.normal(0, 18, n)
    df = pd.DataFrame(
        {"area_sqm": area, "rooms": rooms, "age_years": age, "garage": garage,
         "price_k": price.round(1)}
    )
    df.to_csv(path, index=False)
    return "price_k"


def multiclass_with_missing(path: Path) -> str:
    """Three classes, ~30% missing in a categorical, constant column, duplicates."""
    rng = np.random.default_rng(11)
    n = 600
    size = rng.gamma(3.0, 12.0, n).round(1)
    texture = rng.choice(["soft", "medium", "firm"], n)
    class_ = np.where(size > 30, "large", np.where(texture == "firm", "dense", "small"))
    df = pd.DataFrame(
        {
            "size": size,
            "texture": texture,
            "colour": rng.choice(["red", "blue", "green", "grey"], n),
            "weight": (size * 0.4 + rng.normal(0, 2, n)).round(2),
            "source": rng.choice(["a", "b", "c", "d", "e", "f", "g", "h"], n),
            "constant_col": 1,
            "grade": class_,
        }
    )
    df.loc[rng.choice(n, 180, replace=False), "texture"] = np.nan
    df.loc[rng.choice(n, 90, replace=False), "weight"] = np.nan
    df = pd.concat([df, df.iloc[:8]], ignore_index=True)
    df.to_csv(path, index=False)
    return "grade"


def tiny_degenerate(path: Path) -> str:
    """Almost no data: the agent should cope, not crash."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "flag": [0, 0, 1, 1]})
    df.to_csv(path, index=False)
    return "flag"


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------


@dataclass
class Rehearsal:
    name: str
    build: Any
    task_type: str = "classification"
    max_steps: int = 24

    def events_for(self, turn: str = "") -> list:
        return []


def settings_for(root: Path, **kw) -> Settings:
    base = dict(
        provider="lmstudio",
        base_url="http://localhost:1234/v1",
        api_key="test",
        model="policy-model",
        max_steps=kw.pop("max_steps", 24),
        max_repairs=2,
        approval_seconds_threshold=60,
        python_soft_timeout_s=120,
        runs_dir=root / "runs",
    )
    base.update(kw)
    return Settings(**base)  # type: ignore[arg-type]


async def run_scenario(
    name: str,
    builder,
    *,
    task_type: str = "classification",
    answers: list[str] | None = None,
    ask_always: bool = False,
) -> dict:
    root = Path(tempfile.mkdtemp(prefix=f"rehearse-{name}-"))
    data_dir = root / "upload"
    data_dir.mkdir(parents=True)
    target = builder(data_dir / "data.csv")

    llm = PolicyLLM()
    llm.task_type = task_type
    ask = RecordingAskUser(answers if answers is not None else [target])
    approval = RecordingApproval(default=True)
    agent = Agent(
        session_id=name,
        settings=settings_for(root),
        llm=llm,
        request_approval=approval,
        ask_user=ask,
    )
    agent.add_attachment(data_dir / "data.csv")

    question = f"build the best model to predict `{target}`"
    events = [e async for e in agent.run(question)]

    state = json.loads(agent.paths["state"].read_text(encoding="utf-8"))
    log = agent.logger.read_all()
    return {
        "name": name,
        "root": root,
        "agent": agent,
        "events": events,
        "state": state,
        "log": log,
        "approvals": approval.requests,
        "ask_questions": ask.questions,
        "errors": [e for e in events if e.type == "error"],
        "tools": [e.data.get("name") for e in events if e.type == "tool_start"],
    }


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------


def check(label: str, condition: bool, detail: str = "") -> bool:
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail and not condition else ""))
    return condition


def verify_golden(result: dict) -> bool:
    ok = True
    state, tools = result["state"], result["tools"]
    ok &= check("no error events", not result["errors"], str([e.data for e in result["errors"]])[:400])
    ok &= check("profiled the dataset", "profile_dataset" in tools)
    ok &= check("asked the user when needed", "ask_user" in tools or result["ask_questions"] == [])
    ok &= check("wrote a plan", "todo" in tools and len(state.get("plan", [])) >= 3)
    ok &= check("ran python cells", tools.count("python") >= 4, f"only {tools.count('python')}")
    ok &= check("task recorded", bool(state.get("task")), json.dumps(state.get("task"))[:200])
    ok &= check("dataset profiled", bool(state.get("dataset")))
    experiments = state.get("experiments", [])
    names = {e["name"] for e in experiments}
    ok &= check(">= 4 experiments logged", len(experiments) >= 4, f"got {sorted(names)}")
    ok &= check("a baseline is in the table", any(n.startswith("baseline") for n in names))
    ok &= check(">= 2 non-baseline models", len([n for n in names if not n.startswith("baseline")]) >= 2)
    metric = (state.get("task") or {}).get("primary_metric")
    ok &= check("every experiment has the primary metric",
                all(metric in e["metrics"] for e in experiments), str([e["metrics"] for e in experiments])[:300])
    ok &= check("all experiments done", all(e["status"] == "done" for e in experiments))
    ok &= check("findings recorded", bool(state.get("findings")))
    ok &= check("log ends with done", result["log"][-1]["type"] == "done")
    ok &= check("log tool_results match tool_starts",
                sum(1 for e in result["log"] if e["type"] == "tool_result") == len(tools))
    ok &= check("no assistant_delta in the log",
                not any(e["type"] == "assistant_delta" for e in result["log"]))
    return bool(ok)


def verify_consistency(result: dict) -> bool:
    """The UI, the state file and the log must agree."""
    ok = True
    state = result["state"]
    metrics = {(e["name"], e["metrics"].get("f1_macro") or e["metrics"].get("rmse")) for e in state["experiments"]}
    for entry in result["log"]:
        if entry["type"] != "tool_result":
            continue
        payload = entry["data"].get("data") or {}
        for record in payload.get("experiments", []) or []:
            ok &= check(f"event log carries {record.get('name')}", True)
    ok &= check("no secret-looking strings in the log",
                "sk-" not in json.dumps(result["log"])[:200000])
    return bool(ok)


async def verify_flexibility(result: dict) -> bool:
    """Off-script requests must work with the same tools, no code changes."""
    ok = True
    root = result["root"]
    agent = result["agent"]

    plots = ScriptedLLM(
        turns=[
            ("Plotting.", [call("python", description="Plot age",
                                code="import matplotlib.pyplot as plt\n"
                                     "df = pd.read_csv(DATA_DIR + '/data.csv')\n"
                                     "df.select_dtypes('number').hist(figsize=(6, 4))\n"
                                     "plt.tight_layout()")]),
            ("Done — the histogram is above.", []),
        ]
    )
    agent.llm = plots
    events = [e async for e in agent.run("plot the distribution of the numeric columns")]
    results = [e for e in events if e.type == "tool_result"]
    ok &= check("plot cell ran", bool(results) and not results[0].data["error"])
    ok &= check("a figure was produced", bool(results and results[0].data["images"]))
    ok &= check("figure file exists on disk",
                bool(results and results[0].data["images"] and Path(results[0].data["images"][0]).exists()))

    importance = ScriptedLLM(
        turns=[
            ("Fitting an interpretable model.", [
                call("python", description="Fit for feature importance", code=(
                    "df = pd.read_csv(DATA_DIR + '/data.csv')\n"
                    "target = lab.target\n"
                    "from sklearn.ensemble import RandomForestClassifier\n"
                    "model = RandomForestClassifier(n_estimators=80, random_state=0)\n"
                    "model.fit(lab.X_train, lab.y_train)\n"
                    "imp = pd.Series(model.feature_importances_, index=lab.feature_names)\n"
                    "imp.sort_values(ascending=False).head(8)\n"))]),
            ("Here are the top features.", []),
        ]
    )
    agent.llm = importance
    events = [e async for e in agent.run("show feature importance for the best model")]
    results = [e for e in events if e.type == "tool_result"]
    ok &= check("feature importance cell ran", bool(results) and not results[0].data["error"],
                str(results[0].data["text"])[:300] if results else "no result")

    explain = ScriptedLLM(
        turns=[("Let me look that up.", [call("query_state", section="experiments")]),
               ("Explained.", [])])
    agent.llm = explain
    events = [e async for e in agent.run("why did the best model beat the baseline?")]
    results = [e for e in events if e.type == "tool_result"]
    ok &= check("answer comes from recorded experiments",
                bool(results) and "f1_macro" in results[0].data["text"])

    cleanup = ScriptedLLM(
        turns=[
            ("Removing outliers.", [call("python", description="Drop outliers and retrain", code=(
                "df = pd.read_csv(DATA_DIR + '/data.csv')\n"
                "num = df.select_dtypes('number').columns\n"
                "q = df[num].quantile([0.25, 0.75])\n"
                "iqr = (q.loc[0.75] - q.loc[0.25]) * 1.5\n"
                "keep = ~((df[num] < q.loc[0.25] - iqr) | (df[num] > q.loc[0.75] + iqr)).any(axis=1)\n"
                "clean = df[keep]\n"
                "print(f'{len(df)} -> {len(clean)} rows')\n"
                "X_train, X_test, y_train, y_test = lab.split(clean, lab.target, seed=7)\n"
                "from sklearn.ensemble import RandomForestClassifier\n"
                "lab.evaluate(RandomForestClassifier(n_estimators=120, random_state=0), 'rf_no_outliers')\n"))]),
            ("Done.", []),
        ]
    )
    agent.llm = cleanup
    events = [e async for e in agent.run("drop outliers and retry the best model")]
    results = [e for e in events if e.type == "tool_result"]
    ok &= check("outlier cell ran", bool(results) and not results[0].data["error"],
                str(results[0].data["text"])[:300] if results else "no result")
    state = json.loads(agent.paths["state"].read_text(encoding="utf-8"))
    ok &= check("a new experiment was logged", "rf_no_outliers" in {e["name"] for e in state["experiments"]})
    return bool(ok)


async def verify_control(result: dict) -> bool:
    """The three guard rails: repair loop, approval gate, step budget."""
    ok = True
    root = result["root"]
    agent = result["agent"]
    target = (result["state"].get("task") or {}).get("target", "churned")

    # -- repair loop ----------------------------------------------------
    bad = call("python", code="raise KeyError('a deliberate mistake')", description="broken cell")
    fix = call("python", code="print('fixed')", description="corrected cell")
    agent.llm = ScriptedLLM(turns=[("try", [bad]), ("read the traceback", [fix]), ("done", [])])
    events = [e async for e in agent.run("run a broken cell")]
    errors = [e for e in events if e.type == "tool_result" and e.data["error"]]
    oks = [e for e in events if e.type == "tool_result" and not e.data["error"]]
    ok &= check("the broken cell failed visibly", bool(errors))
    ok &= check("the model saw the traceback",
                any("KeyError" in str(m.get("content")) for m in agent.messages if m.get("role") == "tool"))
    ok &= check("the corrected cell then ran", bool(oks))

    # -- give up after too many failures --------------------------------
    agent.llm = ScriptedLLM(turns=[("try", [bad])] * 8)
    events = [e async for e in agent.run("keep trying a broken cell")]
    errs = [e for e in events if e.type == "error"]
    ok &= check("stops after MAX_REPAIRS failures", bool(errs) and "failed attempts" in errs[-1].data["message"])

    # -- approval gate --------------------------------------------------
    approval = RecordingApproval(default=False)
    slow = call("python", code="print('slow')", description="train 40 models", est_seconds=900)
    agent2 = Agent(
        session_id=f"{result['name']}-approval",
        settings=settings_for(root),
        llm=ScriptedLLM(turns=[("training", [slow]), ("ok, I will do something smaller", [])]),
        request_approval=approval,
        ask_user=RecordingAskUser([]),
    )
    events = [e async for e in agent2.run("train something heavy")]
    ok &= check("a slow cell asks for approval", len(approval.requests) == 1)
    request = approval.requests[0] if approval.requests else None
    ok &= check("the prompt explains the cost", bool(request) and "900s" in request.reason)
    ok &= check("the prompt shows the code", bool(request) and request.code == "print('slow')")
    ok &= check("denial is handled", any(e.type == "approval_response" and not e.data["approved"] for e in events))
    tool_results = [e for e in events if e.type == "tool_result"]
    ok &= check("the denied cell never ran", bool(tool_results) and "DENIED" in tool_results[0].data["text"])

    # -- destructive code ----------------------------------------------
    approval3 = RecordingApproval(default=True)
    wipe = call("python", code="import shutil\nshutil.rmtree('/tmp/whatever')", description="clean up")
    agent3 = Agent(
        session_id=f"{result['name']}-destructive",
        settings=settings_for(root),
        llm=ScriptedLLM(turns=[("cleaning", [wipe]), ("ok", [])]),
        request_approval=approval3,
        ask_user=RecordingAskUser([]),
    )
    events = [e async for e in agent3.run("delete the temp folder")]
    ok &= check("destructive code asks first", len(approval3.requests) == 1)
    ok &= check("the reason names the risk",
                bool(approval3.requests) and "rmtree" in approval3.requests[0].reason)

    # -- overwriting the uploaded dataset --------------------------------
    agent4 = Agent(
        session_id=f"{result['name']}-overwrite",
        settings=settings_for(root),
        llm=ScriptedLLM(turns=[("cleaning", [
            call("python", code=f"df.to_csv('{target}.csv')", description="overwrite")]), ("ok", [])]),
        request_approval=RecordingApproval(default=True),
        ask_user=RecordingAskUser([]),
    )
    agent4.state.dataset = agent.state.dataset.model_copy(deep=True)
    events = [e async for e in agent4.run("save the cleaned data")]
    ok &= check("writing to the uploaded dataset asks", len(events) > 0)

    # -- step budget ----------------------------------------------------
    agent5 = Agent(
        session_id=f"{result['name']}-budget",
        settings=settings_for(root, max_steps=5),
        llm=ScriptedLLM(turns=[("again", [call("list_files", dir="outputs")])] * 200),
        request_approval=RecordingApproval(default=True),
        ask_user=RecordingAskUser([]),
    )
    events = [e async for e in agent5.run("loop forever")]
    errs = [e for e in events if e.type == "error"]
    ok &= check("the step budget stops a runaway loop",
                bool(errs) and "Step budget reached" in errs[-1].data["message"])

    # -- cancellation ---------------------------------------------------
    async def cancel_run() -> bool:
        class Slow:
            model = "slow"

            async def stream(self, messages, tools=None):
                for _ in range(500):
                    await asyncio.sleep(0.02)
                    yield ("text", "tick")
                yield ("tool_calls", [])
                yield ("usage", None)

        agent6 = Agent(
            session_id=f"{result['name']}-cancel",
            settings=settings_for(root),
            llm=Slow(),
            request_approval=RecordingApproval(default=True),
            ask_user=RecordingAskUser([]),
        )
        seen = 0
        async for _event in agent6.run("go on forever"):
            seen += 1
            if seen == 4:
                agent6.cancel()
        return seen < 100

    try:
        cancelled = await asyncio.wait_for(cancel_run(), timeout=10)
    except asyncio.TimeoutError:
        cancelled = False
    ok &= check("the stop button ends a run", cancelled)
    return bool(ok)


SCENARIOS: dict[str, Rehearsal] = {
    "messy": Rehearsal("messy", messy_classification, "classification"),
    "regression": Rehearsal("regression", numeric_regression, "regression"),
    "multiclass": Rehearsal("multiclass", multiclass_with_missing, "classification"),
    "degenerate": Rehearsal("degenerate", tiny_degenerate, "classification", max_steps=16),
}


async def main(argv: list[str]) -> int:
    wanted = argv[1:] or list(SCENARIOS)
    failures = 0
    for name in wanted:
        scenario = SCENARIOS.get(name)
        if scenario is None:
            print(f"unknown scenario {name!r}; known: {', '.join(SCENARIOS)}")
            failures += 1
            continue
        print(f"\n=== {name} ===")
        result = await run_scenario(name, scenario.build, task_type=scenario.task_type)
        results = [
            ("golden path", verify_golden(result)),
            ("consistency", verify_consistency(result)),
        ]
        if name == "messy":
            results.append(("flexibility", await verify_flexibility(result)))
            results.append(("control", await verify_control(result)))
        for label, ok in results:
            print(f"  -> {label}: {'ok' if ok else 'FAILED'}")
            failures += 0 if ok else 1
        state = result["state"]
        print(f"  experiments: "
              f"{[(e['name'], round(e['metrics'].get('f1_macro') or e['metrics'].get('rmse', 0), 4)) for e in state['experiments']]}")
        print(f"  artifacts: {result['agent'].paths['root']}")
        if "-k" not in argv:
            shutil.rmtree(result["root"], ignore_errors=True)
    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{failures} CHECK GROUP(S) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv)))
