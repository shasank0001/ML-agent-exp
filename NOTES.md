# NOTES.md — DataLab Agent V1

Assumptions, deviations and open questions made while building V1.
Kept as the running log required by the build spec; the "Later" list at the
bottom is for things deliberately **not** built.

---

## 1. Deviations from the build spec

### Dependencies
- **`requests` added explicitly.** `chainlit==2.12.0` imports
  `literalai -> traceloop.sdk -> requests` at import time but does not declare
  `requests` as a dependency, so a clean `uv pip install` produced an
  `ImportError` on `import chainlit`. Added to `pyproject.toml`; remove and pin
  an older chainlit if upstream fixes it.
- **`README.md` added** and used as the `pyproject` readme. The spec's layout did
  not list one; a repo that will be pushed to GitHub needs it.

### `datalab/lab.py`
- **`lab.split` returns encoded numeric frames**, not the raw columns. The spec
  only said "-> X_train, X_test, y_train, y_test". Encoding inside `split` means
  a cell like `RandomForestClassifier().fit(X_train, y_train)` just works,
  instead of failing on a string column and burning a repair iteration. The
  fitted `ColumnTransformer` and `feature_names` are exposed so the agent can
  still inspect the mapping.
  Encoding rules: numeric -> median impute; categorical with <= 50 levels ->
  most-frequent impute + one-hot; categorical with > 50 levels -> ordinal
  (unknown = -1) to stop the feature count exploding; datetime-looking ->
  epoch seconds.
- **`baseline()` names its experiments** `baseline_dummy` and
  `baseline_logreg` / `baseline_ridge`, so the results table shows which row is
  the reference.
- **`set_target` without a `task_type` records `classification` as a
  placeholder.** Inference needs the data, and `set_target` does not receive
  it. `lab.split` always re-infers from the real series, so the placeholder
  never survives a real pipeline.
- **Line budget.** `lab.py` is 329 lines against the "~300" guideline rather
  than under it. The extra lines are the encoding logic, which is the part that
  has to be right. Splitting it into `lab.py` + `lab_encode.py` was considered
  and rejected as churn for V1.

### `datalab/agent.py`, `conversation.py`, `tool_dispatch.py`
- **The loop is driven by a task, not directly by the generator.** `run()`
  delegates to `_drive()` running as an `asyncio.Task` that pushes events onto
  a queue. Without this, Chainlit's stop button has nothing to cancel: an async
  generator only runs while the consumer is pulling from it, so
  `agent.cancel()` had no handle.
- **The driver yields to the loop after every event.** `Queue.put` on an
  unbounded queue never suspends, so a turn made only of fast tools ran all the
  way to the step budget without the UI ever getting a slot — and the stop
  button had nothing to interrupt. One `asyncio.sleep(0)` per event fixed it.
- **`conversation.py` owns the message history.** Building messages, bounding
  the window, and repairing orphaned tool calls are fiddly enough to deserve
  their own module; keeping them in `agent.py` pushed it past 500 lines.
- **`tool_dispatch.py` owns "a tool call arrived".** Resolving the tool,
  deciding on approval, running the handler and turning the outcome into events
  is a separate concern from the loop that calls it.
- **The tool_call/tool pairing invariant is enforced once per turn**, at the end
  of `_drive`, rather than at each of the several places a turn can stop. An
  assistant message carrying `tool_calls` with no matching tool response makes
  the OpenAI API reject every later request, so it is worth being structural
  about.
- **Token usage is reported on the `done` event** (per-turn and per-session)
  rather than as its own event type, to keep the `EventType` literal exactly as
  specified. Only providers that return usage populate it.
- **The in-process message history is bounded** to `3 * context_window_messages`
  after a turn, cut only at a point where no tool call is orphaned.
  `events.jsonl` keeps the full transcript, so nothing is lost.

### `datalab/events.py`
- **`EventType` has two extra members** beyond the spec's list: `ask_user` and
  `ask_user_response`. The spec's rendering table includes an `ask_user` row but
  the enum does not, so a clarifying question could not have been rendered
  without extending the enum. The schema is otherwise unchanged.

### `datalab/tools/files.py`
- **`read_file` searches `outputs/`, then `data/`, then `figures/`, then the
  session root**, for a bare relative name. `write_file` writes to `outputs/`
  and nothing else. The spec did not pin the base for reads; without the search
  the agent had to spell out `outputs/report.md` every time.
- **`list_files` is recursive by default.** The agent's first instinct is
  "what files are there?", and a non-recursive root listing returns only the
  three sub-folders.

### `datalab/approvals.py` + `datalab/approval_heuristics.py`
- **Split in two.** The policy (which rule fires, what the dialog says, what
  happens on each answer) is in `approvals.py`; the code-shape analysis is in
  `approval_heuristics.py`. A single file reached 650 lines once the AST pass
  landed, well past the "keep modules small" convention.
- **The analysis is regex + a light `ast` pass, not regex alone.** Regexes miss
  f-strings, `getattr(os, "remove")`, `n_estimators=10**5` and `Path(x) / ".."`.
  The AST pass closes those; the regexes are kept as a backstop for cells that
  will not parse.
- **Path-safety heuristics understand "anchored" expressions.** A literal like
  `'/clean.csv'` inside `df.to_csv(OUTPUT_DIR + '/clean.csv')` is a suffix of a
  known-safe root, not an absolute path. Without this, every legitimate write to
  `OUTPUT_DIR` triggered an approval prompt. Climbing out with
  `'/../../etc/x'` is still caught.
- **A missing `est_seconds` is itself a reason to ask** on a heavy cell, and the
  prompt tells the model to supply the estimate next time (the spec asks for
  this in rule 4).
- **`getattr` / `eval` / `exec` are only flagged when they actually look
  dangerous** — a `getattr` whose args name a sensitive module, an `eval` whose
  string contains a destructive call. Flagging them unconditionally would make
  the dialog useless.

### Found by running it against a real model
A live rehearsal against `stealth/space-bunny-alpha` on OpenRouter (800-row
churn dataset) exposed four things the offline rehearsal could not:

1. **The agent read `lab`'s own source with `inspect.getsource`** to work out
   how to re-split, then poked `lab.X_train = None` to bust the cache — a
   private attribute, and a class of mistake a user watching the chat would see.
   Fixed by (a) adding `lab.split(..., force=True)` and `lab.reset_split()`, and
   (b) putting the complete `lab` API in the system prompt with "you do not need
   to read `lab`'s source or use `dir()`/`inspect`". The same run went from 25
   steps (hitting the budget) to 9.
2. **Re-running the baselines logged duplicate experiments.** After a
   re-split, `baseline_logreg` appeared twice with different numbers, and the
   results table invited the reader to compare a run against itself.
   `lab.evaluate` now replaces an experiment of the same name.
3. **A transient provider hiccup ended the turn.** OpenRouter occasionally
   injects a corrupt SSE frame (`APIError: JSON error injected into SSE
   stream`). The OpenAI SDK does not retry that, so the turn died mid-task.
   `llm.stream` now retries a request that failed before showing any output, and
   the agent retries the whole step twice more; a failure after output has
   started is surfaced as a non-fatal message telling the model to continue from
   the research state.
4. **`__import__("sklearn.ensemble")` triggered a false-positive approval**
   prompt. The gate now flags `__import__` only when it names a sensitive
   module, and `__import__("os").remove(p)` is caught by a precise rule instead
   of a blanket one.

### UI (`app.py`)
- **One `cl.Step` per tool call**, updated in place when the result arrives
  (`tool_start` opens it, `tool_result` fills it), rather than a separate
  "output" step. The approval round trip can take minutes, and a step that
  opens on `tool_start` and closes on `tool_result` stays coherent throughout.
- **The results table is rendered from `agent.lab.results_table()`** as a
  `cl.Dataframe` at the end of any turn that logged a new experiment. This is
  the only path that puts the harness-owned numbers in front of the user, and it
  is the same call the agent makes, so the two cannot disagree.
- **`_ask_prompt` / `_approval_prompt` swallow UI exceptions and return a safe
  default** (no answer / denied) so a disconnected browser cannot wedge a turn.
- **`cl.AskUserMessage` / `cl.AskActionMessage` timeouts are 30 and 30 minutes**
  (the Chainlit defaults are 60s, which is far too short for someone reading a
  traceback before answering).

---

## 2. Known limitations (accepted for V1)

1. **No sandbox.** LLM-generated code runs in the app process with full access
   to the machine. Localhost-bound, single user, approval gate for the obvious
   cases. Documented in `README.md`.
2. **The Python soft timeout cannot kill a thread.** `asyncio.wait_for` gives
   up on the await, but the worker thread keeps running. A timed-out cell leaves
   orphaned work behind: the thread keeps mutating the shared namespace and the
   global pyplot state with no lock, racing the next cell. On timeout the
   process-wide state the cell owned (stdout, stderr, working directory) is put
   back immediately, because `redirect_stdout` restores whatever was bound when
   it was *entered* — an abandoned thread's later exit would otherwise rebind a
   dead buffer over the live process stdout and silently swallow every
   subsequent print. The error message tells the model all of this. A real fix
   needs a subprocess or a Jupyter kernel (the "Later" list).
3. **`os.chdir`, `redirect_stdout` and `redirect_stderr` are all
   process-global.** Cells run with the session folder as their working
   directory so that a bare `df.to_csv('out.csv')` lands inside the session
   rather than in the app's working directory — but that makes concurrency
   unsafe. Fine for one user; the code says so.
4. **The approval policy is regex + light AST on the code text**, not a real
   analysis. It catches the shapes in the spec and misses computed or obfuscated
   paths. It is a speed bump, not a security boundary.
5. **Tool output longer than `MAX_TOOL_OUTPUT_CHARS` is truncated head+tail**
   before it goes back into the model's context. The event log holds the same
   truncated string, not the full output — so the log is faithful to what the
   model saw, not to what the process printed. Chosen deliberately: an
   unbounded log is a disk-fill risk in a demo.
6. **`profile_dataset` will not propose a continuous target.** Target
   candidates exclude columns that are more than 50% unique, so a fully
   continuous regression target (e.g. `np.random.randn`) yields no candidate and
   `suggested_task_type` is `None`. Numeric columns named like a target
   (`price`, `revenue`, ...) with 20-50% uniqueness do get proposed, and the
   agent can always set the target explicitly. Relaxing the uniqueness cutoff
   makes the profiler much noisier on datasets with an `id`-like feature, so V1
   keeps the strict rule and relies on `ask_user`.
7. **No cross-session memory.** State lives in `runs/<session_id>/state.json` and
   is not read back by a later session.
8. **`cl.Dataframe` / `cl.TaskList` are version-dependent.** They exist in
   `chainlit>=2.12.0`, which is the pinned floor. On an older install the
   results table and the to-do list would fall over; the rest of the app would
   not.

---

## 3. Assumptions made

- **Session id == Chainlit thread id**, so `runs/<thread_id>/` is stable across
  a page reload and matches what Chainlit shows in the UI.
- **The target column is confirmed with the user** whenever the profile
  proposes more than one plausible candidate, rather than guessing. Guessing
  makes every downstream number wrong in a way the user cannot see.
- **Classification is detected by target dtype plus cardinality**, not by the
  name of the column. Both `lab.infer_task_type` and `tools/profile.py` use the
  same rules so the profiler and the harness never disagree.
- **The `python` tool's `est_seconds` is the only cost signal.** There is no
  static cost model for a cell; the agent is asked to estimate, and a heavy
  pattern without an estimate asks the user anyway.
- **Uploads are copied into `runs/<id>/data/` and never moved.** The original
  upload location is not tracked, so nothing outside the session folder is
  read or written.
- **`lightgbm` and `xgboost` are dependencies but the agent picks the model.**
  `lab` ships only dummy + linear baselines; the trees are the LLM's job, which
  is the point of the "harness owns ground truth" split.

---

## 4. Open questions

- `lab.evaluate` replaces a same-named experiment (M5) so re-running baselines
  rewrites the row instead of duplicating it. History is intentionally not kept;
  a mixed-split table (new-split rows next to never-re-run rows) has no warning yet.
- The context window is a fixed last-N-messages. A summarising compaction step
  would survive longer sessions better; the state summary already carries most
  of the durable content, so V1 was left simple.
- Approval timeouts (30 min) and `MAX_STEPS=60` are demo-tuned. Neither has been
  swept. One live run used 9 steps and 46k prompt tokens for the full golden path;
  a careful exploratory run used 25 and 153k. Both fit, but the ceiling is close.
  Base cell timeout is `PYTHON_SOFT_TIMEOUT_S=1200` (actual = max(base, est+180),
  cap 3600) since training cells of 5-20 min are normal; timeouts never count
  toward `MAX_REPAIRS`.
- `stealth/space-bunny-alpha` occasionally emits a malformed SSE frame through
  OpenRouter. The step retry absorbs it, but a model that fails this way often
  will feel slow. Worth re-checking before a demo.
- The approval gate flags a cell for `n_jobs=-1` even when the cell is fast.
  That is a deliberate false positive (it does use every core), but it is the
  kind of prompt that trains users to click Approve without reading.

---

## Later (deliberately out of V1)

Real sandbox (Jupyter kernel or Docker); CLI / Python SDK / REST API over the
same event stream; playbooks and skills; AIDE-style metric-guided experiment
search; cross-session memory; an evaluation harness comparing this agent with a
generic LLM, a coding agent, `smolagents CodeAgent` and a LangGraph agent on
reliability, reproducibility and efficiency; concurrent sessions; a real static
analysis pass for the approval policy.
