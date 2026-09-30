# Codebase Review — Must-Do Fixes

Scope: `app.py`, `datalab/agent.py`, `config.py`, `llm.py`, `events.py`, `state.py`, `lab.py`, `approvals.py` + `approval_heuristics.py`, `prompts.py`, `conversation.py`, `datalab/tools/*`, `.chainlit/config.toml`, `tests/`.
Tests: 256 passed at review time — fixes below are hardening/correctness, not currently-red tests.

## P0 — Security (fix before any shared/demo use)

1. **Unsandboxed `python` tool is the top risk** — `datalab/tools/python_exec.py:174-213`
   LLM code runs in-app with `os/sys/subprocess/socket/shutil/pathlib` available, `exec`/`eval` (`:191,:193`), `except BaseException` (`:194`) swallowing `SystemExit`. Approval heuristics are denylist regex/AST and bypassable (`__import__('o'+'s')`, `importlib`, variable indirection — `tests/test_approvals.py:293-296` even enshrines `df.to_csv(path_variable)` as no-approval).
   Must-do: run cells in a subprocess/container with no network, kill on timeout, read-only `data/`, write-only `outputs/`+`figures/`, cap code size (~50 KB) and output (~20 KB). At minimum block `socket/subprocess/os.system/os._exit` at import and enforce session-root writes at runtime, not prompt-time.

2. **Timeout leaks threads, no cap** — `datalab/tools/python_exec.py:140-165`
   `asyncio.wait_for` + `asyncio.to_thread` abandons the worker; repeated timeouts accumulate threads/CPU/RAM. Add a live-thread registry, cap concurrent/abandoned cells, and test "second cell after timeout with mutated namespace".

3. **Process-global `chdir` + `redirect_stdout/stderr`** — `datalab/tools/python_exec.py:182-183,198-200`
   Noted in comments/`NOTES.md` but unfixed: concurrent Chainlit sessions corrupt each other. Must-do if >1 user: per-cell subprocess or remove `chdir` (use absolute paths) and capture output without global redirect.

4. **`session_id` used as path with no sanitization** — `datalab/config.py:125-142`, `datalab/agent.py:66`
   `runs_dir / session_id` with `session_id = cl.context.session.id` (`app.py:39-40`). `../` or absolute id escapes `runs/`. Must-do: allowlist `^[A-Za-z0-9_-]{8,64}$`, reject `..`/`/`, `resolve()` + containment check, `mkdir(mode=0o700)`. Same for `RUNS_DIR` env.

5. **`profile_dataset` path containment bypass** — `datalab/tools/profile.py:253-264`
   `_resolve_path` checks `Path(raw)`, `data_dir/raw`, `data_dir/basename` only. Absolute `/tmp/evil.csv` outside the session is profiled. `files.py:27-56` enforces containment; profile does not. Must-do: reuse `resolve_in_session` with `READ_BASES`, reject outside-session absolutes.

6. **`write_file` can clobber `state.json` / `events.jsonl` / `data/*`** — `datalab/tools/files.py:126-154`
   `WRITE_BASES=("outputs",)` still allows `../state.json`, `../events.jsonl` via `..` (only outside-root is blocked). Add deny-list for `state.json`, `events.jsonl`, `*.tmp`, `data/*` overwrites. Same enforcement needed for `python`-cell writes (currently approval-heuristic only).

7. **Unbounded uploads + permissive Chainlit config** — `app.py:127-136`, `datalab/tools/files.py:197-206`, `.chainlit/config.toml:14-17`
   No extension/size/count checks; `shutil.copy2` preserves mode; `Path(src).name` unsanitized; `accept=["*/*"]`, `max_files=20`, `max_size_mb=512`; `profile.py:212-219` does full `read_csv/parquet/excel` with no `nrows`/size guard (zip-bomb/parquet OOM). Must-do: allowlist `.csv/.tsv/.parquet/.xlsx/.xls`, cap size (~200 MB) + file count, sanitize filenames, lower Chainlit limits, chunked/guarded profiling.

8. **`_close_step` serves arbitrary disk paths** — `app.py:224-231`
   `Path(str(raw)).exists()` with no containment; tool output `images=["/etc/passwd"]` reaches `cl.Image(path=...)`. Must-do: require `is_relative_to(session root)` and `figures/` or `outputs/`, cap count/size.

9. **Secret redaction too narrow** — `datalab/config.py:68-73,105-107`, `datalab/events.py:118-131`
   `len(secret) >= 8` gate misses short keys; `SECRET_ENV_KEYS` covers 4 vars only; tests check only `sk-` (`tests/rehearse.py:238-240`). Only `text` is redacted before store in `agent.py:426`; `arguments`/`images`/`data` logged fuller. Must-do: lower gate to >=4 + strip, add `OPENAI_API_KEY/ghp_/AKIA/xoxb-` patterns, redact `arguments` and `data` before `_emit`, never store unredacted secret in `self.messages`.

## P1 — Correctness bugs

10. **Duplicate `_answer_orphans` definition** — `datalab/agent.py:298-300` vs `:308-327`
    First def is dead (second wins). Delete the first; keep the history-scanning version.

11. **Cancelled-error emit is discarded** — `datalab/agent.py:283-289`
    `self._emit("error",...)` return value dropped; only the second pushed emit survives. Stream vs. JSONL log diverge on cancel. Yield/queue the terminal event instead of bare `_emit`.

12. **`state.save()` can kill the turn** — `datalab/agent.py:436`, `datalab/state.py:89-96`
    Unguarded save inside `_run_one_tool`; an `OSError`/disk-full raises out of `_drive`, hanging `_loop` on `queue.get()`. Wrap in `try/except OSError` → emit error tool-result. Also `state_path` defaults to `Path("")/"state.json"` (CWD) when `root_dir=""` (`state.py:86-87`); require `root_dir`.

13. **`EventLogger` lock held across file IO + buffer race** — `datalab/events.py:60,89-93`
    `_buffer.append` outside lock (order race); `with self._lock, open("a")` blocks all emitters on disk. Move append inside lock or use queue; do file append via `asyncio.to_thread`; `fsync` or document loss window; surface `JSONDecodeError` count in `read_all` (`:111-114`) instead of silent skip.

14. **LLM stream never closed, inputs unvalidated** — `datalab/llm.py:86-100,141-192`
    No `try/finally: await stream.close()`; empty `model`/`api_key` falls back to `"not-set"` (`:96`) instead of raising `AgentError`; `timeout=300/max_retries=2` hardcoded. Validate before `create()`, make timeout/retries settings. Also `__raw_arguments__` sentinel (`:79`) is dispatched as a tool arg — reject explicitly in `agent.py:330-350` with "unknown tool args" error.

15. **`lab.evaluate` fitted-probe is fragile** — `datalab/lab.py:260-263`
    `model.predict(X_train.iloc[:1])` has side effects and misclassifies unfitted models that don't raise `NotFittedError`. Use `sklearn.utils.validation.check_is_fitted`. Also: validate `test_size in (0,1)` + int `seed` (`:185`), remove `print` (`:245`, use logger), allow duplicate experiment `name`s explicitly or refuse (currently silent duplicates — open question in `NOTES.md`).

16. **Re-profiling leaves stale `task`** — `datalab/tools/profile.py:267-274`
    Handler overwrites `state.dataset` but keeps old `task/split/experiments`; second dataset inherits wrong target/metric. `tests/test_profile.py:105-112` enshrines this. Must-do: if dataset path changes and `task` exists, clear `task`/split or warn explicitly + add multi-dataset test.

17. **`read_file` violates never-raise contract** — `datalab/tools/files.py:93,95-103`
    `int(args.get("max_chars"))` raises `ValueError/TypeError` on `"abc"/[]/{}`; `path.stat()` outside `try`; whole file read before truncate (large/binary OOM). Wrap types, `stat` inside `try`, stream/chunk read, use relative paths in errors (currently leaks absolute path `:90`).

18. **`ask_user` / `todo` handlers can raise** — `datalab/tools/ask_user.py:62-63`, `datalab/tools/todo.py:49-71`
    `answer.strip()` assumes `str`; `ctx.state.set_plan(cleaned)` `ValidationError` propagates. Add `isinstance` guard + `try/except` around `await agent.ask_user` and `set_plan`; cap todo items (~20), text length (~200), reject duplicate `id`s (currently coerced/silently skipped).

19. **Prompt threshold mismatch** — `datalab/tools/python_exec.py:279`, `datalab/prompts.py:52-53`, `datalab/config.py:116`
    Schema says "REQUIRED above ~30s", `APPROVAL_SECONDS_THRESHOLD` defaults to 60. Inject the real threshold into the prompt dynamically instead of hardcoding 30.

20. **Approval TOCTOU + `est_seconds` parsing** — `datalab/approvals.py:67-75,177-187`
    `target.exists()` then `stat()` twice; `est_seconds` accepts only `int/float/digit-str` (rejects `"90s"`, floats-as-string). `stat()` once via `lstat()`, re-check after approval or open `O_NOFOLLOW`; parse robustly; default-deny unknown tools instead of `return None` (`:64`).

## P2 — Robustness / validation gaps

21. **UI stream merge can diverge** — `app.py:162-168,191-192`
    `assistant_message` after streaming only calls `stream.update()` without setting final content; `finally: await stream.update()` can raise and masks `CancelledError`. Set `stream.content` before update; guard `finally` with `try/except`; reset `stream=None`.

22. **Tool-step keying + untyped `est`** — `app.py:170-174,204-205`
    `steps` keyed by `str(id)` collides on missing id (`"None"`); `est_seconds` rendered untyped (`~{'x':1}s`). Validate id present; `isinstance(est,(int,float))` check.

23. **Approval/ask failures indistinguishable from deny** — `app.py:288-330`
    Timeout/exception/`None`/explicit-deny all return `False`/`""`; `ask_user.py` then "proceeds with assumption". Distinguish `timed_out` vs `denied` vs `error` in `approval_response`/`ask_user_response`.

24. **Unbounded growth, weak history trim** — `datalab/agent.py:404,465-469`, `datalab/state.py:110-140`
    No per-turn/per-session tool-call caps beyond `max_steps`; `findings/experiments/plan` unbounded; `_trim_context(head=2)` assumes first two messages are a safe pair. Add `tool_call_count` caps, cap list lengths, fix trim to start from a message boundary with no open `tool_calls`.

25. **Silent truncations everywhere** — `datalab/tools/state_tools.py:46-47`, `datalab/tools/profile.py:227-250`, `app.py:282-284`
    600-char finding cut, 4000-char/40-col report cut, `_preview` ellipsis — none disclosed. State "showing X of Y" in output so model/user knows data is hidden.

26. **Executor output unbounded; >8 figures silently dropped** — `datalab/tools/python_exec.py:215-242`
    `_compose` has no total cap (relies on agent-level truncate); figures beyond 8 dropped without telling the model. Cap in executor, mention dropped figures in text.

27. **Trailing-expression `eval` runs side effects** — `datalab/tools/python_exec.py:89,193`
    `__import__('os').system(...)` as a bare last line executes as "value". Restrict trailing eval to safe display or run it under the same approval policy with a warning.

28. **`base.py` direct indexing** — `datalab/tools/base.py:52-66`
    `paths["root"]` etc. raise `KeyError` if misconfigured. Validate keys exist at `ToolContext` construction.

29. **Empty message silently dropped** — `app.py:138-141`
    Empty text + no uploads returns with no feedback. Send "I didn't get any text or file" message.

## P3 — Tests and ops

30. **Zero coverage for `app.py`** — no `tests/test_app*.py`
    All of `on_chat_start/on_message/_render_events/_open_step/_close_step/_render_results/_publish_plan/_approval_prompt/_ask_prompt` untested. Add `tests/test_app.py`: stream merge, missing-id step, bad `est`, outside-session image rejected, plan cap, approval timeout-vs-deny, upload validation.

31. **Untested handler paths** — `files.py`, `profile.py`, `python_exec.py`, `ask_user.py`, `state_tools.py`, `todo.py`
    Missing: `max_chars="bad"`, binary read, `stat` race, `write_file ../state.json` (should fail), absolute-outside-session profile, non-CSV readers, `>40 cols` truncation, `>8` figures, huge stdout, `await` cell, `os._exit`/network exfil, callback-raising `ask_user`, `query_state(all)` with huge state, todo duplicate-ids/1000-items.

32. **Brittle/weak assertions to fix**
    `tests/test_agent.py:324` exact `n_rows==603`; `:326-331` exact experiment-name list; `tests/test_rehearsal.py:44-46` hardcoded `f1_macro`; `tests/rehearse.py:385-393` overwrite check only `len(events)>0`; module-scoped `messy_result` (`test_rehearsal.py:26-28`) leaks temp dirs on failure.

33. **Docs/config nits**
    `pyproject.toml:5` uses `NOTES.md` as package readme (should be `README.md`); `chainlit.md` vs `README.md` onboarding drift; `NOTES.md` "Later" list already acknowledges sandbox/concurrency gaps — link each P0 above to that list so demo users see the risk.
