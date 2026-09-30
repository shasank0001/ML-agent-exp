# DataLab Agent

A **chat-based ML co-pilot**. Upload a dataset, say *"build the best model to predict
`churned`"*, and the agent profiles the data, writes a visible plan, runs a baseline,
trains and compares models, and reports a results table it can defend.

Every action is visible in the chat: the code it wrote, what it printed, the plots, the
to-do list. **Metrics are computed by the harness, not by the model** — the LLM writes
model code, but the split, the evaluation and the research state belong to
`datalab/`. That is the one idea the whole design is built on.

## Quick start

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

cp .env.example .env      # then set LMSTUDIO_MODEL or OPENROUTER_MODEL
uv run chainlit run app.py --host 127.0.0.1
```

Open http://127.0.0.1:8000, upload a CSV, and ask for a model.

**Use a model that supports function calling.** The app probes the configured model on
startup and warns you in the chat if it cannot call tools.

### Providers

Both speak the OpenAI chat-completions API; only the env vars change.

| | LM Studio (local) | OpenRouter (hosted) |
|---|---|---|
| `LLM_PROVIDER` | `lmstudio` | `openrouter` |
| base URL | `LMSTUDIO_BASE_URL` | `OPENROUTER_BASE_URL` |
| key | `LMSTUDIO_API_KEY` (any string) | `OPENROUTER_API_KEY` |
| model | `LMSTUDIO_MODEL` | `OPENROUTER_MODEL` |

## Layout

```
app.py                  Chainlit: renders the event stream, handles uploads and buttons
datalab/
  agent.py              the loop — UI-agnostic, an async generator of events
  llm.py                OpenAI-compatible streaming client + tool-call self-test
  events.py             Event model + JSONL log
  state.py              ResearchState: dataset, task, plan, experiments, findings
  lab.py                ground truth: split, baseline, evaluate, results table
  approvals.py          the Approve/Deny policy
  prompts.py            system prompt
  config.py             env loading, provider selection, session paths
  tools/                python executor, profiler, files, todo, state tools, ask_user
runs/<session_id>/      data/ outputs/ figures/ state.json events.jsonl
tests/                  pytest, incl. a scripted LLM that rehearses the golden path
```

## Design notes

- **No agent framework.** Plain `async`/`await` and the OpenAI SDK. The loop is ~200
  readable lines in `datalab/agent.py`.
- **The agent runtime never imports Chainlit.** It yields events; `app.py` renders
  them. That is what makes the event log usable as a replay source and as evaluation
  material.
- **Notebook semantics.** One persistent namespace per session; a trailing expression
  is evaluated and shown; matplotlib figures are saved and closed automatically.
- **Everything risky asks first.** Long cells, destructive code, writes outside the
  session folder and big searches raise an Approve/Deny question that shows the code.

## Tests

```bash
pytest -q
```

`tests/fake_llm.py` contains a `ScriptedLLM` (replays fixed turns) and a `PolicyLLM`
(a small rule-based agent) so the whole golden path is rehearsed offline, with no
provider and no network.

## Safety

LLM-generated code runs **in the app process with no sandbox**, on localhost only. Run
it on a machine and in folders you are happy for the agent to touch. Chainlit's MCP
feature is explicitly disabled in `.chainlit/config.toml`. See `NOTES.md` for the full
list of assumptions and known limitations.
