"""Environment loading and provider selection.

One OpenAI-compatible client serves both providers; the difference is the
base URL, the key and the model id. See ``.env.example``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"

#: Secrets that must never reach the event log.
SECRET_ENV_KEYS = (
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "LMSTUDIO_API_KEY",
    "ANTHROPIC_API_KEY",
)

#: Session ids become directory names — allowlist to block traversal.


def _session_id_ok(session_id: str) -> bool:
    import re

    return bool(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", session_id))


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration resolved from the environment."""

    provider: str
    base_url: str
    api_key: str
    model: str
    max_steps: int
    max_repairs: int
    approval_seconds_threshold: int
    python_soft_timeout_s: int
    context_window_messages: int = 20
    max_tool_output_chars: int = 8_000
    max_state_summary_chars: int = 6_000
    llm_timeout_s: float = 600.0
    llm_max_retries: int = 1
    runs_dir: Path = field(default=RUNS_DIR)
    _secret_values: tuple[str, ...] = ()

    @property
    def secrets(self) -> tuple[str, ...]:
        """Secret literals that must never be written to the event log."""
        return self._secret_values

    @property
    def is_configured(self) -> bool:
        """True when a model id and a non-empty key are present."""
        return bool(self.model.strip()) and bool(self.api_key.strip())

    def missing_pieces(self) -> list[str]:
        """Human-readable list of what still needs to be set in ``.env``."""
        out: list[str] = []
        if not self.model.strip():
            var = "LMSTUDIO_MODEL" if self.provider == "lmstudio" else "OPENROUTER_MODEL"
            out.append(f"{var} is not set")
        if not self.api_key.strip():
            var = "LMSTUDIO_API_KEY" if self.provider == "lmstudio" else "OPENROUTER_API_KEY"
            out.append(f"{var} is not set")
        if self.provider not in ("lmstudio", "openrouter"):
            out.append(f"LLM_PROVIDER={self.provider!r} is not one of lmstudio|openrouter")
        return out

    def redact(self, text: str) -> str:
        """Replace any known secret literal found in ``text`` with a placeholder."""
        for secret in self._secret_values:
            cleaned = (secret or "").strip()
            if cleaned and len(cleaned) >= 4 and cleaned in text:
                text = text.replace(cleaned, "***redacted***")
        return text


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def load_settings(dotenv_path: Path | str | None = None, *, load_env: bool = True) -> Settings:
    """Read configuration from ``.env`` (if present) and the process environment.

    Existing environment variables always win over ``.env`` so the app can be
    driven from the shell during the demo.
    """
    if load_env:
        load_dotenv(dotenv_path or (ROOT / ".env"), override=False)

    provider = os.getenv("LLM_PROVIDER", "lmstudio").strip().lower() or "lmstudio"
    if provider == "lmstudio":
        base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1").strip()
        api_key = os.getenv("LMSTUDIO_API_KEY", "lm-studio").strip()
        model = os.getenv("LMSTUDIO_MODEL", "").strip()
    else:
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        model = os.getenv("OPENROUTER_MODEL", "").strip()

    secrets = tuple(
        v for v in (os.getenv(k, "").strip() for k in SECRET_ENV_KEYS) if len(v) >= 4
    )

    python_soft_timeout_s = _int_env("PYTHON_SOFT_TIMEOUT_S", 1200)
    approval_seconds_threshold = _int_env("APPROVAL_SECONDS_THRESHOLD", 180)
    if python_soft_timeout_s <= approval_seconds_threshold:
        # A training cell approved for N seconds must not be killed before N.
        python_soft_timeout_s = approval_seconds_threshold + 300

    return Settings(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model=model,
        max_steps=_int_env("MAX_STEPS", 60),
        max_repairs=_int_env("MAX_REPAIRS", 5),
        approval_seconds_threshold=approval_seconds_threshold,
        python_soft_timeout_s=python_soft_timeout_s,
        context_window_messages=_int_env("CONTEXT_WINDOW_MESSAGES", 20),
        max_tool_output_chars=_int_env("MAX_TOOL_OUTPUT_CHARS", 8_000),
        llm_timeout_s=_float_env("LLM_TIMEOUT_S", 600.0),
        llm_max_retries=_int_env("LLM_MAX_RETRIES", 1),
        runs_dir=Path(os.getenv("RUNS_DIR", str(RUNS_DIR))).expanduser(),
        _secret_values=secrets,
    )


def session_paths(runs_dir: Path, session_id: str) -> dict[str, Path]:
    """Return the canonical per-session directory layout under ``runs_dir``."""
    if not _session_id_ok(session_id):
        raise ValueError(
            f"invalid session_id {session_id!r}: expected 1-64 chars of [A-Za-z0-9_-]"
        )
    runs_resolved = Path(runs_dir).expanduser().resolve()
    root = (runs_resolved / session_id).resolve()
    if root != runs_resolved and root.parent != runs_resolved:
        raise ValueError(f"session_id {session_id!r} escapes runs_dir")
    return {
        "root": root,
        "data": root / "data",
        "outputs": root / "outputs",
        "figures": root / "figures",
        "state": root / "state.json",
        "events": root / "events.jsonl",
    }


def ensure_session_dirs(runs_dir: Path, session_id: str) -> dict[str, Path]:
    """Create the per-session directory layout and return its paths."""
    paths = session_paths(runs_dir, session_id)
    for key in ("root", "data", "outputs", "figures"):
        paths[key].mkdir(parents=True, exist_ok=True, mode=0o700)
    return paths
