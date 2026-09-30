"""Shared fixtures: a temporary runs/ dir, session scaffolding and sample datasets."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datalab.config import Settings


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    d.mkdir()
    return d


@pytest.fixture
def settings(runs_dir: Path) -> Settings:
    return Settings(
        provider="lmstudio",
        base_url="http://localhost:1234/v1",
        api_key="test-key",
        model="test-model",
        max_steps=12,
        max_repairs=2,
        approval_seconds_threshold=60,
        python_soft_timeout_s=60,
        runs_dir=runs_dir,
    )


@pytest.fixture
def messy_csv(tmp_path: Path) -> Path:
    """600 rows, mixed dtypes, NaNs, a near-unique id, imbalance, 3 duplicates."""
    rng = np.random.default_rng(7)
    n = 600
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:04d}" for i in range(n)],
            "age": rng.integers(18, 80, n).astype(float),
            "plan": rng.choice(["basic", "plus", "pro"], n, p=[0.5, 0.3, 0.2]),
            "spend": rng.gamma(2.0, 30.0, n).round(2),
            "signup_month": rng.integers(1, 13, n),
            "is_active": rng.choice([True, False], n, p=[0.7, 0.3]),
            "churned": rng.choice([0, 1], n, p=[0.8, 0.2]),
        }
    )
    df.loc[rng.choice(n, 40, replace=False), "age"] = np.nan
    df.loc[rng.choice(n, 25, replace=False), "spend"] = np.nan
    df.loc[df["spend"] > df["spend"].quantile(0.995), "spend"] *= 12  # outliers
    df = pd.concat([df, df.iloc[:3]], ignore_index=True)  # duplicate rows
    path = tmp_path / "customers.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def regression_csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(11)
    n = 400
    x = rng.normal(size=n)
    noise = rng.normal(scale=0.4, size=n)
    df = pd.DataFrame(
        {
            "area_sqft": rng.integers(600, 4000, n),
            "rooms": rng.integers(1, 7, n),
            "age_years": rng.integers(0, 90, n),
            "garage": rng.choice(["none", "one", "two"], n),
            "price_k": (40 + 0.06 * x * 40 + noise).round(1),
        }
    )
    path = tmp_path / "houses.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture(autouse=True)
def _no_matplotlib_gui(monkeypatch: pytest.MonkeyPatch) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLBACKEND", "Agg")
