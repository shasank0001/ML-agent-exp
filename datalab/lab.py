"""Ground-truth split + metrics, injected into agent ``python`` cells as ``lab``."""
from __future__ import annotations
import math, uuid, warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from .state import Experiment, ResearchState, TaskSpec, TaskType

__all__ = ["LabSession", "infer_task_type", "default_primary_metric", "is_higher_better", "compute_metrics"]
_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
_LOW_CARD = 50
_LOWER_IS_BETTER = {"rmse", "mae", "mse", "log_loss", "mape", "rmsle"}
_KNOWN_METRICS = ("accuracy", "f1_macro", "roc_auc", "rmse", "mae", "r2")
_TASKS = ("classification", "regression")

def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
def infer_task_type(series: pd.Series) -> TaskType:
    """Object/category/bool, or low-cardinality integer-valued, is classification."""
    dt = series.dtype
    if (pd.api.types.is_bool_dtype(dt) or isinstance(dt, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(dt) or pd.api.types.is_string_dtype(dt)):
        return "classification"
    if pd.api.types.is_datetime64_any_dtype(dt) or not pd.api.types.is_numeric_dtype(dt):
        return "classification" if series.nunique(dropna=True) <= 20 else "regression"
    vals = series.dropna()
    if vals.empty or vals.nunique() > 20:
        return "regression"
    if pd.api.types.is_integer_dtype(dt):
        return "classification"
    try:  # low-cardinality float flags (0/1/2 stored as float) count as classes
        arr = vals.to_numpy(dtype=float)
        return "classification" if bool(np.all(np.isclose(arr, np.round(arr), atol=1e-9))) else "regression"
    except (TypeError, ValueError):
        return "regression"
def default_primary_metric(task_type: str) -> str:
    """f1_macro for classification, rmse for regression."""
    if task_type == "classification":
        return "f1_macro"
    if task_type == "regression":
        return "rmse"
    raise ValueError(f"unknown task_type {task_type!r}")
def is_higher_better(metric: str) -> bool:
    """True unless ``metric`` is an error measure (rmse/mae/...)."""
    return str(metric).lower() not in _LOWER_IS_BETTER
def _roc_auc(y_true: Any, y_proba: Any) -> float:
    proba = np.asarray(y_proba, dtype=float)
    n_labels = pd.Series(np.asarray(y_true)).nunique(dropna=True)
    if n_labels < 2:
        raise ValueError("roc_auc needs 2+ classes")
    binary = proba.ndim == 1 or (proba.ndim == 2 and proba.shape[1] == 2)
    if binary and n_labels == 2:
        return float(roc_auc_score(y_true, proba if proba.ndim == 1 else proba[:, 1]))
    if proba.ndim == 2 and proba.shape[1] > 2:
        return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
    raise ValueError(f"cannot compute roc_auc from shape {proba.shape}")
def compute_metrics(y_true: Any, y_pred: Any, *, task_type: str, y_proba: Any | None = None, classes: Any | None = None) -> dict[str, float]:
    """Metrics for ``task_type``; uncomputable ones are omitted, never raised."""
    _ = classes  # compatibility only; predict_proba columns already align with classes_
    out: dict[str, float] = {}
    def _try(key: str, fn: Any) -> None:
        try: out[key] = float(fn())
        except Exception: pass  # noqa: BLE001 - degenerate y must not crash evaluate
    fns = {"classification": (("accuracy", lambda: accuracy_score(y_true, y_pred)),
            ("f1_macro", lambda: f1_score(y_true, y_pred, average="macro", zero_division=0))),
           "regression": (("rmse", lambda: np.sqrt(mean_squared_error(y_true, y_pred))),
            ("mae", lambda: mean_absolute_error(y_true, y_pred)),
            ("r2", lambda: r2_score(y_true, y_pred)))}
    if task_type not in fns:
        raise ValueError(f"unknown task_type {task_type!r}")
    for key, fn in fns[task_type]:
        _try(key, fn)
    if task_type == "classification" and y_proba is not None:
        _try("roc_auc", lambda: _roc_auc(y_true, y_proba))
    return {k: v for k, v in out.items() if math.isfinite(v)}
def _looks_like_datetime(series: pd.Series) -> bool:
    """True when an object/string column mostly parses as datetimes."""
    sample = series.dropna()
    if sample.empty or pd.api.types.is_numeric_dtype(sample.dtype):
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if float(pd.to_numeric(sample, errors="coerce").notna().mean()) >= 0.9:
                return False  # numeric-looking strings, not dates
            return bool(float(pd.to_datetime(sample.iloc[:500], errors="coerce").notna().mean()) >= 0.9)
    except (TypeError, ValueError):
        return False
class LabSession:
    """Per-session ground truth: owns the split, fits models, logs experiments."""
    def __init__(self, state: ResearchState, *, data_dir: Path, output_dir: Path) -> None:
        self.state, self.data_dir, self.output_dir = state, Path(data_dir), Path(output_dir)
        self.target: str | None = None
        self.task_type: TaskType | None = None
        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None
        self._preprocess: ColumnTransformer | None = None
        self._feature_names: list[str] = []
    @property
    def has_split(self) -> bool:
        """True once an encoded split is held in memory."""
        return self.X_train is not None and self.X_test is not None
    def reset_split(self) -> None:
        """Forget the cached split so the next `split()` call rebuilds it.

        Experiments already logged keep their numbers; only the cached frames
        and the recorded split config are cleared.
        """
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self._preprocess = None
        self._feature_names = []
        if self.state.task is not None:
            self.state.task.split = {}

    @property
    def preprocess(self) -> ColumnTransformer | None:
        """The fitted column transformer (None before the first split)."""
        return self._preprocess
    @property
    def feature_names(self) -> list[str]:
        """Post-encoding column names."""
        return list(self._feature_names)
    def load(self, path: str | Path) -> pd.DataFrame:
        """Read csv/tsv/parquet/xlsx; relative paths resolve against data_dir."""
        full = Path(path)
        full = full if full.is_absolute() else self.data_dir / full
        suffix = full.suffix.lower()
        if suffix in (".csv", ".tsv", ".tab"):
            return pd.read_csv(full, sep="\t" if suffix != ".csv" else ",")
        if suffix in (".parquet", ".pq"): return pd.read_parquet(full)
        if suffix in (".xlsx", ".xls"): return pd.read_excel(full)
        raise ValueError(f"lab.load: unsupported extension {full.suffix!r}")
    def _record_task(self, target: str, task_type: TaskType, primary_metric: str | None = None) -> TaskSpec:
        if task_type not in _TASKS:
            raise ValueError(f"unknown task_type {task_type!r}")
        cur = self.state.task
        split = dict(cur.split) if cur is not None and cur.target == target else {}
        if primary_metric is None:
            primary_metric = (cur.primary_metric if cur is not None and cur.target == target
                              and cur.primary_metric else default_primary_metric(task_type))
        self.state.task = task = TaskSpec(target=target, task_type=task_type, primary_metric=primary_metric, split=split)
        self.target, self.task_type = target, task_type
        return task
    def set_target(self, target: str, task_type: str | None = None, primary_metric: str | None = None) -> dict[str, Any]:
        """Record the task in state without splitting. Idempotent."""
        name = str(target).strip()
        if not name:
            raise ValueError("lab.set_target needs a non-empty target name")
        cur = self.state.task
        # No data here; split() re-infers from the series.
        resolved = task_type or (cur.task_type if cur is not None and cur.target == name else "classification")
        return self._record_task(name, resolved, primary_metric).model_dump()  # type: ignore[arg-type]
    @staticmethod
    def _column_groups(X: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
        numeric, dates, low, high = [], [], [], []
        for col in X.columns:
            s = X[col]
            if s.isna().all():
                continue  # all missing: unusable
            if pd.api.types.is_bool_dtype(s.dtype):
                low.append(col)
            elif pd.api.types.is_datetime64_any_dtype(s.dtype):
                dates.append(col)
            elif pd.api.types.is_numeric_dtype(s.dtype):
                if pd.api.types.is_complex_dtype(s.dtype):
                    continue  # no median to impute
                numeric.append(col)
            elif _looks_like_datetime(s):
                dates.append(col)
            else:
                try:
                    (low if s.nunique(dropna=True) <= _LOW_CARD else high).append(col)
                except (TypeError, ValueError):
                    continue  # dtype pandas cannot coerce: drop it
        return numeric, dates, low, high
    @staticmethod
    def _encode_frame(X: pd.DataFrame, *, numeric: list[str], dates: list[str]) -> pd.DataFrame:
        work = X.copy()
        for col in numeric:
            work[col] = pd.to_numeric(work[col], errors="coerce")
        for col in dates:  # datetimes become epoch seconds
            work[col] = (pd.to_datetime(work[col], errors="coerce", utc=True) - _EPOCH).dt.total_seconds().astype(float)
        return work
    def split(self, df: pd.DataFrame, target: str, task_type: str | None = None,
              test_size: float = 0.2, seed: int = 42, force: bool = False
              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Encode features and create (or reuse) the train/test split for ``target``.

        Re-splitting the same target with different features, ``test_size`` or
        ``seed`` replaces the cached split automatically. Pass ``force=True``
        to rebuild it even when nothing changed -- e.g. after cleaning the
        frame in place.
        """
        if target not in df.columns:
            raise ValueError(f"lab.split: target {target!r} not in columns")
        resolved: TaskType = task_type or infer_task_type(df[target])  # type: ignore[assignment]
        if resolved not in _TASKS:
            raise ValueError(f"unknown task_type {task_type!r}")
        cur = self.state.task
        if (not force and self.X_train is not None and self.y_train is not None
                and cur is not None and cur.target == target
                and cur.task_type == resolved and cur.split.get("seed") == seed
                and cur.split.get("test_size") == test_size):
            assert self.X_test is not None and self.y_test is not None
            return self.X_train, self.X_test, self.y_train, self.y_test
        keep = df[target].notna()  # rows with a missing target can neither train nor score
        X_raw = df.drop(columns=[target]).loc[keep].reset_index(drop=True)
        y = df[target].loc[keep].reset_index(drop=True)
        if len(y) < 2:
            raise ValueError("lab.split: need 2+ rows with a non-missing target")
        stratify_labels = None
        if resolved == "classification":
            counts = y.value_counts(dropna=False)
            if len(counts) >= 2 and bool((counts >= 2).all()):
                stratify_labels = y
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y, test_size=test_size, random_state=seed, stratify=stratify_labels)
            stratified = stratify_labels is not None
        except ValueError:
            if stratify_labels is None:
                raise
            X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y, test_size=test_size, random_state=seed)
            stratified = False
        numeric, dates, low, high = self._column_groups(X_tr)
        if not (numeric or dates or low or high):
            raise ValueError("lab.split: no usable feature columns")
        transformers: list[Any] = []
        if numeric or dates:
            transformers.append(("num", SimpleImputer(strategy="median"), numeric + dates))
        low_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        if low: transformers.append(("cat_low", low_pipe, low))
        if high: transformers.append(("cat_high", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), high))
        preprocess = ColumnTransformer(transformers, remainder="drop")
        Xtr_p = self._encode_frame(X_tr, numeric=numeric, dates=dates)
        Xte_p = self._encode_frame(X_te, numeric=numeric, dates=dates)
        try:
            Xtr_enc, Xte_enc = preprocess.fit_transform(Xtr_p), preprocess.transform(Xte_p)
        except Exception:  # noqa: BLE001 - numeric-only fallback for hostile dtypes
            preprocess = ColumnTransformer([("num", SimpleImputer(strategy="median"), numeric + dates)], remainder="drop")
            Xtr_enc, Xte_enc = preprocess.fit_transform(Xtr_p), preprocess.transform(Xte_p)
        try:
            names = [str(n) for n in preprocess.get_feature_names_out()]
        except Exception:  # noqa: BLE001 - generic names beat a crash
            names = [f"f{i}" for i in range(np.asarray(Xtr_enc).shape[1])]
        self.X_train = pd.DataFrame(np.asarray(Xtr_enc, dtype=float), columns=names)
        self.X_test = pd.DataFrame(np.asarray(Xte_enc, dtype=float), columns=names)
        self.y_train = pd.Series(np.asarray(y_tr), name=target)
        self.y_test = pd.Series(np.asarray(y_te), name=target)
        self._preprocess, self._feature_names = preprocess, names
        self._record_task(target, resolved).split = {"seed": seed, "test_size": test_size, "stratified": stratified,
            "n_train": len(self.X_train), "n_test": len(self.X_test), "n_features": len(names),
            "target": target, "task_type": resolved, "created_at": _now()}
        print(f"split ready: {resolved} on {target!r}, {len(self.X_train)} train / {len(self.X_test)} test, "
              f"{len(names)} features, seed={seed}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    def _require_split(self) -> TaskSpec:
        task = self.state.task
        if (task is None or not task.split or self.X_train is None or self.X_test is None
                or self.y_train is None or self.y_test is None):
            raise RuntimeError("no split yet — call lab.split(...) first")
        if task.task_type not in _TASKS:
            raise ValueError(f"unknown task_type {task.task_type!r}")
        return task
    def evaluate(self, model: Any, name: str, params: dict[str, Any] | None = None, notes: str = "") -> dict[str, float]:
        """Fit (unless already fitted), score on X_test, log an Experiment."""
        task = self._require_split()
        assert self.X_train is not None and self.y_train is not None and self.X_test is not None and self.y_test is not None
        try:
            model.predict(self.X_train.iloc[:1])  # already fitted: reuse as-is
        except (NotFittedError, AttributeError):
            model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        try:
            y_proba = model.predict_proba(self.X_test) if hasattr(model, "predict_proba") else None
        except Exception:  # noqa: BLE001 - score without probabilities instead
            y_proba = None
        metrics = compute_metrics(self.y_test, y_pred, task_type=task.task_type,
                                  y_proba=y_proba, classes=getattr(model, "classes_", None))
        model_name = ("+".join(type(s).__name__ for _, s in model.steps) + "|Pipeline"
                      if isinstance(model, Pipeline) else type(model).__name__)
        self._log(Experiment(id=uuid.uuid4().hex[:8], name=name, model=model_name,
            params=dict(params or {}), metrics=metrics, primary_metric=task.primary_metric,
            status="done", notes=notes or ""))
        return dict(metrics)

    def _log(self, exp: Experiment) -> None:
        """Record an experiment, replacing an earlier run of the same name.

        Re-splitting and re-running the baselines is normal, and a results table
        that lists `baseline_logreg` twice with different numbers invites the
        reader to compare a run against itself.
        """
        for i, existing in enumerate(self.state.experiments):
            if existing.name == exp.name:
                self.state.experiments[i] = exp
                return
        self.state.add_experiment(exp)
    def baseline(self) -> dict[str, dict[str, float]]:
        """Log a dummy and a linear baseline; returns {name: metrics}."""
        task = self._require_split()
        if task.task_type == "classification":
            specs = (("baseline_dummy", DummyClassifier(strategy="most_frequent")), ("baseline_logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))))
        else:
            specs = (("baseline_dummy", DummyRegressor(strategy="mean")), ("baseline_ridge", make_pipeline(StandardScaler(), Ridge())))
        return {n: self.evaluate(m, n, notes="baseline") for n, m in specs}
    def _ranked(self) -> list[Experiment]:
        primary = self.state.task.primary_metric if self.state.task else ""
        higher = is_higher_better(primary)
        def key(exp: Experiment) -> tuple[int, float]:
            try:
                v = float(exp.metrics.get(primary)) if primary else None
            except (TypeError, ValueError):
                v = None
            return (1, 0.0) if v is None or not math.isfinite(v) else (0, -v if higher else v)
        return sorted(self.state.experiments, key=key)
    def results_table(self) -> pd.DataFrame:
        """Experiments as a DataFrame, best-first for the primary metric."""
        primary = self.state.task.primary_metric if self.state.task else ""
        others = sorted({k for e in self.state.experiments for k in e.metrics} - {primary})
        columns = ["name", "model"] + ([primary] if primary else []) + others + ["notes", "created_at"]
        rows = [{"name": e.name, "model": e.model, **({primary: e.metrics.get(primary)} if primary else {}),
                 **{m: e.metrics.get(m) for m in others}, "notes": e.notes, "created_at": e.created_at} for e in self._ranked()]
        return pd.DataFrame(rows, columns=columns)
    def experiment(self, name: str) -> dict[str, Any]:
        """One experiment as a dict; raises KeyError when missing."""
        for exp in self.state.experiments:
            if exp.name == name:
                return exp.model_dump()
        raise KeyError(f"no experiment named {name!r}")
    def set_primary_metric(self, metric: str) -> str:
        """Switch the primary metric for the task and all logged experiments."""
        name = str(metric).strip()
        if name not in _KNOWN_METRICS:
            raise ValueError(f"unknown metric {metric!r}; expected one of {_KNOWN_METRICS}")
        if self.state.task is None:
            raise RuntimeError("no task yet — call lab.split(...) first")
        self.state.task.primary_metric = name
        for exp in self.state.experiments: exp.primary_metric = name
        return name
    def results_markdown(self) -> str:
        """The results table as a GitHub-flavoured markdown string."""
        table = self.results_table()
        if table.empty:
            return "No experiments logged yet."
        cols = [str(c) for c in table.columns]
        def fmt(v: Any) -> str:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return ""
            return f"{v:.4f}" if isinstance(v, float) else str(v)
        header = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
        return "\n".join(header + ["| " + " | ".join(fmt(v) for v in row) + " |" for _, row in table.iterrows()])
