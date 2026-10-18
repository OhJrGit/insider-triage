"""Unsupervised anomaly detectors over the `features` table.

Models:
  iforest      Isolation Forest (baseline)
  ocsvm        One-Class SVM (baseline; trained on a subsample, it is O(n^2))
  autoencoder  small sklearn MLP autoencoder -- PLACEHOLDER. Replace with the team's
               PyTorch autoencoder / Transformer; just return one score per session.

Split is by time (train on the earlier part, score the later part) so nothing from the
future leaks into training. Labels are only used for evaluation, never for training,
unless --train-benign-only is set (then it is semi-supervised; say so in the report).

Writes: scores(session_id, model, score, rank), alerts(top-k of the primary model),
        work/detector_metrics.json
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from . import config
from .features import COUNT_FEATURES, MODEL_FEATURES


@dataclass
class DetectConfig:
    train_frac: float = 0.5
    models: tuple[str, ...] = ("iforest", "ocsvm", "autoencoder")   # + "torch_ae", "transformer" (PyTorch)
    primary: str = "iforest"
    top_k: int = 200
    train_benign_only: bool = False
    max_train_ocsvm: int = 20_000
    max_train_ae: int = 100_000
    seed: int = 42
    overwrite_alerts: bool = False
    # Per-model hyperparameter overrides, e.g. {"ocsvm": {"nu": 0.005}}. Empty = the original defaults.
    # Produced by `python cli.py tune` (work/best_params.json) and loaded with --params.
    params: dict = field(default_factory=dict)


def _matrix(df: pd.DataFrame) -> np.ndarray:
    X = df[MODEL_FEATURES].astype(float).copy()
    n_nan = int(X.isna().sum().sum())
    if n_nan:
        print(f"  [warn] {n_nan} missing feature values filled with 0 -- check features.py")
        X = X.fillna(0.0)
    for c in COUNT_FEATURES:
        X[c] = np.log1p(X[c].clip(lower=0))
    return X.to_numpy()


# Defaults reproduce the original hard-coded settings exactly (so results without --params are unchanged).
DEFAULT_PARAMS = {
    "iforest": {"n_estimators": 300, "max_samples": "auto", "max_features": 1.0},
    "ocsvm": {"nu": 0.01, "gamma_mult": 1.0},          # gamma = gamma_mult * gamma("scale")
    "autoencoder": {"bottleneck": None, "alpha": 1e-4},  # bottleneck None -> max(d // 4, 2)
}


def _fit_score(name: str, Xtr: np.ndarray, Xte: np.ndarray, cfg: DetectConfig, ctx: dict | None = None) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    p = {**DEFAULT_PARAMS.get(name, {}), **cfg.params.get(name, {})}
    if name == "iforest":
        m = IsolationForest(n_estimators=int(p["n_estimators"]), max_samples=p["max_samples"],
                            max_features=float(p["max_features"]),
                            random_state=cfg.seed, n_jobs=-1).fit(Xtr)
        return -m.score_samples(Xte)
    if name == "ocsvm":
        idx = rng.choice(len(Xtr), min(len(Xtr), cfg.max_train_ocsvm), replace=False)
        Xs = Xtr[idx]
        gamma = "scale" if p["gamma_mult"] == 1.0 else float(p["gamma_mult"]) / (Xs.shape[1] * Xs.var())
        m = OneClassSVM(nu=float(p["nu"]), gamma=gamma).fit(Xs)
        return -m.decision_function(Xte)
    if name == "autoencoder":
        idx = rng.choice(len(Xtr), min(len(Xtr), cfg.max_train_ae), replace=False)
        d = Xtr.shape[1]
        b = int(p["bottleneck"]) if p["bottleneck"] else max(d // 4, 2)
        mid = max(d // 2, b, 4)
        warnings.filterwarnings("ignore", message=".*Maximum iterations.*")
        m = MLPRegressor(hidden_layer_sizes=(d, mid, b, mid, d), alpha=float(p["alpha"]),
                         max_iter=200, early_stopping=True, random_state=cfg.seed).fit(Xtr[idx], Xtr[idx])
        return ((m.predict(Xte) - Xte) ** 2).mean(axis=1)
    if name == "torch_ae":
        from .deep import torch_ae_scores
        return torch_ae_scores(Xtr, Xte, seed=cfg.seed)
    if name == "transformer":
        from .deep import load_sequences, transformer_scores
        if ctx is None:
            raise ValueError("transformer needs the database context")
        if "seqs" not in ctx:
            print("  building event sequences for the transformer (one-off)...")
            ctx["seqs"] = load_sequences(ctx["con"])
        return transformer_scores(ctx["seqs"], ctx["train_ids"], ctx["test_ids"], seed=cfg.seed)
    raise ValueError(f"unknown model {name}")


def _metrics(test: pd.DataFrame, score: np.ndarray, ks: list[int]) -> dict:
    y = test["label"].to_numpy()
    out: dict = {"n_test": int(len(y)), "n_pos": int(y.sum())}
    if y.sum() == 0 or y.sum() == len(y):
        return out
    out["roc_auc"] = float(roc_auc_score(y, score))
    out["pr_auc"] = float(average_precision_score(y, score))
    order = np.argsort(-score)
    insider_users = set(test.loc[test.label == 1, "user"])
    users = test["user"].to_numpy()
    for k in ks:
        top = order[:k]
        hits = int(y[top].sum())
        out[f"precision@{k}"] = hits / k
        out[f"recall@{k}"] = hits / int(y.sum())
        # STRICT (same definition as precision@k): insiders with at least one *malicious session* in the top k.
        out[f"insiders_caught@{k}"] = f"{len(set(users[top][y[top] == 1]))}/{len(insider_users)}"
        # LOOSE (what insiders_caught@k used to report): insiders with *any* session in the top k, malicious or
        # not. A benign session by an insider counted as a catch, so this can be > 0 while precision@k is 0.
        out[f"insider_users_in_top@{k}"] = f"{len(insider_users & set(users[top]))}/{len(insider_users)}"
    return out


def run(con, cfg: DetectConfig = DetectConfig()) -> dict:
    from .db import table_exists
    reports = [p for p in config.REPORTS_DIR.glob("*.json")] if config.REPORTS_DIR.exists() else []
    if table_exists(con, "alerts") and reports and not cfg.overwrite_alerts:
        raise SystemExit(
            "  Refusing to rebuild the alerts table: work/reports already holds reports for the current alerts,\n"
            "  and new alerts would no longer match them. Use `detect-seeds` to evaluate models (read-only),\n"
            "  or pass --overwrite-alerts if you really want new alerts.")
    df = con.execute("SELECT * FROM features ORDER BY start_ts").df()
    cut = df["start_ts"].quantile(cfg.train_frac)
    train, test = df[df.start_ts < cut].copy(), df[df.start_ts >= cut].copy()
    has_labels = df["label"].notna().all()
    if cfg.train_benign_only and has_labels:
        train = train[train.label == 0]
    scaler = StandardScaler().fit(_matrix(train))
    Xtr, Xte = scaler.transform(_matrix(train)), scaler.transform(_matrix(test))
    print(f"  train {len(train):,} sessions (< {cut}), test {len(test):,} sessions")

    ks = sorted({k for k in (10, 50, 100, cfg.top_k) if k <= len(test)})
    all_scores, metrics = [], {"split_time": str(cut), "features": MODEL_FEATURES, "models": {}}
    ctx = {"con": con, "train_ids": train.session_id.tolist(), "test_ids": test.session_id.tolist()}
    for name in cfg.models:
        s = _fit_score(name, Xtr, Xte, cfg, ctx)
        frame = pd.DataFrame({"session_id": test.session_id.values, "model": name, "score": s})
        frame["rank"] = frame.score.rank(ascending=False, method="first").astype(int)
        all_scores.append(frame)
        if has_labels:
            metrics["models"][name] = _metrics(test.reset_index(drop=True), s, ks)
            m = metrics["models"][name]
            print(f"  {name:<12} ROC-AUC {m.get('roc_auc', float('nan')):.3f}  "
                  f"PR-AUC {m.get('pr_auc', float('nan')):.3f}  "
                  f"P@{ks[-1]} {m.get(f'precision@{ks[-1]}', float('nan')):.3f}  "
                  f"insiders@{ks[-1]} {m.get(f'insiders_caught@{ks[-1]}', '-')}")
        else:
            print(f"  {name:<12} scored (no labels loaded, skipping metrics)")

    scores = pd.concat(all_scores, ignore_index=True)
    con.register("scores_df", scores)
    con.execute("CREATE OR REPLACE TABLE scores AS SELECT * FROM scores_df")
    con.unregister("scores_df")
    con.execute(f"""
        CREATE OR REPLACE TABLE alerts AS
        SELECT s.session_id, s.model, s.score, s.rank, f."user", f.pc, f.start_ts, f.end_ts, f.label, f.scenario
        FROM scores s JOIN features f USING (session_id)
        WHERE s.model = '{cfg.primary}' AND s.rank <= {int(cfg.top_k)}
        ORDER BY s.rank
    """)
    config.ensure_dirs()
    (config.WORK_DIR / "detector_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    return metrics


def _summarise(runs: list[dict]) -> dict:
    summ = {}
    for key in runs[0]:
        vals = [r[key] for r in runs]
        if isinstance(vals[0], float):
            summ[key] = {"mean": round(float(np.mean(vals)), 4),
                         "std": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0.0}
        elif isinstance(vals[0], str) and "/" in vals[0]:
            caught = [int(v.split("/")[0]) for v in vals]
            summ[key] = {"mean": round(float(np.mean(caught)), 1), "min": min(caught), "max": max(caught),
                         "of": vals[0].split("/")[1]}
    return summ


def run_seeds(con, cfg: DetectConfig, seeds: list[int]) -> dict:
    """Repeat training/scoring with several random seeds and report mean / std per metric.
    Read-only: does NOT touch the scores/alerts tables, so the alerts the agent was evaluated
    on stay exactly as they are."""
    from dataclasses import replace
    df = con.execute("SELECT * FROM features ORDER BY start_ts").df()
    cut = df["start_ts"].quantile(cfg.train_frac)
    train, test = df[df.start_ts < cut].copy(), df[df.start_ts >= cut].copy()
    if cfg.train_benign_only:
        train = train[train.label == 0]
    scaler = StandardScaler().fit(_matrix(train))
    Xtr, Xte = scaler.transform(_matrix(train)), scaler.transform(_matrix(test))
    test = test.reset_index(drop=True)
    ks = sorted({k for k in (10, 50, 100, cfg.top_k) if k <= len(test)})
    per = {m: [] for m in cfg.models}
    ctx = {"con": con, "train_ids": train.session_id.tolist(), "test_ids": test.session_id.tolist()}
    path = config.WORK_DIR / "detector_seeds.json"
    config.ensure_dirs()
    out = {"seeds": seeds, "split_time": str(cut), "n_test": len(test), "models": {}}
    if path.exists():                                   # merge, so running a subset of models
        try:                                            # never erases the others' results
            old = json.loads(path.read_text())
            if old.get("split_time") == out["split_time"] and old.get("n_test") == out["n_test"]:
                out["models"] = old.get("models", {})
        except (ValueError, OSError):
            pass
    for sd in seeds:
        c = replace(cfg, seed=sd)
        for name in cfg.models:
            per[name].append(_metrics(test, _fit_score(name, Xtr, Xte, c, ctx), ks))
            m = per[name][-1]
            print(f"  seed {sd} {name:<12} ROC {m['roc_auc']:.3f} PR {m['pr_auc']:.3f} "
                  f"P@{ks[-1]} {m[f'precision@{ks[-1]}']:.3f} insiders {m[f'insiders_caught@{ks[-1]}']}",
                  flush=True)
            out["models"][name] = _summarise(per[name])
            out["models"][name]["seeds_done"] = len(per[name])
            path.write_text(json.dumps(out, indent=2))   # saved after every run
    return out
