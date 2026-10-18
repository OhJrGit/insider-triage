"""Hyperparameter tuning for the sklearn detectors on a validation slice (no test-label leakage).

Time-ordered three-way split (same test set as `detect`, so numbers stay comparable):

    |-------- train --------|--- val ---|------------- test -------------|
    0                  test_cut-val_frac  test_cut                       1      (quantiles of start_ts)

  1. For every model and every grid point: fit on `train`, score `val`, record the validation metrics.
  2. Pick the grid point with the best mean validation metric (default PR-AUC) over the tuning seeds.
  3. Refit the chosen setting AND the original defaults on train+val, score `test` ONCE, and report both,
     so "tuned vs default" is a like-for-like comparison. The test labels never influence the choice.

Honest caveat for the report: the selection step uses validation LABELS (to rank configurations), so
tuned detectors are weakly supervised in their hyperparameters even though training is unsupervised.

Writes work/tuning.json (all grid results + final comparison) and work/best_params.json, which
`detect --params` / `detect-seeds --params` can load.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config
from .detect import DEFAULT_PARAMS, DetectConfig, _fit_score, _matrix, _metrics

GRIDS: dict[str, dict[str, list]] = {
    "iforest": {"n_estimators": [100, 300, 500],
                "max_samples": ["auto", 1024, 8192],
                "max_features": [0.5, 0.75, 1.0]},
    "ocsvm": {"nu": [0.001, 0.005, 0.01, 0.05],
              "gamma_mult": [0.25, 1.0, 4.0]},
    "autoencoder": {"bottleneck": [3, 6, 12],
                    "alpha": [1e-4, 1e-2]},
}
QUICK_GRIDS: dict[str, dict[str, list]] = {
    "iforest": {"n_estimators": [100, 300], "max_samples": ["auto", 1024], "max_features": [0.5, 1.0]},
    "ocsvm": {"nu": [0.005, 0.01], "gamma_mult": [0.25, 1.0]},
    "autoencoder": {"bottleneck": [3, 6], "alpha": [1e-4, 1e-2]},
}


def split_by_time(df: pd.DataFrame, train_frac: float, val_frac: float):
    """Returns train, val, test. `test` equals the test set of `detect` for the same train_frac."""
    if not 0 < val_frac < train_frac:
        raise ValueError("need 0 < val_frac < train_frac")
    test_cut = df["start_ts"].quantile(train_frac)
    val_cut = df["start_ts"].quantile(train_frac - val_frac)
    train = df[df.start_ts < val_cut]
    val = df[(df.start_ts >= val_cut) & (df.start_ts < test_cut)]
    test = df[df.start_ts >= test_cut]
    return train.copy(), val.copy(), test.copy(), (val_cut, test_cut)


def _grid_points(grid: dict[str, list]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def _scaled(train: pd.DataFrame, *others: pd.DataFrame):
    sc = StandardScaler().fit(_matrix(train))
    return [sc.transform(_matrix(d)) for d in (train, *others)]


def _score_metric(test_df: pd.DataFrame, score: np.ndarray, select_by: str, k: int) -> dict:
    m = _metrics(test_df.reset_index(drop=True), score, [k])
    if select_by not in m:
        raise SystemExit(f"  no '{select_by}' for this split: it has {m['n_pos']} positive sessions "
                         f"out of {m['n_test']}. Widen --val-frac or change --select-by.")
    return m


def run(con, cfg: DetectConfig, models: list[str], val_frac: float = 0.15, seeds: list[int] | None = None,
        select_by: str = "pr_auc", quick: bool = False) -> dict:
    seeds = seeds or [cfg.seed]
    grids = QUICK_GRIDS if quick else GRIDS
    unknown = [m for m in models if m not in grids]
    if unknown:
        raise SystemExit(f"  can't tune {unknown}: supported models are {sorted(grids)} "
                         f"(torch_ae / transformer have fixed architectures in deep.py)")

    df = con.execute("SELECT * FROM features ORDER BY start_ts").df()
    if not df["label"].notna().all():
        raise SystemExit("  features has no labels; tuning needs the answers loaded (see `ingest --answers-dir`).")
    train, val, test, (val_cut, test_cut) = split_by_time(df, cfg.train_frac, val_frac)
    fit_part = train[train.label == 0] if cfg.train_benign_only else train
    print(f"  train {len(fit_part):,} (< {val_cut}), val {len(val):,} ({int(val.label.sum())} malicious), "
          f"test {len(test):,} ({int(test.label.sum())} malicious, from {test_cut})", flush=True)
    k = min(100, len(val))
    sel_key = select_by                      # "pr_auc", "roc_auc" or "precision@100"

    Xtr, Xva = _scaled(fit_part, val)
    out: dict = {"split": {"val_cut": str(val_cut), "test_cut": str(test_cut), "val_frac": val_frac,
                           "n_train": len(fit_part), "n_val": len(val), "n_test": len(test),
                           "val_malicious": int(val.label.sum()), "test_malicious": int(test.label.sum())},
                 "select_by": sel_key, "tune_seeds": seeds, "quick": quick, "models": {}}
    best_params: dict[str, dict] = {}

    for name in models:
        points = _grid_points(grids[name])
        print(f"\n  [{name}] {len(points)} configurations x {len(seeds)} seed(s)", flush=True)
        rows = []
        for i, pt in enumerate(points, 1):
            vals = []
            for sd in seeds:
                c = replace(cfg, seed=sd, params={name: pt})
                m = _score_metric(val, _fit_score(name, Xtr, Xva, c), sel_key, k)
                vals.append(m)
            row = {"params": pt,
                   "val_pr_auc": float(np.mean([v["pr_auc"] for v in vals])),
                   "val_roc_auc": float(np.mean([v["roc_auc"] for v in vals])),
                   f"val_precision@{k}": float(np.mean([v[f"precision@{k}"] for v in vals])),
                   "val_selected": float(np.mean([v[sel_key] for v in vals]))}
            rows.append(row)
            print(f"    {i:>3}/{len(points)} {json.dumps(pt)}  val PR {row['val_pr_auc']:.4f}  "
                  f"ROC {row['val_roc_auc']:.3f}  P@{k} {row[f'val_precision@{k}']:.2f}", flush=True)
        rows.sort(key=lambda r: -r["val_selected"])
        best = rows[0]["params"]
        best_params[name] = best
        print(f"    best by val {sel_key}: {json.dumps(best)}", flush=True)
        out["models"][name] = {"grid_results": rows, "best_params": best}

    # ---- final: refit on train+val, evaluate on test once (tuned vs defaults) -------------------
    trval = pd.concat([train, val])
    fit_final = trval[trval.label == 0] if cfg.train_benign_only else trval
    Xf, Xte = _scaled(fit_final, test)
    ks = sorted({kk for kk in (10, 50, 100, cfg.top_k) if kk <= len(test)})
    print("\n  Test set (refit on train+val; evaluated once):", flush=True)
    for name in models:
        final = {}
        for tag, params in (("default", DEFAULT_PARAMS[name]), ("tuned", best_params[name])):
            c = replace(cfg, params={name: params})
            final[tag] = _metrics(test.reset_index(drop=True), _fit_score(name, Xf, Xte, c), ks)
            m = final[tag]
            print(f"    {name:<12} {tag:<8} ROC {m['roc_auc']:.3f}  PR {m['pr_auc']:.4f}  "
                  f"P@{ks[-1]} {m[f'precision@{ks[-1]}']:.3f}  insiders@{ks[-1]} {m[f'insiders_caught@{ks[-1]}']}",
                  flush=True)
        out["models"][name]["test"] = final

    config.ensure_dirs()
    (config.WORK_DIR / "tuning.json").write_text(json.dumps(out, indent=2, default=str))
    (config.WORK_DIR / "best_params.json").write_text(json.dumps(best_params, indent=2))
    return out
