"""Separate development study: never reads test measurements or changes frozen v1."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from triage.common import ARTIFACTS, SEED, now, save_json
from triage.data import load_data
from triage.model import FeatureFilter, metrics


class TrainingQuantileClip(BaseEstimator, TransformerMixin):
    """Bound model inputs using training quantiles; preserve original source values."""
    def __init__(self, lower=.01, upper=.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        self.bounds_ = np.quantile(np.asarray(X), [self.lower, self.upper], axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X), *self.bounds_)


def candidate_pipeline(params):
    steps = [("filter", FeatureFilter()),
             ("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if params["clip"]:
        steps.append(("clip", TrainingQuantileClip()))
    if params["features"] != "all":
        steps.append(("select", SelectKBest(f_classif, k=params["features"])))
    steps.append(("scale", RobustScaler() if params["robust"] else StandardScaler()))
    steps.append(("model", LogisticRegression(C=params["C"], class_weight="balanced",
                                                max_iter=4000, random_state=SEED)))
    return Pipeline(steps)


def improve():
    """Select by training-window AP only; validation is a previously observed dev set."""
    output = ARTIFACTS / "v2_development"
    if (output / "study.json").exists():
        raise RuntimeError("Development study already frozen; preserve its results.")
    x, original = load_data()
    manifest = pd.read_csv(ARTIFACTS / "split_manifest.csv", parse_dates=["timestamp"])
    # Deliberately subset before any experiment: test data are not passed to models.
    train_ids = manifest[(manifest.partition == "train") & manifest.eligible].sort_values(["timestamp", "record_id"]).index
    dev_ids = manifest[(manifest.partition == "validation") & manifest.eligible].sort_values(["timestamp", "record_id"]).index
    xt, yt = x.loc[train_ids], original.loc[train_ids, "failure"]
    xd, yd = x.loc[dev_ids], original.loc[dev_ids, "failure"]
    folds = []
    times = manifest.loc[train_ids, "timestamp"].reset_index(drop=True)
    for a, b in TimeSeriesSplit(n_splits=3).split(xt):
        a = a[times.iloc[a].to_numpy() < times.iloc[b[0]]]
        if len(a) and yt.iloc[a].nunique() == 2 and yt.iloc[b].nunique() == 2:
            folds.append((a, b))
    if len(folds) < 2:
        raise RuntimeError("Fewer than two valid temporal folds; study not selected.")
    # Predeclared bounded grid. No test metrics and no validation-driven selection.
    grid = [{"clip": clip, "robust": clip, "features": k, "C": c}
            for clip in (False, True) for k in (20, 50, 100, "all") for c in (.01, .1, 1.)]
    results = []
    for params in grid:
        fold_metrics = []
        for a, b in folds:
            estimator = candidate_pipeline(params).fit(xt.iloc[a], yt.iloc[a])
            frame = pd.DataFrame({"record_id": original.loc[train_ids[b], "record_id"].to_numpy(),
                                  "failure": yt.iloc[b].to_numpy(),
                                  "priority_score": estimator.predict_proba(xt.iloc[b])[:, 1]})
            fold_metrics.append(metrics(frame))
        results.append({"params": params, "folds": fold_metrics,
                        "mean_average_precision": float(np.mean([f["average_precision"] for f in fold_metrics]))})
        print(f"Candidate {len(results)}/{len(grid)} AP={results[-1]['mean_average_precision']:.4f}", flush=True)
    winner = max(results, key=lambda item: item["mean_average_precision"])
    estimator = candidate_pipeline(winner["params"]).fit(xt, yt)
    frame = pd.DataFrame({"record_id": original.loc[dev_ids, "record_id"].to_numpy(),
                          "failure": yd.to_numpy(), "priority_score": estimator.predict_proba(xd)[:, 1]})
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "development_predictions.csv", index=False)
    save_json(output / "study.json", {
        "at": now(), "seed": SEED, "selection_metric": "Mean training expanding-window average precision",
        "boundary": "Validation was previously inspected and is development-only. Test is not evaluated in this study; an independent future holdout is required.",
        "source_values": "Not deleted or modified. Quantile clipping is an optional training-fitted model transform.",
        "valid_folds": len(folds), "candidates": results, "selected": winner,
        "development_capacity_metrics": {str(c): metrics(frame, c) for c in (.1, .2, .3, .5, 1.)}})
    print("Selected:", winner["params"], "Development at 20%:", metrics(frame), flush=True)


def nonlinear_study():
    """Second exploratory stage, motivated by weak first-stage development capture."""
    output = ARTIFACTS / "v2_development"
    if (output / "nonlinear.json").exists():
        raise RuntimeError("Nonlinear study already frozen.")
    x, original = load_data()
    manifest = pd.read_csv(ARTIFACTS / "split_manifest.csv", parse_dates=["timestamp"])
    train_ids = manifest[(manifest.partition == "train") & manifest.eligible].sort_values(["timestamp", "record_id"]).index
    dev_ids = manifest[(manifest.partition == "validation") & manifest.eligible].index
    xt, yt = x.loc[train_ids], original.loc[train_ids, "failure"]
    times = manifest.loc[train_ids, "timestamp"].reset_index(drop=True)
    folds = []
    for a, b in TimeSeriesSplit(n_splits=3).split(xt):
        a = a[times.iloc[a].to_numpy() < times.iloc[b[0]]]
        if len(a) and yt.iloc[a].nunique() == 2 and yt.iloc[b].nunique() == 2:
            folds.append((a, b))
    if len(folds) < 2:
        raise RuntimeError("Insufficient valid temporal folds")
    def build(params):
        return Pipeline([("filter", FeatureFilter()), ("model", HistGradientBoostingClassifier(
            class_weight="balanced", max_iter=100, learning_rate=.05, early_stopping=False,
            random_state=SEED, **params))])
    grid = [{"max_leaf_nodes": leaves, "l2_regularization": penalty, "min_samples_leaf": minimum}
            for leaves in (7, 15) for penalty in (1., 10.) for minimum in (10, 20)]
    results = []
    for params in grid:
        fold_metrics = []
        for a, b in folds:
            estimator = build(params).fit(xt.iloc[a], yt.iloc[a])
            frame = pd.DataFrame({"record_id": original.loc[train_ids[b], "record_id"].to_numpy(),
                                  "failure": yt.iloc[b].to_numpy(),
                                  "priority_score": estimator.predict_proba(xt.iloc[b])[:, 1]})
            fold_metrics.append(metrics(frame))
        results.append({"params": params, "folds": fold_metrics,
                        "mean_average_precision": float(np.mean([f["average_precision"] for f in fold_metrics]))})
        print(f"Boosting candidate {len(results)}/{len(grid)} AP={results[-1]['mean_average_precision']:.4f}", flush=True)
    winner = max(results, key=lambda item: item["mean_average_precision"])
    estimator = build(winner["params"]).fit(xt, yt)
    frame = pd.DataFrame({"record_id": original.loc[dev_ids, "record_id"].to_numpy(),
                          "failure": original.loc[dev_ids, "failure"].to_numpy(),
                          "priority_score": estimator.predict_proba(x.loc[dev_ids])[:, 1]})
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "nonlinear_predictions.csv", index=False)
    save_json(output / "nonlinear.json", {"at": now(), "seed": SEED,
        "stage": "Exploratory second stage after observing weak stage-one development capture",
        "selection_metric": "Mean training expanding-window average precision",
        "boundary": "No test evaluation. Previously inspected validation is development-only, not independent evidence of generalization.",
        "model": "Histogram gradient boosting; native missing values; no random early-stopping holdout",
        "valid_folds": len(folds), "candidates": results, "selected": winner,
        "development_capacity_metrics": {str(c): metrics(frame, c) for c in (.1, .2, .3, .5, 1.)}})
    print("Boosting selected:", winner["params"], "Development at 20%:", metrics(frame), flush=True)
