"""Training-only preparation and budget-aligned chronological studies."""
from __future__ import annotations

import hashlib
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from triage.common import ARTIFACTS, RAW, SEED, digest, load_json, now, save_json
from triage.data import load_data
from triage.model import FeatureFilter, metrics, pipeline


class MeasurementPreparation(BaseEstimator, TransformerMixin):
    """Scale/clip values only; append untouched, named training-observed masks."""
    def __init__(self, clip=False, robust=False, features="all"):
        self.clip = clip
        self.robust = robust
        self.features = features

    def fit(self, X, y=None):
        self.filter_ = FeatureFilter().fit(X)
        values = self.filter_.transform(X)
        self.columns_ = self.filter_.columns_
        self.mask_columns_ = list(values.columns[values.isna().any()])
        self.imputer_ = SimpleImputer(strategy="median").fit(values)
        matrix = self.imputer_.transform(values)
        self.bounds_ = np.quantile(matrix, [.01, .99], axis=0)
        if self.clip:
            matrix = np.clip(matrix, *self.bounds_)
        self.scaler_ = (RobustScaler() if self.robust else StandardScaler()).fit(matrix)
        prepared = self._prepare(X)
        # Clipping may make values constant; masks are never clipped or scaled.
        self.variable_ = np.var(prepared, axis=0) > 0
        names = np.array(self.columns_ + ["missingindicator_" + c for c in self.mask_columns_])
        self.feature_names_ = names[self.variable_]
        self.selector_ = None
        if self.features != "all":
            self.selector_ = SelectKBest(f_classif, k=min(int(self.features), len(self.feature_names_)))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.selector_.fit(prepared[:, self.variable_], y)
            self.selection_warnings_ = [str(w.message) for w in caught]
            self.feature_names_ = self.feature_names_[self.selector_.get_support()]
        return self

    def _prepare(self, X):
        values = self.filter_.transform(X)
        matrix = self.imputer_.transform(values)
        if self.clip:
            matrix = np.clip(matrix, *self.bounds_)
        matrix = self.scaler_.transform(matrix)
        masks = values[self.mask_columns_].isna().to_numpy(dtype=float)
        return np.column_stack([matrix, masks])

    def transform(self, X):
        prepared = self._prepare(X)[:, self.variable_]
        return self.selector_.transform(prepared) if self.selector_ is not None else prepared

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_.copy()


def budget_pipeline(params):
    return Pipeline([("prepare", MeasurementPreparation(params["clip"], params["robust"], params["features"])),
                     ("model", LogisticRegression(C=params["C"], class_weight="balanced",
                                                  max_iter=4000, random_state=SEED))])


def temporal_folds(meta, y):
    """Expand over complete timestamp groups, including both validation ends."""
    times = pd.to_datetime(meta.timestamp).reset_index(drop=True)
    if not times.is_monotonic_increasing or times.isna().any():
        raise ValueError("Study timestamps must be valid and chronological")
    groups = np.array_split(times.unique(), 4)
    valid, diagnostics = [], []
    for fold in range(1, 4):
        earlier = np.concatenate(groups[:fold])
        a = np.flatnonzero(times.isin(earlier))
        b = np.flatnonzero(times.isin(groups[fold]))
        ok = bool(len(a) and len(b) and y.iloc[a].nunique() == 2 and y.iloc[b].nunique() == 2)
        diagnostics.append({"fold": fold, "valid": ok, "train_records": len(a), "review_records": len(b),
                            "failures": int(y.iloc[b].sum()),
                            "reason": None if ok else "empty or single-class fold"})
        if ok:
            valid.append((a, b))
    if len(valid) < 2:
        raise ValueError("At least two valid temporal folds are required")
    return valid, diagnostics


def selection_key(candidate):
    """Declared before scoring: mean recall@20, worst-window recall, then AP."""
    recalls = [f["recall"] for f in candidate["folds"]]
    return float(np.mean(recalls)), float(min(recalls)), candidate["mean_average_precision"]


def compare_candidates(x, meta, candidates):
    """The study interface excludes test/development fitting and accepts real recipe adapters."""
    if not meta.record_id.is_unique or set(meta.partition) != {"train"}:
        raise ValueError("Study fitting requires unique training records only")
    if len(x) != len(meta) or not x.index.equals(meta.index):
        raise ValueError("Measurement and record alignment mismatch")
    y = meta.failure
    folds, diagnostics = temporal_folds(meta, y)
    results = []
    for name, params, build in candidates:
        scores = []
        fit_warnings = []
        for a, b in folds:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fitted = build(params).fit(x.iloc[a], y.iloc[a])
            fit_warnings.extend({"fold": len(scores) + 1, "category": w.category.__name__, "message": str(w.message)} for w in caught)
            frame = pd.DataFrame({"record_id": meta.record_id.iloc[b].to_numpy(),
                                  "failure": y.iloc[b].to_numpy(),
                                  "priority_score": fitted.predict_proba(x.iloc[b])[:, 1]})
            scores.append(metrics(frame, .2))
        entry = {"name": name, "params": params, "folds": scores,
                 "mean_average_precision": float(np.mean([s["average_precision"] for s in scores])),
                 "fit_warnings": fit_warnings,
                 "eligible_for_selection": not any(w["category"] == "ConvergenceWarning" for w in fit_warnings)}
        entry["selection_key"] = list(selection_key(entry))
        results.append(entry)
        print(f"Candidate {len(results)}/{len(candidates)} {name}: recall={entry['selection_key'][0]:.3f}", flush=True)
    eligible = [r for r in results if r["eligible_for_selection"]]
    if not eligible:
        raise ValueError("No converged candidates eligible for selection")
    return results, max(eligible, key=selection_key), diagnostics


def improve_budget():
    output = ARTIFACTS / "v3_budget_verified"
    if output.exists():
        raise RuntimeError("Budget study exists; preserve its immutable outputs")
    # Verify source and frozen membership before creating any new study.
    for source in load_json(ARTIFACTS / "provenance.json")["sources"]:
        if hashlib.sha256((RAW / source["file"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("Source checksum mismatch")
    manifest = pd.read_csv(ARTIFACTS / "split_manifest.csv", parse_dates=["timestamp"])
    frozen = load_json(ARTIFACTS / "run.json")
    if digest(manifest[["record_id", "partition", "eligible"]].to_dict("records")) != frozen["split_hash"]:
        raise ValueError("Frozen membership changed")
    x, original = load_data()
    if manifest.record_id.tolist() != original.record_id.tolist() or manifest.failure.tolist() != original.failure.tolist():
        raise ValueError("Manifest record/label alignment changed")
    train = manifest[(manifest.partition == "train") & manifest.eligible].sort_values(["timestamp", "record_id"])
    dev = manifest[(manifest.partition == "validation") & manifest.eligible].sort_values(["timestamp", "record_id"])
    grid = [{"clip": clip, "robust": robust, "features": k, "C": c}
            for clip in (False, True) for robust in (False, True) for k in (20, "all") for c in (.01, .1, 1.)]
    recipes = [("corrected_logistic", p, budget_pipeline) for p in grid]
    recipes += [("v1_logistic_reference", frozen["selected_params"]["logistic"], lambda p: pipeline("logistic", p))]
    protocol = {"at": now(), "seed": SEED, "parent_run": frozen["run_id"],
                "objective": "Converged candidates only; mean recall at 20% workload; ties: worst-fold recall, then mean AP; stable grid order",
                "grid": [{"name": name, "params": p} for name, p, _ in recipes],
                "folds": "Three expanding complete-timestamp-group folds within original training",
                "exposure": "Original validation/test already inspected. Validation remains development-only; test is not scored or used to select recipes.",
                "promotion": "No automatic promotion. Requires stable gains and independent new evidence."}
    output.mkdir()
    save_json(output / "protocol.json", protocol)  # Published before any candidate scoring.
    results, winner, folds = compare_candidates(x.loc[train.index], train, recipes)
    build = next(build for name, p, build in recipes if name == winner["name"] and p == winner["params"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator = build(winner["params"]).fit(x.loc[train.index], train.failure)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError("Selected full-training fit failed to converge; no development evaluation")
    frame = pd.DataFrame({"record_id": dev.record_id.to_numpy(), "failure": dev.failure.to_numpy(),
                          "priority_score": estimator.predict_proba(x.loc[dev.index])[:, 1]})
    frame.to_csv(output / "development_predictions.csv", index=False)
    save_json(output / "study.json", {"protocol": protocol, "fold_diagnostics": folds,
        "candidates": results, "selected": winner,
        "full_fit_warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught],
        "development_capacity_metrics": {str(c): metrics(frame, c) for c in (.1, .2, .3, .5)},
        "selected_feature_names": estimator[0].get_feature_names_out().tolist() if winner["name"] == "corrected_logistic" else None,
        "selection_warnings": getattr(estimator[0], "selection_warnings_", [])})
    print("Selected", winner["params"], "Development", metrics(frame), flush=True)
