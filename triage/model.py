from __future__ import annotations

import math
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from triage.common import ARTIFACTS, PRIVATE, SEED, LIMITATIONS, digest, now, save_json, load_json
from triage.data import load_data, partition_data, audit


class FeatureFilter(BaseEstimator, TransformerMixin):
    def __init__(self, max_missing=.8):
        self.max_missing = max_missing

    def fit(self, X, y=None):
        self.columns_ = list(X.columns[(X.isna().mean() <= self.max_missing) & (X.nunique() > 1)])
        if not self.columns_:
            raise ValueError("No usable measurements remain")
        return self

    def transform(self, X):
        return X[self.columns_]


def pipeline(kind, params):
    if kind == "logistic":
        estimator = LogisticRegression(class_weight="balanced", max_iter=4000, random_state=SEED, **params)
    elif kind == "forest":
        estimator = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                            random_state=SEED, n_jobs=2, **params)
    else:
        estimator = DummyClassifier(strategy="prior")
    steps = [("filter", FeatureFilter()), ("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if kind == "logistic":
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps + [("model", estimator)])


def train():
    if (ARTIFACTS / "run.json").exists():
        raise RuntimeError("Frozen run exists. Preserve v1; use a new checkout/output for further experiments.")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    x, meta = load_data()
    meta = partition_data(x, meta)
    audit(x, meta)
    indices = meta[(meta.partition == "train") & meta.eligible].sort_values(["timestamp", "record_id"]).index
    xt, yt = x.loc[indices], meta.loc[indices, "failure"]
    folds = []
    skipped = []
    for a, b in TimeSeriesSplit(n_splits=3).split(xt):
        # Adjust fold boundary to keep shared timestamps on the validation side.
        times = meta.loc[indices, "timestamp"].reset_index(drop=True)
        a = a[times.iloc[a].to_numpy() < times.iloc[b[0]]]
        if len(a) and yt.iloc[a].nunique() == 2 and yt.iloc[b].nunique() == 2:
            folds.append((a, b))
        else:
            skipped.append({"train_records": len(a), "validation_records": len(b), "reason": "single-class or empty fold"})
    grids = {"logistic": [{"C": c} for c in (.01, .1, 1.)],
             "forest": [{"max_depth": depth, "min_samples_leaf": leaf} for depth in (5, None) for leaf in (5, 10)],
             "dummy": [{}]}
    defaults = {"logistic": {"C": .1}, "forest": {"max_depth": 5, "min_samples_leaf": 10}, "dummy": {}}
    results, selected = [], {}
    for kind, grid in grids.items():
        candidates = []
        if len(folds) >= 2:
            for params in grid:
                scores = []
                for a, b in folds:
                    estimator = pipeline(kind, params).fit(xt.iloc[a], yt.iloc[a])
                    scores.append(float(average_precision_score(yt.iloc[b], estimator.predict_proba(xt.iloc[b])[:, 1])))
                entry = {"model": kind, "params": params, "fold_average_precision": scores,
                         "mean_average_precision": float(np.mean(scores))}
                results.append(entry)
                candidates.append(entry)
            selected[kind] = max(candidates, key=lambda item: item["mean_average_precision"])["params"]
        else:
            selected[kind] = defaults[kind]
        estimator = pipeline(kind, selected[kind]).fit(xt, yt)
        joblib.dump(estimator, PRIVATE / f"{kind}.joblib")
    provenance = load_json(ARTIFACTS / "provenance.json")
    split_hash = digest(meta[["record_id", "partition", "eligible"]].to_dict("records"))
    run_id = digest({"sources": provenance["sources"], "split_hash": split_hash,
                     "selected": selected, "seed": SEED})[:16]
    save_json(ARTIFACTS / "run.json", {
        "schema_version": "1.0", "run_id": run_id, "frozen_at": now(), "seed": SEED,
        "python": platform.python_version(), "queue_model": "logistic", "review_capacity": .2,
        "policy": "top ceil(capacity*N), descending score, ascending record ID ties",
        "selected_params": selected, "cv_valid_folds": len(folds), "cv_skipped": skipped,
        "cv_results": results, "split_hash": split_hash, "limitations": LIMITATIONS,
        "test_selection_policy": "No test-driven parameter, preprocessing, or capacity changes",
    })
    save_json(ARTIFACTS / "development_ledger.json", [
        {"at": now(), "stage": "freeze", "detail": "Split manifest, training-only tuning, queue model and 20% policy fixed before test evaluation."}
    ])
    print(f"Frozen run {run_id}; valid training CV folds: {len(folds)}")


def ranked_queue(frame, capacity=.2):
    if not 0 <= capacity <= 1:
        raise ValueError("Review capacity must be between zero and one")
    ordered = frame.sort_values(["priority_score", "record_id"], ascending=[False, True]).copy()
    ordered["rank"] = np.arange(1, len(ordered) + 1)
    ordered["selected"] = ordered["rank"] <= math.ceil(capacity * len(ordered))
    return ordered


def metrics(frame, capacity=.2):
    queue = ranked_queue(frame, capacity)
    y = queue.failure.to_numpy()
    flag = queue.selected.to_numpy()
    tp = int(y[flag].sum())
    fp = int(flag.sum()) - tp
    fn = int(y.sum()) - tp
    tn = len(y) - tp - fp - fn
    return {"records": len(y), "failures": int(y.sum()), "reviewed": int(flag.sum()),
            "captured": tp, "false_alarms": fp, "missed": fn, "true_negatives": tn,
            "precision": tp / int(flag.sum()) if flag.any() else None,
            "recall": tp / int(y.sum()) if y.sum() else None,
            "average_precision": float(average_precision_score(y, queue.priority_score)) if y.sum() else None,
            "random_expected_captured": float(flag.sum() * y.mean()) if len(y) else None}


def bootstrap(frame, repeats=1000):
    rng = np.random.default_rng(SEED)
    positive = frame[frame.failure == 1]
    negative = frame[frame.failure == 0]
    if positive.empty or negative.empty:
        return {"status": "Unavailable: partition lacks both classes"}
    values = {key: [] for key in ("precision", "recall", "average_precision")}
    for _ in range(repeats):
        sample = pd.concat([positive.iloc[rng.integers(len(positive), size=len(positive))],
                            negative.iloc[rng.integers(len(negative), size=len(negative))]])
        result = metrics(sample)
        for key in values:
            values[key].append(result[key])
    return {"repeats": repeats, "method": "stratified percentile bootstrap, conditional on observed prevalence",
            "caveat": "Temporal dependence can make these approximate 95% intervals optimistic.",
            "intervals": {key: list(map(float, np.quantile(value, [.025, .975]))) for key, value in values.items()}}


def explain(estimator, x, xt):
    filt = estimator.named_steps["filter"]
    imp = estimator.named_steps["impute"]
    features = imp.get_feature_names_out(filt.columns_)
    transformed = estimator[:-1].transform(x)
    coef = estimator.named_steps["model"].coef_[0]
    contributions = transformed * coef
    intercept = float(estimator.named_steps["model"].intercept_[0])
    expected = estimator.decision_function(x)
    if not np.allclose(contributions.sum(axis=1) + intercept, expected, atol=1e-8):
        raise AssertionError("Explanation reconstruction failed")
    references = {}
    for col in filt.columns_:
        values = xt[col].dropna()
        references[col] = {"median": float(values.median()), "p05": float(values.quantile(.05)),
                           "p95": float(values.quantile(.95)), "missing_fraction": float(xt[col].isna().mean())}
    output = {}
    for pos, (idx, row) in enumerate(x.iterrows()):
        contrib = contributions[pos]
        top = list(np.argsort(contrib)[-5:][::-1]) + list(np.argsort(contrib)[:5])
        details = []
        for j in dict.fromkeys(top):
            name = str(features[j])
            missingness = name.startswith("missingindicator_")
            col = name.removeprefix("missingindicator_")
            value = None if pd.isna(row[col]) else float(row[col])
            details.append({"feature": name, "measurement": col, "kind": "missingness" if missingness else "value",
                            "value": value, "imputed": value is None, "contribution": float(contrib[j]),
                            "reference": references[col]})
        output[str(idx)] = {"intercept": intercept, "log_odds": float(expected[pos]),
                            "reconstruction_error": float(abs(contrib.sum() + intercept - expected[pos])),
                            "features": details}
    global_coeff = pd.DataFrame({"feature": features, "coefficient": coef})
    global_coeff["absolute_coefficient"] = global_coeff.coefficient.abs()
    global_coeff.sort_values("absolute_coefficient", ascending=False).to_csv(ARTIFACTS / "coefficients.csv", index=False)
    save_json(ARTIFACTS / "references.json", references)
    return output


def evaluate():
    if (ARTIFACTS / "evaluation.json").exists():
        print("Returning frozen evaluation; no retuning or test re-selection.")
        return load_json(ARTIFACTS / "evaluation.json")
    run = load_json(ARTIFACTS / "run.json")
    x, original_meta = load_data()
    meta = pd.read_csv(ARTIFACTS / "split_manifest.csv", parse_dates=["timestamp"])
    meta["exclusion"] = meta.exclusion.fillna("")
    if digest(meta[["record_id", "partition", "eligible"]].to_dict("records")) != run["split_hash"]:
        raise ValueError("Frozen split manifest changed")
    # Verify data bytes before loading trusted, locally generated model artifacts.
    import hashlib
    from triage.common import RAW
    for source in load_json(ARTIFACTS / "provenance.json")["sources"]:
        if hashlib.sha256((RAW / source["file"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("Downloaded source changed since freeze")
    result = {"run_id": run["run_id"], "evaluated_at": now(), "policy_capacity": .2, "models": {}}
    all_predictions = []
    for kind in ("dummy", "logistic", "forest"):
        estimator = joblib.load(PRIVATE / f"{kind}.joblib")
        result["models"][kind] = {}
        for part in ("validation", "test"):
            mask = (meta.partition == part) & meta.eligible
            frame = meta.loc[mask, ["record_id", "timestamp", "failure", "partition"]].copy()
            if frame.empty:
                raise ValueError(f"Empty {part} partition")
            frame["priority_score"] = estimator.predict_proba(x.loc[mask])[:, 1]
            frame["model"] = kind
            all_predictions.append(frame)
            result["models"][kind][part] = {"metrics": metrics(frame), "uncertainty": bootstrap(frame)}
            p, r, _ = precision_recall_curve(frame.failure, frame.priority_score)
            save_json(ARTIFACTS / f"pr_{kind}_{part}.json", {"precision": p.tolist(), "recall": r.tolist()})
    predictions = pd.concat(all_predictions)
    predictions.to_csv(ARTIFACTS / "predictions.csv", index=False)
    logistic = joblib.load(PRIVATE / "logistic.joblib")
    remaining = meta.eligible & (meta.partition != "train")
    explanations = explain(logistic, x.loc[remaining], x.loc[meta.eligible & (meta.partition == "train")])
    save_json(ARTIFACTS / "explanations.json", {meta.loc[int(idx), "record_id"]: value for idx, value in explanations.items()})
    # Validation permutation study limited to the 30 largest training-coefficient measurements.
    coef = pd.read_csv(ARTIFACTS / "coefficients.csv")
    cols = list(dict.fromkeys(coef.feature.str.replace("missingindicator_", "", regex=False)))[:30]
    val = meta.eligible & (meta.partition == "validation")
    base_ap = average_precision_score(meta.loc[val, "failure"], logistic.predict_proba(x.loc[val])[:, 1])
    rng = np.random.default_rng(SEED)
    permutation = []
    for col in cols:
        drops = []
        for _ in range(5):
            sample = x.loc[val].copy()
            sample[col] = rng.permutation(sample[col].to_numpy())
            drops.append(float(base_ap - average_precision_score(meta.loc[val, "failure"], logistic.predict_proba(sample)[:, 1])))
        permutation.append({"measurement": col, "mean_ap_drop": float(np.mean(drops)), "std_ap_drop": float(np.std(drops)), "repeats": 5})
    pd.DataFrame(permutation).to_csv(ARTIFACTS / "permutation_validation.csv", index=False)
    # Bounded raw measurements for explainable reference plots; no labels in this source.
    export = x.loc[remaining, logistic.named_steps["filter"].columns_].copy()
    export.insert(0, "record_id", meta.loc[remaining, "record_id"])
    export.to_csv(ARTIFACTS / "measurements.csv", index=False)
    train_values = x.loc[meta.eligible & (meta.partition == "train"), logistic.named_steps["filter"].columns_]
    histograms = {}
    for col in train_values:
        counts, edges = np.histogram(train_values[col].dropna(), bins=20)
        histograms[col] = {"counts": counts.tolist(), "edges": edges.tolist()}
    save_json(ARTIFACTS / "training_histograms.json", histograms)
    save_json(ARTIFACTS / "evaluation.json", result)
    ledger = load_json(ARTIFACTS / "development_ledger.json")
    ledger.append({"at": now(), "stage": "evaluate", "detail": "Frozen models evaluated once; results are descriptive, no subsequent tuning."})
    save_json(ARTIFACTS / "development_ledger.json", ledger)
    print({kind: result["models"][kind]["test"]["metrics"] for kind in result["models"]})
    return result

