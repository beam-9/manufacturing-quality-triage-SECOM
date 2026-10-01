import numpy as np
import pandas as pd
import pytest
from triage.study import MeasurementPreparation, temporal_folds, selection_key, compare_candidates, budget_pipeline


def test_rare_missingness_survives_value_clipping_and_scaling():
    x = pd.DataFrame({"a": np.arange(1000, dtype=float), "b": np.arange(1000, dtype=float)**2})
    x.loc[0, "a"] = np.nan
    before = x.copy()
    prep = MeasurementPreparation(clip=True, robust=True).fit(x)
    names = prep.get_feature_names_out().tolist()
    mask = prep.transform(x)[:, names.index("missingindicator_a")]
    assert mask[0] == 1 and mask.sum() == 1 and set(mask) == {0, 1}
    pd.testing.assert_frame_equal(x, before)
    bounds = prep.bounds_.copy()
    future = pd.DataFrame({"a": [1e12], "b": [np.nan]})
    assert np.isfinite(prep.transform(future)).all()
    np.testing.assert_equal(bounds, prep.bounds_)


def test_selected_feature_names_match_output():
    rng = np.random.default_rng(42)
    x = pd.DataFrame(rng.normal(size=(100, 10)), columns=list("abcdefghij"))
    prep = MeasurementPreparation(clip=True, features=3).fit(x, (x.a > 0).astype(int))
    assert len(prep.get_feature_names_out()) == prep.transform(x).shape[1] == 3


def test_folds_keep_timestamp_groups_whole():
    times = pd.Series(pd.date_range("2008-01-01", periods=12)).repeat(3).reset_index(drop=True)
    meta = pd.DataFrame({"timestamp": times})
    y = pd.Series([0, 1, 0] * 12)
    folds, _ = temporal_folds(meta, y)
    for a, b in folds:
        assert times.iloc[a].max() < times.iloc[b].min()
        for timestamp in times.iloc[b].unique():
            assert set(np.flatnonzero(times == timestamp)).issubset(b)


def test_selection_prioritizes_budget_over_ap():
    a = {"folds": [{"recall": .2}, {"recall": .3}], "mean_average_precision": .9}
    b = {"folds": [{"recall": .4}, {"recall": .4}], "mean_average_precision": .1}
    assert selection_key(b) > selection_key(a)


def test_study_rejects_test_records_before_fitting():
    x = pd.DataFrame({"a": [1, 2]})
    meta = pd.DataFrame({"record_id": ["a", "b"], "partition": ["train", "test"]})
    with pytest.raises(ValueError, match="training records"):
        compare_candidates(x, meta, [])


def test_complete_study_and_logistic_reconstruction():
    rng = np.random.default_rng(42)
    x = pd.DataFrame(rng.normal(size=(80, 6)), columns=list("abcdef"))
    y = pd.Series([0, 1] * 40)
    meta = pd.DataFrame({"record_id": [f"R{i}" for i in range(80)], "failure": y,
                         "timestamp": pd.date_range("2008-01-01", periods=80), "partition": "train"})
    params = {"clip": True, "robust": True, "features": 3, "C": .1}
    results, winner, diagnostics = compare_candidates(x, meta, [("fixture", params, budget_pipeline)])
    assert winner == results[0] and winner["eligible_for_selection"]
    assert len(winner["folds"]) == 3 and all(d["valid"] for d in diagnostics)
    fit = budget_pipeline(params).fit(x, y)
    contributions = fit[0].transform(x) * fit[-1].coef_[0]
    assert np.allclose(contributions.sum(axis=1) + fit[-1].intercept_[0], fit.decision_function(x))
    assert contributions.shape[1] == len(fit[0].get_feature_names_out())


def test_nonconverged_candidate_cannot_win():
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    class Unconverged:
        def fit(self, x, y):
            warnings.warn("unfinished fit", ConvergenceWarning)
            return self
        def predict_proba(self, x):
            return np.tile([.5, .5], (len(x), 1))
    x = pd.DataFrame({"a": np.arange(40)})
    meta = pd.DataFrame({"record_id": [str(i) for i in range(40)], "partition": "train",
                         "timestamp": pd.date_range("2008-01-01", periods=40), "failure": [0, 1]*20})
    with pytest.raises(ValueError, match="converged"):
        compare_candidates(x, meta, [("bad", {}, lambda p: Unconverged())])


def test_published_budget_study_is_independently_consistent():
    import json
    from triage.common import ARTIFACTS
    folder = ARTIFACTS / "v3_budget_verified"
    if not (folder / "study.json").exists():
        pytest.skip("Budget study not generated")
    report = json.loads((folder / "study.json").read_text())
    eligible = [c for c in report["candidates"] if c["eligible_for_selection"]]
    assert report["selected"] == max(eligible, key=selection_key)
    frame = pd.read_csv(folder / "development_predictions.csv")
    manifest = pd.read_csv(ARTIFACTS / "split_manifest.csv")
    assert set(frame.record_id) == set(manifest.loc[(manifest.partition == "validation") & manifest.eligible, "record_id"])
    for capacity, result in report["development_capacity_metrics"].items():
        n = int(np.ceil(float(capacity)*len(frame)))
        chosen = frame.sort_values(["priority_score", "record_id"], ascending=[False, True]).head(n)
        captured = int(chosen.failure.sum())
        assert result["captured"] == captured and result["false_alarms"] == n-captured
        assert result["missed"] == int(frame.failure.sum())-captured
