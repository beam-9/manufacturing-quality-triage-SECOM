import numpy as np
import pandas as pd
import pytest
import json
from triage.common import ARTIFACTS
from triage.experiments import TrainingQuantileClip, candidate_pipeline


def test_clip_uses_training_only_and_preserves_source():
    train = np.arange(100, dtype=float).reshape(-1, 1)
    clip = TrainingQuantileClip().fit(train)
    original = np.array([[9999.], [-9999.]])
    bounded = clip.transform(original)
    assert np.allclose(bounded[:, 0], [98.01, .99])
    assert original[0, 0] == 9999
    assert np.allclose(clip.bounds_[:, 0], [.99, 98.01])


def test_development_pipeline_reproducible():
    rng = np.random.default_rng(42)
    x = pd.DataFrame(rng.normal(size=(80, 25)))
    x.columns = [f"Measurement {i:03d}" for i in range(25)]
    y = (x.iloc[:, 0] > 0).astype(int)
    params = {"clip": True, "robust": True, "features": 20, "C": .1}
    a, b = candidate_pipeline(params).fit(x, y), candidate_pipeline(params).fit(x, y)
    assert np.allclose(a.predict_proba(x), b.predict_proba(x))


@pytest.mark.parametrize("report,predictions", [("study.json", "development_predictions.csv"),
                                              ("nonlinear.json", "nonlinear_predictions.csv")])
def test_study_selection_and_independent_capture(report, predictions):
    folder = ARTIFACTS / "v2_development"
    if not (folder / report).exists():
        pytest.skip("Study artifacts not available")
    study = json.loads((folder / report).read_text())
    assert study["selected"] == max(study["candidates"], key=lambda row: row["mean_average_precision"])
    frame = pd.read_csv(folder / predictions)
    manifest = pd.read_csv(ARTIFACTS / "split_manifest.csv")
    valid = set(manifest.loc[(manifest.partition == "validation") & manifest.eligible, "record_id"])
    assert set(frame.record_id) == valid
    for capacity, counts in study["development_capacity_metrics"].items():
        ranked = frame.sort_values(["priority_score", "record_id"], ascending=[False, True])
        n = int(np.ceil(float(capacity) * len(frame)))
        captured = int(ranked.head(n).failure.sum())
        assert counts["captured"] == captured
        assert counts["missed"] == int(frame.failure.sum()) - captured
        assert counts["false_alarms"] == n - captured
