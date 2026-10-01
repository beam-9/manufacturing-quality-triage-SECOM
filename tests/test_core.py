import json
import math
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import requests

from triage.common import ARTIFACTS, digest, load_json
from triage.data import partition_data, load_data
from triage.model import FeatureFilter, pipeline, ranked_queue, metrics
from triage.briefs import (Fact, EvidenceBundle, InvestigationBrief, Observation, UNCERTAINTY,
                           template, validate_brief, generate, markdown)


@pytest.fixture
def bundle():
    return EvidenceBundle(scope="unit", record_ids=["SECOM-0001"], population_hash="fixture", run_id="fixture",
                          review_capacity=.2, facts=[Fact(evidence_id="E01", text="Priority score 0.123456.")]).sealed()


def sample_frame():
    return pd.DataFrame({"record_id": ["d", "b", "a", "c", "e"],
                         "priority_score": [.1, .8, .8, .2, .3], "failure": [0, 1, 0, 1, 0]})


def test_ranking_and_independent_confusion():
    frame = sample_frame()
    queue = ranked_queue(frame, .4)
    assert queue.record_id.tolist() == ["a", "b", "e", "c", "d"]
    assert queue.selected.sum() == math.ceil(.4 * len(frame))
    outcome = metrics(frame, .4)
    # Direct arithmetic, independent of helper's implementation.
    assert outcome["captured"] == 1 and outcome["false_alarms"] == 1
    assert outcome["missed"] == 1 and outcome["true_negatives"] == 2
    assert outcome["precision"] == .5 and outcome["recall"] == .5
    assert outcome["random_expected_captured"] == .8


@pytest.mark.parametrize("capacity", [-.01, 1.01])
def test_bad_capacity(capacity):
    with pytest.raises(ValueError):
        ranked_queue(sample_frame(), capacity)


def test_zero_capacity():
    assert not ranked_queue(sample_frame(), 0).selected.any()
    assert metrics(sample_frame(), 0)["precision"] is None


def test_training_only_filter_and_pipeline():
    x = pd.DataFrame({"variable": [1., 2., 3., 4., 5., 6.], "constant": [2]*6,
                      "empty": [np.nan]*6, "missing": [1., np.nan, 3., 4., 5., 6.]})
    y = [0, 0, 1, 0, 1, 1]
    estimator = pipeline("logistic", {"C": .1}).fit(x, y)
    assert estimator.named_steps["filter"].columns_ == ["variable", "missing"]
    statistics = estimator.named_steps["impute"].statistics_.copy()
    future = x.copy()
    future["variable"] = 100000
    future["empty"] = 10
    estimator.predict_proba(future)
    assert estimator.named_steps["filter"].columns_ == ["variable", "missing"]
    assert np.array_equal(statistics, estimator.named_steps["impute"].statistics_)
    assert np.allclose(estimator.predict_proba(x), pipeline("logistic", {"C": .1}).fit(x, y).predict_proba(x))
    transformed = estimator[:-1].transform(x)
    score = transformed @ estimator[-1].coef_[0] + estimator[-1].intercept_[0]
    assert np.allclose(score, estimator.decision_function(x))


def test_partition_ties_and_duplicates():
    x = pd.DataFrame({"a": range(10), "b": range(10)})
    x.loc[9] = x.loc[0]
    meta = pd.DataFrame({"record_id": [str(i) for i in range(10)],
                         "timestamp": pd.to_datetime(["2008-01-01"]*2 + [f"2008-01-{i:02d}" for i in range(2, 10)]),
                         "failure": [0, 0, 0, 1, 0, 0, 1, 0, 0, 0]})
    result = partition_data(x, meta)
    assert result.groupby("timestamp").partition.nunique().max() == 1
    assert result.loc[9, "exclusion"] == "exact duplicate of earlier partition"
    meta.loc[9, "failure"] = 1
    conflict = partition_data(x, meta)
    assert not conflict.loc[[0, 9], "eligible"].any()


def test_template_valid_and_markdown(bundle):
    brief = template(bundle)
    validate_brief(brief, bundle)
    result = generate(bundle)
    assert result["generation"]["generation_mode"] == "template"
    assert "0.123456" in markdown(result)
    assert "Human review" in markdown(result)


@pytest.mark.parametrize("sentence", ["Failure probability is 90%.", "A sensor is overheating.",
                                      "This caused failure.", "Replace the motor.", "Measurement 042 is high."])
def test_reject_unsupported_interpretations(bundle, sentence):
    brief = template(bundle)
    brief.observations[0].finding = sentence
    with pytest.raises(ValueError):
        validate_brief(brief, bundle)


def test_unknown_reference_and_scope(bundle):
    brief = template(bundle)
    brief.observations[0].evidence_ids = ["E99"]
    with pytest.raises(ValueError):
        validate_brief(brief, bundle)
    brief = template(bundle)
    brief.scope = "batch"
    with pytest.raises(ValueError):
        validate_brief(brief, bundle)


def test_tampered_evidence(bundle):
    bundle.facts[0].text = "Altered"
    with pytest.raises(ValueError):
        validate_brief(template(bundle), bundle)


def test_malformed_schema():
    with pytest.raises(ValueError):
        InvestigationBrief.model_validate_json('{"scope":"unit"}')


def test_empty_evidence_rejected():
    with pytest.raises(ValueError):
        EvidenceBundle(scope="unit", record_ids=["a"], population_hash="fixture", run_id="fixture", review_capacity=.2, facts=[])


def test_population_change_invalidates_evidence(bundle):
    first = bundle.content_hash
    bundle.population_hash = "other-population"
    assert bundle.sealed().content_hash != first


def test_duplicate_evidence_rejected():
    with pytest.raises(ValueError):
        EvidenceBundle(scope="unit", record_ids=["a"], population_hash="fixture", run_id="fixture", review_capacity=.2,
                       facts=[Fact(evidence_id="E01", text="a"), Fact(evidence_id="E01", text="b")])


def test_valid_sentence_with_wrong_signal_rejected(bundle):
    brief = template(bundle)
    brief.observations[0].finding = "positive"
    with pytest.raises(ValueError, match="not supported"):
        validate_brief(brief, bundle)


def test_positive_sentence_requires_all_citations(bundle):
    bundle.facts[0].signals = ["positive"]
    bundle.sealed()
    brief = template(bundle)
    brief.observations[0].finding = "positive"
    validate_brief(brief, bundle)


def test_hosted_generation_never_network(bundle, monkeypatch):
    session = Mock()
    monkeypatch.setattr("triage.briefs.requests.Session", lambda: session)
    generate(bundle, live=False)
    session.get.assert_not_called()
    session.post.assert_not_called()


def test_unavailable_ollama_fallback(bundle, monkeypatch, tmp_path):
    monkeypatch.setattr("triage.briefs.PRIVATE", tmp_path)
    monkeypatch.setattr("triage.briefs.model_info", Mock(side_effect=requests.ConnectionError("offline")))
    result = generate(bundle, live=True)
    assert result["generation"]["generation_mode"] == "template"
    assert result["generation"]["errors"]


def test_timeout_has_one_repair_then_fallback(bundle, monkeypatch, tmp_path):
    monkeypatch.setattr("triage.briefs.PRIVATE", tmp_path)
    monkeypatch.setattr("triage.briefs.model_info", lambda session: ("digest", "version"))
    session = Mock()
    session.post.side_effect = requests.Timeout("test timeout")
    monkeypatch.setattr("triage.briefs.requests.Session", lambda: session)
    result = generate(bundle, live=True, use_cache=False)
    assert session.post.call_count == 2
    assert result["generation"]["generation_mode"] == "template"


@pytest.mark.skipif(not (ARTIFACTS / "evaluation.json").exists(), reason="Run official pipeline first")
def test_official_data_and_frozen_artifacts():
    if not (ARTIFACTS.parent / "data/raw/secom.data").exists():
        pytest.skip("Raw data intentionally not committed")
    x, meta = load_data()
    assert x.shape == (1567, 590) and meta.failure.sum() == 104
    manifest = pd.read_csv(ARTIFACTS / "split_manifest.csv")
    assert manifest.record_id.is_unique
    for a, b in (("train", "validation"), ("validation", "test")):
        assert manifest[manifest.partition == a].timestamp.max() < manifest[manifest.partition == b].timestamp.min()
    predictions = pd.read_csv(ARTIFACTS / "predictions.csv")
    test = predictions[(predictions.model == "logistic") & (predictions.partition == "test")]
    selected = test.sort_values(["priority_score", "record_id"], ascending=[False, True]).head(math.ceil(.2*len(test)))
    frozen = load_json(ARTIFACTS / "evaluation.json")["models"]["logistic"]["test"]["metrics"]
    assert int(selected.failure.sum()) == frozen["captured"]
    assert len(selected) - int(selected.failure.sum()) == frozen["false_alarms"]
    explanations = load_json(ARTIFACTS / "explanations.json")
    assert max(v["reconstruction_error"] for v in explanations.values()) < 1e-8
