from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from triage.common import ARTIFACTS, LIMITATIONS, load_json
from triage.model import ranked_queue, metrics
from triage.briefs import unit_bundle, batch_bundle, generate, markdown, mark_review, EvidenceBundle

st.set_page_config(page_title="Manufacturing quality triage", page_icon="🔎", layout="wide")
MODE = os.getenv("APP_MODE", "hosted")
if MODE not in ("hosted", "local"):
    st.error("APP_MODE must be hosted or local.")
    st.stop()


@st.cache_data
def json_artifact(name, stamp):
    return load_json(ARTIFACTS / name)


def js(name):
    path = ARTIFACTS / name
    return json_artifact(name, path.stat().st_mtime_ns)


@st.cache_data
def csv_artifact(name, stamp):
    return pd.read_csv(ARTIFACTS / name)


def csv(name):
    path = ARTIFACTS / name
    return csv_artifact(name, path.stat().st_mtime_ns)


def brief_panel(bundle, label):
    saved = js("demo_briefs.json") if (ARTIFACTS / "demo_briefs.json").exists() else {}
    key = "brief_" + bundle.content_hash
    current = st.session_state.get(key, saved.get(bundle.content_hash))
    if MODE == "local":
        if st.button(f"Generate {label} with local Ollama", key="generate_" + bundle.content_hash):
            with st.spinner("Generating and checking references; never an autonomous decision…"):
                current = generate(bundle, live=True)
                st.session_state[key] = current
    if current is None:
        current = generate(bundle, live=False)
    gen = current["generation"]
    source = "Saved Ollama draft" if bundle.content_hash in saved and gen["generation_mode"] == "ollama" else (
        "Local Ollama draft" if gen["generation_mode"] == "ollama" else "Deterministic template — not an AI generation")
    st.caption(f"{source} · {gen['review_status']} · generated {gen['generated_at'][:19]} UTC")
    if gen["errors"]:
        with st.expander("Generation checks and fallback details"):
            st.write(gen["errors"])
    st.markdown(markdown(current).split("## Uncertainty")[0].split("\n\n", 2)[-1])
    st.warning("Draft only. Associations are not causes; measurement meanings are unknown. Human review is required.")
    with st.expander("Evidence and generation provenance"):
        st.json(current)
    if MODE == "local" and key in st.session_state:
        approval, rejection = st.columns(2)
        with approval:
            if st.button("Approve after reviewing evidence", key="approve_" + bundle.content_hash):
                st.session_state[key] = mark_review(current, "approved")
                st.rerun()
        with rejection:
            if st.button("Reject draft", key="reject_" + bundle.content_hash):
                st.session_state[key] = mark_review(current, "rejected")
                st.rerun()
    st.download_button(f"Download {label} brief (.md)", markdown(current),
                       file_name=f"{label.replace(' ', '_')}_brief.md", key="export_" + bundle.content_hash)


st.title("Manufacturing quality triage")
st.caption("SECOM · retrospective semiconductor quality review · public data, not Seagate data")
if not (ARTIFACTS / "evaluation.json").exists():
    st.info("No evaluated run is available. Run: triage download → triage train → triage evaluate → triage build-demo.")
    st.stop()

run = js("run.json")
audit = js("audit.json")
evaluation = js("evaluation.json")
predictions = csv("predictions.csv")
explanations = js("explanations.json")

with st.sidebar:
    st.header("Review workspace")
    page = st.radio("View", ["Review queue", "Model performance", "Data and methods"])
    st.caption(f"Run {run['run_id']}\n\nFrozen {run['frozen_at'][:10]}")
    st.info("Hosted: saved briefs only" if MODE == "hosted" else "Local: live Ollama + review log")
    st.link_button("Source dataset / license", "https://archive.ics.uci.edu/dataset/179/secom")
    st.link_button("Project repository", "https://github.com/beam-9/manufacturing-quality-triage-SECOM")

test = predictions[(predictions.model == "logistic") & (predictions.partition == "test")].copy()

if page == "Review queue":
    st.subheader("Where should an engineer look first?")
    st.write("Rank units by model priority, inspect the supporting evidence, then review the draft brief. The illustrative default reviews the highest-scoring fifth of units.")
    capacity = st.slider("Review capacity (%)", 0, 100, 20, 5, key="capacity") / 100
    st.button("Reset to frozen 20% policy", on_click=lambda: st.session_state.update(capacity=20))
    if capacity != .2:
        st.info("Exploratory scenario. The published evaluation uses the frozen 20% policy; no model is retrained.")
    queue = ranked_queue(test, capacity)
    selected = queue[queue.selected]
    c1, c2, c3 = st.columns(3)
    c1.metric("Units in held-out population", len(queue))
    c2.metric("Units selected for review", len(selected))
    c3.metric("Review capacity", f"{capacity:.0%}")
    st.caption("Priority score is an uncalibrated model output, not a percentage chance of failure. Test outcomes are deliberately hidden here.")
    saturated = int(((queue.priority_score <= 1e-6) | (queue.priority_score >= 1 - 1e-6)).sum())
    if saturated:
        st.warning(f"{saturated} scores are near the ends of the model scale. Extreme inputs can produce saturated scores; inspect reference distributions rather than treating these as certainty.")
    table = queue[["record_id", "timestamp", "rank", "priority_score", "selected"]].copy()
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "priority_score": st.column_config.NumberColumn("Priority score", format="%.4f"),
        "record_id": "Unit", "timestamp": "Test-point timestamp", "rank": "Priority rank", "selected": "Selected for review"})
    st.download_button("Download current queue (.csv)", table.to_csv(index=False), "review_queue.csv", "text/csv")
    if selected.empty:
        st.info("No units selected at this capacity. Unit evidence remains available; batch brief is unavailable.")
    record = st.selectbox("Inspect unit", queue.record_id.tolist())
    feature_data = pd.DataFrame(explanations[record]["features"])
    feature_data["display_feature"] = feature_data.feature.str.replace("missingindicator_", "Missing: ", regex=False)
    st.subheader(f"Evidence for {record}")
    fig = px.bar(feature_data.sort_values("contribution"), x="contribution", y="display_feature", orientation="h",
                 color="kind", color_discrete_map={"value": "#087F8C", "missingness": "#B05F00"},
                 labels={"contribution": "Contribution to model log-odds", "display_feature": "", "kind": "Evidence type"})
    fig.add_vline(x=0, line_color="#172B4D")
    fig.update_layout(height=400, margin=dict(l=10, r=20, t=15, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Positive contributions raise the model score; negative contributions lower it. These are model explanations, not physical diagnoses. Only the strongest displayed contributions are shown; reconstruction checks use all features.")
    display = feature_data[["measurement", "kind", "value", "imputed", "contribution"]].copy()
    display["training median"] = feature_data.reference.map(lambda r: r["median"])
    display["training p05"] = feature_data.reference.map(lambda r: r["p05"])
    display["training p95"] = feature_data.reference.map(lambda r: r["p95"])
    st.dataframe(display, hide_index=True, width="stretch")
    measurement = st.selectbox("Inspect training reference distribution", feature_data.measurement.drop_duplicates().tolist())
    hist = js("training_histograms.json")[measurement]
    edges = hist["edges"]
    centers = [(a+b)/2 for a, b in zip(edges[:-1], edges[1:])]
    reference = go.Figure(go.Bar(x=centers, y=hist["counts"], marker_color="#087F8C", name="Training observations"))
    raw_value = feature_data[feature_data.measurement == measurement].iloc[0]["value"]
    if pd.notna(raw_value) and edges[0] <= raw_value <= edges[-1]:
        reference.add_vline(x=float(raw_value), line_dash="dash", line_color="#B05F00", annotation_text="Selected unit")
    elif pd.notna(raw_value):
        st.warning(f"Selected value {raw_value:.6g} lies outside the plotted training range {edges[0]:.6g}–{edges[-1]:.6g}. Check data integrity and process context; this is not a diagnosis.")
    else:
        st.info("This unit's original measurement is missing; no fabricated point is plotted.")
    reference.update_layout(xaxis_title=f"{measurement} — units unknown", yaxis_title="Training observations", xaxis_range=[edges[0], edges[-1]], height=280)
    st.plotly_chart(reference, use_container_width=True)
    unit = unit_bundle(record, test, explanations, capacity)
    with st.expander("Unit investigation brief", expanded=True):
        brief_panel(unit, "unit")
    if not selected.empty:
        with st.expander("Selected batch investigation brief"):
            brief_panel(batch_bundle(test, explanations, capacity), "batch")

elif page == "Model performance":
    st.subheader("Retrospective evaluation — known test outcomes")
    st.write("This view uses historical pass/fail labels to assess triage, not to generate operational briefs. Models and review policy were frozen before test evaluation.")
    st.warning("Generalization warning: logistic triage captured no failures at the default capacity in validation, despite capturing some in test. This prototype is not ready for operational reliance.")
    part = st.radio("Evaluation partition", ["test", "validation"], horizontal=True)
    scores = evaluation["models"]["logistic"][part]["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Failures captured", f"{scores['captured']} / {scores['failures']}")
    c2.metric("Failures missed", scores["missed"])
    c3.metric("Passing units flagged", scores["false_alarms"])
    c4.metric("Units reviewed", f"{scores['reviewed']} / {scores['records']}")
    st.caption(f"At identical capacity, random review would capture {scores['random_expected_captured']:.1f} failures on average—not a measured random trial.")
    comparison = []
    for kind, partitions in evaluation["models"].items():
        m = partitions[part]["metrics"]
        comparison.append({"Model": kind, "Average precision": m["average_precision"], "Precision at capacity": m["precision"],
                           "Recall at capacity": m["recall"], "Captured": m["captured"], "Missed": m["missed"], "False alarms": m["false_alarms"]})
    st.dataframe(pd.DataFrame(comparison), hide_index=True, width="stretch")
    st.caption("Average precision summarizes precision across recall levels; precision is the fraction of reviewed units that failed, and recall is the fraction of failures captured. All models review the same number of units. The dummy score ties are resolved by unit ID, not outcome.")
    curves = go.Figure()
    for kind in evaluation["models"]:
        values = js(f"pr_{kind}_{part}.json")
        curves.add_trace(go.Scatter(x=values["recall"], y=values["precision"], name=kind, mode="lines", line_shape="hv"))
    curves.add_hline(y=scores["failures"]/scores["records"], line_dash="dot", annotation_text="Failure prevalence")
    curves.update_layout(xaxis_title="Recall — share of failures captured", yaxis_title="Precision — failures / reviewed units",
                         xaxis_range=[0, 1], yaxis_range=[0, 1], height=350)
    st.plotly_chart(curves, use_container_width=True)
    uncertainty = evaluation["models"]["logistic"][part]["uncertainty"]
    st.subheader("Uncertainty and workload tradeoff")
    if "intervals" in uncertainty:
        st.dataframe(pd.DataFrame([{"Metric": k, "Lower 95% bound": v[0], "Upper 95% bound": v[1]} for k, v in uncertainty["intervals"].items()]), hide_index=True)
        st.caption("1,000 seeded stratified bootstrap resamples, conditional on observed prevalence. Temporal dependence can make intervals optimistic; small failure counts limit confidence.")
    else:
        st.warning(uncertainty["status"])
    population = predictions[(predictions.model == "logistic") & (predictions.partition == part)]
    workload = pd.DataFrame([{"Capacity": c, **metrics(population, c)} for c in (.1, .2, .3, .4, .5)])
    st.plotly_chart(px.line(workload, x="reviewed", y=["captured", "false_alarms"], markers=True,
                           labels={"reviewed": "Units reviewed", "value": "Units", "variable": "Outcome"}), use_container_width=True)
    st.caption("Capacity changes are retrospective scenarios, not another tuned policy or an estimate of financial savings.")
    st.subheader("Global model associations")
    coeff = csv("coefficients.csv").head(20)
    coefficients_plot = px.bar(coeff.sort_values("coefficient"), x="coefficient", y="feature", orientation="h",
                               labels={"feature": "", "coefficient": "Standardized logistic coefficient"})
    coefficients_plot.update_layout(height=580)
    st.plotly_chart(coefficients_plot, use_container_width=True)
    st.dataframe(csv("permutation_validation.csv"), hide_index=True, width="stretch")
    st.caption("Permutation study: five shuffles on validation, bounded to the 30 strongest training-coefficient measurements. Negative drops and unstable rankings are retained. Correlated inputs can redistribute importance; these are not causal findings.")

else:
    st.subheader("Data quality, provenance and reproducibility")
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw production records", audit["raw_records"])
    c2.metric("Raw measurements", audit["raw_measurements"])
    c3.metric("Historical failures", audit["raw_failures"])
    st.dataframe(pd.DataFrame(audit["partitions"]), hide_index=True, width="stretch")
    st.caption("Approximate 60/20/20 chronological partitions; tied timestamps stay together. Publisher timezone is unspecified. Exclusions are disclosed below.")
    d1, d2, d3 = st.columns(3)
    d1.metric("Missing source values", f"{audit['missing_fraction']:.2%}")
    d2.metric("Constant source columns", audit["constant_columns"])
    d3.metric("Exact duplicate rows", audit["duplicate_rows"])
    st.caption("No source records excluded." if not audit["exclusions"] else f"Exclusions: {audit['exclusions']}")
    dictionary = csv("data_dictionary.csv")
    st.plotly_chart(px.histogram(dictionary, x="missing_fraction", nbins=20,
                               labels={"missing_fraction": "Fraction of missing source values"}), use_container_width=True)
    daily = csv("daily_quality.csv")
    daily["failure_fraction"] = daily.failures / daily.records
    st.plotly_chart(px.scatter(daily, x="date", y="failure_fraction", size="records",
                              hover_data=["failures", "records"], labels={"failure_fraction": "Historical failure fraction"}), use_container_width=True)
    st.caption("Historical descriptive data only; sparse daily denominators can make rates noisy.")
    measure = st.selectbox("Compare partition distributions", dictionary.measurement.tolist())
    distributions = csv("partition_distributions.csv")
    st.dataframe(distributions[distributions.measurement == measure], hide_index=True, width="stretch")
    st.download_button("Download measurement dictionary", dictionary.to_csv(index=False), "data_dictionary.csv", "text/csv")
    with st.expander("Preprocessing, tuning and frozen run"):
        st.write("Training-only filter: remove >80% missing or constant columns; median imputation + missingness indicators; scaling for logistic regression. No ID, timestamp or label predictors. Three expanding training folds where both classes exist; defaults if fewer than two valid folds. No test-driven tuning.")
        st.json(run)
    with st.expander("Original sources, attribution and checksums"):
        st.json(js("provenance.json"))
    with st.expander("Summary evaluation"):
        if (ARTIFACTS / "summary_evaluation.json").exists():
            st.json(js("summary_evaluation.json"))
        else:
            st.info("Summary fixture evaluation has not yet been run.")
    st.subheader("Limitations")
    for limitation in LIMITATIONS:
        st.write("• " + limitation)
    st.write("Industry connection: Seagate's historical Project Athena case study describes wafer-defect inspection for HDD read/write-head manufacturing. This project transfers the human-review workflow, not its images or internal technology.")
    st.link_button("Seagate Project Athena case study (2019)", "https://www.seagate.com/www-content/enterprise-storage/it-4-0/images/cs595-1-1901-seagate-athena.pdf")
