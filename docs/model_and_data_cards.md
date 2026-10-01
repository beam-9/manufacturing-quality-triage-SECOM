# Model and data cards

## Intended use

A public portfolio prototype for retrospective semiconductor-quality review. A quality engineer can inspect ranked units and a draft brief, then decide what evidence needs checking. It is not an autonomous agent, a physical root-cause system, a manufacturing control system, or validated predictive-maintenance software.

## Data card

Source: [UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom), McCann and Johnston (2008), CC BY 4.0. Each observation is a production entity with anonymous measurements and an in-house pass/fail test timestamp. Raw files contain 1,567 rows and 590 measurements; 104 failures. Publisher metadata lists 591; raw shape controls this implementation. Timezone, physical units, measurement semantics, and pre-test feature timing are unknown.

Missing values remain missing in source exports. Training statistics alone determine retained columns, imputation and scaling. Exact duplicate conflicts are excluded; later-partition duplicates of earlier records are excluded from evaluation. Raw and eligible counts/exclusions are inspectable in the audit and manifest. Outliers are not silently deleted. All dates are historical source timestamps; download time is not data freshness.

## Model card

Queue model: regularized, class-weighted logistic regression. Model-family choice is predefined for explainability, not selected using test results. A random forest is a published comparator even if it performs better. Dummy prior scores provide a constant-score baseline; record-ID tie-breaking must not be mistaken for a meaningful dummy ranking.

Approximately 60/20/20 chronological partitions, with equal timestamps kept together. Three expanding-window folds inside training tune average precision. Invalid single-class folds are omitted; fewer than two valid folds triggers documented defaults. No ID, timestamp or target predictors. No resampling/SMOTE or test-informed feature selection. Filtering removes >80% missing and constant training features; median imputation includes missingness indicators, and logistic values/indicators are standardized.

At the predefined 20% capacity, the top `ceil(0.2*N)` units are reviewed. Precision, recall, captured/missed failures, false alarms and average precision have explicit denominators. Random expected captures equal `reviewed * observed failure prevalence`; this is an expectation, not a trial. Approximate 95% stratified-bootstrap intervals condition on the partition's prevalence and may be optimistic under temporal dependence.

Scores are not calibrated probabilities. Coefficient × transformed-value contributions sum with the intercept to the log-odds; all-feature reconstruction is tested. A displayed top-contribution subset does not itself sum to the full score. Global coefficients and validation permutation changes are model associations; correlated features can mask/reallocate importance. The bounded permutation study is not an exhaustive causal screen.

## LLM card

Local Ollama `qwen3:8b`, temperature zero, thinking disabled, schema-constrained response, context 4,096 and generation cap 600 tokens. Requests have a 90-second read timeout, one repair attempt, then deterministic fallback. Timeout bounds individual requests, not the entire two-attempt workflow. Actual runtime/digest/latency are retained in each generation record.

LLM input contains only computed facts, anonymous feature evidence, policy and limitations. Actual test labels are absent. Version 2.1 selects/orders finding codes and cited evidence; all cited facts must carry the required computed support signal. Python renders fixed interpretation wording and numerical facts from the sealed bundle. Earlier free-prose semantic failures motivated this stricter contract. This is evidence-grounded report composition, not unrestricted narrative generation. Human review is still needed for contextual usefulness and downstream action. No arbitrary prompts, external tools or generated code execution.

Hosted mode does not call Ollama or write SQLite. Local mode logs generation errors and review status. No private Seagate information or credentials are used. Public artifacts are derived from openly licensed SECOM data.
