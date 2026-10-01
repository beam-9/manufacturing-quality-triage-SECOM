# Frozen results — run 9ab3115398f717e5

These values come from `artifacts/evaluation.json` and the row-level `predictions.csv`, checked independently in the tests. They are historical observations from this one frozen experiment, not performance guarantees.

## Held-out test: 314 units, 17 failures, 63 reviewed

| Model | Failures captured | Failures missed | Passing units flagged | Precision | Recall | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| Logistic (queue) | 7 | 10 | 56 | 11.11% | 41.18% | 0.0861 |
| Random forest | 3 | 14 | 60 | 4.76% | 17.65% | 0.0681 |
| Dummy prior / ID ties | 3 | 14 | 60 | 4.76% | 17.65% | 0.0541 |

Random review at identical capacity has **3.41 expected captures** (`63 * 17 / 314`). Logistic triage captured roughly twice this expectation in this particular test window. This is not statistical proof of durable improvement: the cohort is small and the windows disagree.

Logistic approximate 95% stratified-bootstrap intervals: precision 4.76%–15.87%, recall 17.65%–58.82%, average precision 0.0627–0.1465. Resamples condition on observed prevalence; temporal dependence can make these optimistic. No capacity was changed after seeing test results.

## The important unfavorable result

Validation contains **313 units and 11 failures**. Logistic triage captured **zero** failures in 63 reviewed units; the random forest captured three. Logistic validation average precision was 0.0321, below the 0.0351 failure prevalence. This indicates unstable generalization and is disclosed in the dashboard, rather than hidden behind the better test result.

The next legitimate research step is a new development study of temporal shifts, robust transformations and missingness ablations using only training/development data, followed by a new untouched external evaluation. The current test is already observed and must not be reused as an untouched holdout for a tuned v2.

## Three defensible findings

1. **Operational workload matters:** catching seven failures involved reviewing 63 units and flagging 56 passes. A good-looking score alone would hide that burden.
2. **Time-aware evaluation changes the story:** the model did not succeed consistently across validation and test. Failure prevalence also changed from 76/940 in training to 11/313 in validation and 17/314 in test; this is descriptive evidence of different windows, not proof of a particular process change.
3. **Extreme and missing inputs affect explanations:** some test measurements are far outside training references, producing saturated scores. Missingness indicators also contribute to predictions. The prototype surfaces both facts for review without claiming a physical manufacturing cause.

## Data-quality receipt

1,567 records × 590 measurements; 104 failures; approximately 4.54% missing cells; 116 constant source columns; no exact duplicate rows or conflicting duplicate exclusions in this source. Training has 940 records, validation 313, test 314. Chronological coverage is July–October 2008; source timezone is unspecified. Preprocessing decisions use training data alone. Three valid expanding-window training folds were available.

