# Model improvement study — not a replacement benchmark

Hosting is paused. V1 was committed and pushed first as `c0852c1`. Its evaluation and dashboard queue remain frozen. This development study responds to the weak validation capture without quietly replacing the original results.

## Protocol

The original validation results and final test results have already been viewed. Consequently, the original validation partition is explicitly a **development set**, and another look at the original test cannot establish independent generalization. Neither experiment scores the test partition. A new external/future holdout is required before promoting a revised model.

Stage one compares 24 logistic pipelines: 20/50/100/all training-selected features, C=0.01/0.1/1, and either standard scaling or training-fitted 1st–99th percentile clipping plus robust scaling. All imputation, filtering, supervised feature selection and clipping are fitted inside each training fold. Raw source observations remain unchanged. Some clipped inputs become constant; scikit-learn reports constant-feature/undefined F-statistic warnings, and SelectKBest ranks undefined scores last. This is recorded, not evidence that those measurements have known physical meaning.

Stage two was motivated by stage one's weak development capture: eight histogram gradient-boosting settings (7/15 leaves, L2=1/10, minimum leaf size=10/20). It handles missing values natively and uses no random early-stopping split. Both stages select parameters using mean average precision across three expanding-window folds inside the original training partition, **not development performance**. Grid construction, candidate/fold metrics, chosen parameters and development predictions are committed for inspection.

## Measured results at 20% development review capacity

All rows refer to the same 313 development units, including 11 failures; 63 units are reviewed.

| Model | Failures captured | Failures missed | Passing units flagged | Average precision |
|---|---:|---:|---:|---:|
| Frozen v1 logistic | 0 | 11 | 63 | 0.0321 |
| Frozen v1 random forest | 3 | 8 | 60 | 0.0521 |
| Robust, 20-feature logistic | 2 | 9 | 61 | 0.0547 |
| Histogram gradient boosting | 2 | 9 | 61 | 0.0457 |

Random selection at the same capacity captures **2.21 failures in expectation**. Both new selected candidates capture 2: neither demonstrates an operationally convincing improvement at this capacity. The robust logistic raises average precision and capture relative to v1 logistic, but does not outperform the frozen forest's development capture. Do not describe this as a validated better failure detector.

The robust logistic captures 1/11 at 10% capacity, 2/11 at 20%, 3/11 at 30%, and 7/11 at 50%. Reviewing every unit captures all historical failures by construction, but provides no workload reduction. These are workload tradeoffs, not recommended factory thresholds.

## Interpretation and next decision

There are 590 raw measurements but only 76 training failures and 11 development failures. Failure prevalence also changes from 8.09% in training to 3.51% in development. This makes model selection unstable; temporal drift is plausible but not established as the sole explanation. Anonymous measurements prevent diagnosing the physical mechanism behind errors. More complex modeling alone did not fix the observed weakness.

Keep v1 as an honest retrospective workflow demonstration, **not a dependable failure-screening system**. Next model work should first investigate training-window stability, missingness shifts and measurement-distribution shifts, then predeclare a small study and evaluate against genuinely new manufacturing observations. Avoid searching repeatedly until this small development set looks good. No v2 model is promoted into the queue or explanation/LLM workflow here.

## Reproduce

With the official raw files downloaded and the frozen split manifest present:

```bash
triage improve
triage compare-nonlinear
```

Each command refuses to overwrite its existing results. Use a separate checkout with these exact study outputs backed up to rerun. Reports: `artifacts/v2_development/study.json` and `nonlinear.json`; predictions in the same folder. The dashboard still loads only the original v1 artifacts.
