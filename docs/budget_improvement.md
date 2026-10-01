# Budget-aligned improvement, first implementation

This is a separate development study. The original v1 queue, evaluation, explanations and briefs remain unchanged; no model is automatically promoted and hosting remains paused.

## Changes implemented

- A training-fitted preparation module treats measurements and missingness separately. It clips/scales only measurement values, then appends unchanged binary missingness masks. Feature names survive filtering and supervised selection. Constant transformed inputs are removed using training data only.
- A shared chronological study module owns timestamp grouping, fold eligibility, candidate comparison and result recording. Both ends of each validation fold preserve full timestamp groups. Source checksums, frozen membership and record/label alignment are verified before experiments.
- Candidate selection now targets **mean recall at 20% review capacity**, with worst-fold recall and then average precision resolving ties. Candidates with convergence warnings are excluded. The final full-training fit must also converge before development scoring.
- Clipping and robust scaling vary independently, unlike the historical study. The predeclared grid contains 24 logistic recipes plus the frozen v1 logistic recipe refitted on the same training folds as a reference. It compares 20 selected features versus all retained inputs and C=0.01/0.1/1.

The CLI command is `triage improve-budget`. It writes a protocol before scoring, refuses to overwrite a study, and never scores the test partition. The original AP-selected command recipes remain historical implementations so their published results can be reproduced; the old clipped recipe still contains the documented mask-clipping flaw and must not be used as corrected preparation.

## Measured results

The selected converged candidate uses **20 features, C=0.1, standard scaling, and no clipping**. Preserving masks is a correctness improvement in the candidate family, but the chosen model does not use clipping; do not claim the clipping fix itself caused its gain.

| Same chronological training folds | Fold 1 | Fold 2 | Fold 3 |
|---|---:|---:|---:|
| Failures in review population | 26 | 8 | 10 |
| Reference v1 logistic captures | 4 | 1 | 2 |
| Selected revised logistic captures | 7 | 4 | 5 |

Mean fold recall improves from **15.96% to 42.31%**. These are model-selection results from reused training folds, not independent confirmation. Group-aware folds contain 238/237/233 review records, so these figures must not be directly compared with older 235-record fold reports. The reference and revised candidate above use identical new folds.

On the previously inspected development partition, at 20% capacity:

| Measure | Revised logistic |
|---|---:|
| Units reviewed | 63 of 313 |
| Failed units captured | 3 of 11 |
| Failed units missed | 8 |
| Passing units flagged | 60 |
| Precision | 4.76% |
| Recall | 27.27% |
| Average precision | 0.05097 |

For comparison, frozen v1 logistic captured zero development failures, the earlier robust candidate captured two, and the original forest captured three. Random review captures about 2.21 failures in expectation. **This is a promising training-study change, not a convincing new failure detector.** The revised candidate only matches the original forest's observed development capture, with very few failure examples.

The revised candidate captures 1/11 at 10% workload, 3/11 at 20%, 3/11 at 30%, and 8/11 at 50%. A larger review population is a workload change, not a model improvement.

## Verification and remaining work

Tests cover rare missingness survival, unchanged source data, training-fitted transforms, feature-name alignment, complete timestamp groups, budget-priority selection, refusal to fit test records, full synthetic study execution, exclusion of nonconverged candidates, independent recomputation of saved counts, and exact additive score reconstruction.

`artifacts/v3_budget` preserves the initial execution without convergence eligibility filtering. `artifacts/v3_budget_verified` preserves the repeat with explicit warning recording/filtering; the winner and development scores are unchanged. The repeat verified training eligibility, not a second tuning round driven by development outcomes.

Next: investigate window-level measurement/missingness shifts and feature-selection stability, then predeclare one small representation or recency study. A genuinely new compatible holdout is needed before claiming generalization. Do not wire the new pipeline into v1 explanation code: its selected feature mapping needs a compatible evaluated-run/explanation path first.
