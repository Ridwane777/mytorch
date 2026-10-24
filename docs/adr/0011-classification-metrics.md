# ADR-0011: Classification metrics in the library, conventions and export formats

- **Status**: Proposed
- **Date**: 2026-10-24
- **Deciders**: Ridwane777 (awaiting Meric-BG, since `Network.hpp` changes)
- **Issue**: #11

## Context

The subject values evaluation with several metrics, and accuracy alone hides what matters here: the
dataset is imbalanced (300,000 `Nothing` against 79,424 `Checkmate Black`), so a network that
always answers `Nothing` already reaches 41 % accuracy while finding no check at all. On the
overfitting benchmark (#7, ADR-0010) accuracy also barely moved while the loss changed a lot.
Issue #11 asks for accuracy, per-class and macro precision, recall and F1, the confusion matrix,
log-loss, top-k, ROC AUC, calibration (reliability bins, ECE) and an export of the training
history. Several conventions have to be fixed so that results are reproducible and comparable with
scikit-learn (#43).

## Options considered

**Where do the metrics live?**

| Option | Pros | Cons |
| --- | --- | --- |
| **In the library (`include/my_torch/Metrics.hpp`), on probability and target matrices** | Reusable for XOR, the benchmarks and the analyzer; knows nothing about chess | The analyzer maps class indices to names itself |
| In the analyzer | Close to the chess labels | Not usable by the library tests and benchmarks |

**Conventions**

| Question | Chosen | Rejected |
| --- | --- | --- |
| Predicted class | Argmax, ties to the lowest index; a single output column is binary with threshold 0.5 | Random tie-break: not reproducible |
| Precision of a class never predicted (0/0) | 0, and the report keeps the `predicted` count so the case is visible (scikit-learn's default) | NaN: breaks the macro averages |
| AUC with tied scores | Average ranks, a tie counts 1/2; O(n log n) | All pairs: O(n²), unusable on 700,000 positions |
| AUC of a class always or never expected | NaN, written `null` in JSON | 0 or 0.5: a made-up value |
| Log-loss when p = 0 | p bounded below by 1e-15 | Infinite value |

**What `fit` computes each epoch**

| Option | Pros | Cons |
| --- | --- | --- |
| **Training and validation accuracy only, added to `EpochStats`** | Almost free: the predictions are already computed | The full report is a separate call |
| The full report every epoch | Everything in the history | Sorting for the AUC every epoch on the full dataset |

**Export**

| Option | Pros | Cons |
| --- | --- | --- |
| **History as CSV, report as JSON, both written by hand** | CSV opens in any plotting tool; JSON keeps the nested report; no dependency | A small hand-written writer to maintain |
| A JSON library | Robust | A dependency for a few lines of output |

## Decision

`Metrics.hpp` provides `classesOf`, `confusion`, `report` (accuracy, per-class precision, recall, F1,
support and predicted counts, macro and support-weighted averages, log-loss), `topKAccuracy`,
`rocAuc` (one-vs-rest, average ranks), `calibration` (equal-width bins of the top probability and
the expected calibration error), `writeHistoryCsv` and `writeReportJson`, with the conventions
above. `EpochStats` gains `trainAccuracy`, measured on the batches as they are trained, and
`validationAccuracy`.

## Consequences

- ✅ Every metric is tested on cases computed by hand: a 3-class confusion matrix with all
  per-class, macro and weighted values, the "lazy network" (90 % accuracy, recall 0 on the rare
  class), AUC with ties, a perfectly calibrated and an overconfident model.
- ✅ The rank-based AUC matches the pairwise definition on 200 random rows with many ties (tested);
  swapping precision and recall, or ignoring ties in the AUC, makes the tests fail.
- ✅ Same conventions as scikit-learn for 0/0, so the comparison (#43) is like for like.
- ❌ `Network.hpp` changes again (two fields in `EpochStats`).
- ❌ Training accuracy is measured while training: with dropout it is the accuracy of the dropped
  network, and the weights move between batches. Validation accuracy is the reliable one.
- ❌ Accuracy fields are meaningless for regression targets; they are filled anyway.
- ❌ Calibration uses equal-width bins only; with few examples, many bins are empty.
- Follow-up: the analyzer (#15) writes the report with the chess class names; the comparison report
  (#44) reads the JSON exports.
