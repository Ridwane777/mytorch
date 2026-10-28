# ADR-0026: Published scores computed by the library

- **Status**: Accepted (2026-10-28)
- **Date**: 2026-10-27
- **Deciders**: Ridwane777

## Context
The library has had tested metrics since ADR-0011 (`my_torch::report`), but they take
probabilities, and the scripts only have files of labels (`my_torch_analyzer --predict` writes one
label per line). So five Python scripts recomputed accuracy, precision, recall and F1 on their own:
`evaluate_predictions.py` (the final network, ADR-0022), `score_encodings.py`,
`score_imbalance.py`, `score_augmentation.py` and `bonus/scripts/score_predictions.py`. The most
important numbers of the project (96.2 % accuracy, macro F1 0.948) therefore came from untested
code, and nothing checked that it agreed with the library.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| Keep the Python scripts | Nothing to do | Two implementations of every metric; the published ones are not the tested ones |
| A `--score` mode in `my_torch_analyzer` | No extra tool | The command line is fixed by the subject (ADR-0014): adding a mode risks the autograder |
| **A small C++ tool, `scripts/score.cpp`, calling the library (chosen)** | One implementation, the tested one; class names read from the network file, so it also serves the six-class bonus; JSON output that the scripts read | One more tool to build in the scripts; the library's API grows |

## Decision
The library gains three functions in `Metrics.hpp`, on the model of scikit-learn's
`confusion_matrix` and `classification_report`: `confusion(expected, predicted, classes)` from class
indices, `report(ConfusionMatrix)` for every count-based metric, and `merge(confusion, groupOf,
groups)` to merge classes into families. The probability version of `report` now goes through
`report(ConfusionMatrix)`, so precision, recall and F1 have a single implementation.
`scripts/score.cpp` (`score NETWORK EXPECTED PREDICTIONS OUTPUT_DIRECTORY [--families]`) writes
`scores.txt`, `confusion.csv`, `report.json` and, with `--families`, `families.json`.
`train_final.sh`, the three benchmarks and the stalemate bonus call it; the Python scripts only read
`report.json` and draw (`evaluate_predictions.py` becomes `plot_run.py`). This replaces the scoring
step described in ADR-0022, which is otherwise unchanged.

## Consequences
- ✅ Every published score is computed by the tested library code.
- ✅ Checked on the final network's 71,555 test predictions: `scores.txt` is byte-identical to the
  one of the Python script, and `confusion.csv` has the same content (Python's `csv` module wrote
  Windows line endings, the tool writes `\n`). The numbers already published do not change.
- ✅ A unit test proves that `report` from probabilities and from a confusion matrix give the same
  numbers bit for bit.
- ❌ The predictions of the earlier benchmarks were temporary files and were not kept, so their
  rows could not be re-scored; they stay valid because the formulas are the same.
- ❌ One difference in principle: the old `evaluate_predictions.py` averaged the macro F1 over the
  labels present in the expected or predicted files, the library over every class of the network.
  They agree whenever every class appears, as in all our runs; for a class absent from both, the
  library counts an F1 of 0, like scikit-learn with an explicit list of labels.
- ❌ `Metrics.hpp`, a public header, changes: the review of both members is needed (AGENTS.md).
