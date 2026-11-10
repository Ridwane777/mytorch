# ADR-0022: Untouched validation set at the end of the training file, one-command training pipeline

- **Status**: Accepted
- **Date**: 2026-10-26
- **Deciders**: Ridwane777
- **Issue**: #17

## Context

The final network trains on the augmented training split (ADR-0019) and needs early stopping, hence
a validation set. `my_torch_analyzer --train` takes one file, and its only validation option so far
was `validation=<fraction>`, positions drawn at random from that file (ADR-0014). On an augmented
file this leaks: the mirror or the flip of a validation position sits in the training part, so the
validation loss looks better than it is and early stopping stops at the wrong time (the ❌ of
ADR-0019). The dataset split (ADR-0017) already provides a separate validation set. The specification also
requires every training to be reproducible from the scripts and configurations kept in the
repository.

## Options considered

**Validation set of an augmented training run**

| Option | Pros | Cons |
| --- | --- | --- |
| **A. `validation_tail=N`: the last N lines of the file are the validation set, in order** | No leak: the untouched validation split is appended after the augmented training lines; early stopping stays available; the required command line is unchanged | One more key in the `train` line of the network file |
| B. No validation (`validation=0`) and a fixed number of epochs | Nothing to write | No early stopping; the number of epochs has to be guessed |
| C. Keep `validation=<fraction>` on the augmented file | Nothing to write | The leak described in ADR-0019 |

**Reproducing a training**

| Option | Pros | Cons |
| --- | --- | --- |
| **One script, `scripts/train_final.sh CONFIG`, from the raw dataset to the test scores, with a run card** | Anyone reruns the exact training and checks the figures; every step and seed is written down | A long shell script to maintain |
| A documented list of commands | Simple | Easy to run in another order or with other options |

## Decision

`TrainConfig::validationTail` (`validation_tail=` in the `train` line) makes the analyzer use the
last N lines of its file as the validation set and the others as the training set; giving both
`validation` and `validation_tail` is an error. `scripts/train_final.sh CONFIG [RUN]` splits the
dataset (ADR-0017), augments the training split only (ADR-0019), appends the validation split
untouched and sets `validation_tail` to its size, generates the network from CONFIG, trains it,
predicts the test split, scores it with `scripts/evaluate_predictions.py` (precision, recall and F1
per class, confusion matrix, learning curve) and writes a run card (commit, configuration checksum,
machine, compiler, sizes, times, peak memory) to `runs/<name>/`, which Git ignores.
`configs/final_network.conf` holds the configuration of the final network.

## Consequences

- ✅ The validation lines are never trained: two files that differ only in their last lines train
  byte-identical networks, and the test fails when the tail is trained on purpose (tested).
- ✅ Reproducible: two runs of the pipeline on 1 % of the data gave identical scores and
  byte-identical networks (98 s each).
- ✅ The test split is used only to score, after training; it is never seen by the network.
- ❌ One more key in the network file format; files written before it simply lack it (default 0).
- ❌ The validation set is the last lines of the file: a user who sorts or shuffles the file loses
  the separation without any error.
- ❌ `runs/` is not versioned: the trained network and its figures must be copied into the
  repository on purpose (the pushed `my_torch_network_*.nn` and `docs/performance/`).
- Follow-up: the trial runs and the final training of #17 use this pipeline.
