# ADR-0032: Checkpoints, saving the training state and resuming identically

- **Status**: Proposed
- **Date**: 2026-10-28
- **Deciders**: Ridwane777 (issue #35), review by Meric-BG (the change touches `include/my_torch/`)

## Context
A full training of the chess network lasts 1 h 45 for the final network (6,285 s, 14 epochs) and
longer for the candidate of ADR-0030. A stop (Ctrl+C, a crash, a closed lid, the non-finite stop of
ADR-0031) lost the whole run. Issue #35 asks to save the optimizer state, the epoch, the seed and the
history, and to resume so that an interrupted then resumed training gives the same weights as a
continuous one, within 1e-6.

The weights alone are not enough to resume identically: Adam keeps two running averages and a step
count (SGD a velocity, RMSProp and Adagrad their sums), the shuffling generator decides the next
epoch's batches, each dropout layer has its own generator, the scheduler reads the epoch number, and
early stopping keeps the best loss, the epochs since it and the best weights.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| Save the weights only (the network file) | Exists already | Adam restarts cold, other batches and masks: not the same training |
| **The whole state, at the end of each epoch (chosen)** | Simple: no partial sums to save; the loop restarts at an epoch boundary | An epoch cut in the middle is run again (about 7.5 minutes for the final network) |
| The whole state after every N batches | Loses less on a stop | Batch index, partial loss and accuracy sums, the epoch's order to save; more code and more writes |
| **Text, 17 significant digits, checksum (chosen)** | Every double reads back exactly; same style as the network file (ADR-0012); readable | 16.4 MB for the final architecture, larger than a binary file |
| Binary file | Smaller and faster | Byte order and format to define; unreadable when debugging |
| **Resume automatically when the file exists (chosen)** | Rerunning the same command resumes, as the hyperparameter search does (ADR-0028) | A forgotten file would resume an old run: guarded by a fingerprint, see below |
| An explicit `--resume` option | No surprise | Two commands to remember; a script must know whether it is the first run |
| Keep one checkpoint per epoch | Can go back to any epoch | Disk use grows with the epochs; not asked |

## Decision
`fit` takes an optional checkpoint path. At the end of each epoch it writes the whole state: the
weights, the optimizer's memory (`Optimizer::saveState`, implemented by the five optimizers), the
layers' state (`Layer::saveState`, the generator of `Dropout`), the shuffling generator (the text
form of `std::mt19937_64`, which the C++ standard specifies, so it is portable), the history and the
early stopping state. The file is written to `path.tmp` then renamed, so a crash while writing keeps
the previous checkpoint; it ends with an FNV-1a checksum. When the file exists at the start, `fit`
restores everything and continues after the last saved epoch.

A fingerprint of the training is stored and checked: every `train` setting except `epochs`, the
optimizer type, the layer types and shapes, and the sizes of the data. A checkpoint of another
training is refused with an error. `epochs` is left out so that a finished training can be extended
by raising it.

The analyzer gets `--checkpoint FILE` in train mode (`docs/usage.md`). It extends ADR-0014 without
changing it: the usage line of the subject is unchanged and the option is listed apart in `--help`.

## Consequences
- ✅ Interrupted then resumed equals continuous, **bit for bit** (stricter than the 1e-6 asked):
  tested with each of the five optimizers, a dropout layer and a step-decay scheduler, the training
  interrupted in the middle of an epoch by a data source that fails on purpose, then resumed with
  new objects (`tests/my_torch/test_checkpoint.cpp`); also with early stopping, interrupted before
  and after the best epoch. Corrupted, truncated and foreign checkpoints are refused; the tests also
  pass under the sanitizers.
- ✅ Cheap: on the final architecture (901 → 256 → 128 → 5, Adam, no early stopping), one checkpoint
  is 16.4 MB and costs about 0.24 s, measured on an idle laptop (5 runs,
  `docs/performance/16-benchmarks.md`; a first measurement under load gave 0.3 to 0.45 s); an epoch
  of the full training takes about 450 s, so less than 0.1 %.
- ❌ An epoch interrupted in the middle is run again.
- ❌ The data itself is not in the fingerprint, only its size: a different file of the same size
  would be resumed. Hashing it would cost a pass over the 2.3 million positions at every start.
- ❌ One more file per training, not deleted at the end so that rerunning does not retrain; with
  early stopping it also holds the best weights, about a third more (estimated, not measured).
- ❌ `Optimizer` gets two pure virtual functions: an optimizer written outside the library must now
  implement them.
- Follow-up: save within an epoch if epochs grow past an hour.

## Note (2026-10-29)
Format version 2: the checkpoint also stores the layers' state at the best epoch (`best_layers`),
restored with the best weights, after BatchNorm (ADR-0033) showed that restoring the best weights
with the running averages of the last epoch breaks a network. Version 1 files are refused.
