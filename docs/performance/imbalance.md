# Class imbalance: none, weighting, oversampling, undersampling

Measurement for issue #16: which way of handling the imbalance of the dataset gives the best F1 per
class? Related decision: [ADR-0018](../adr/0018-class-imbalance.md) (Proposed).

**Status: results pending the real dataset.** The benchmark below has not been run on the real
dataset (`dataset/` is not extracted where this was written). The scripts were only exercised on a
small synthetic dataset to check that they run end to end; **no measurement is reported in this
document**, and none should be quoted from it.

## Method

1. **One seeded sample** of `TRAIN + TEST` positions (default 20,000 + 5,000) drawn without
   replacement from the three dataset folders with Python's `random.Random(SEED).sample`.
2. **Test set** = the last `TEST` positions, **in their natural distribution and never resampled**,
   identical for the four options. The first `TRAIN` positions are the training sample.
3. **Four training files / losses**, derived from the training sample by
   `scripts/resample_imbalance.py`:

   | Option | Training file | Loss |
   | --- | --- | --- |
   | `none` | the sample unchanged | cross-entropy |
   | `weighted` | the sample unchanged | cross-entropy with `weights=` inverse class frequency, normalised to mean 1, in the order of the labels of the configuration |
   | `oversample` | minority classes duplicated with replacement up to the majority count | cross-entropy |
   | `undersample` | every class drawn without replacement down to the minority count | cross-entropy |

   The class counts before and after are written to `imbalance_logs/<option>_resample.log`, the
   weights to `imbalance_logs/weighted_resample.log`.
4. **Network and training** (`configs/imbalance/*.conf`, the same for the four options): one-hot 768
   → 128 → 5, ReLU, Softmax, cross-entropy, Adam 0.001, L2 0.0003, 20 epochs, batches of 64,
   shuffling seed 42, **`validation=0 patience=0`** (no early stopping: a validation split of an
   oversampled file would hold duplicates of training positions, see the ADR).
5. **Three networks per option**, from three different initial weights (`init seed` 1 to 3, the
   same for every option), trained with `my_torch_analyzer --train`, timed with `/usr/bin/time`
   (wall time, peak memory).
6. **Scoring** (`scripts/score_imbalance.py`) on the test set: F1 of each of the five labels, F1 of
   the three families (`Nothing`, `Check` = both colours, `Checkmate` = both colours), macro F1
   over the five labels, accuracy. A class never predicted gets an F1 of 0.
7. **Graphs** (`scripts/plot_imbalance.py`): mean of the three runs with the minimum-maximum range
   and the individual runs.

- **Machine**: to be filled in when the benchmark is run (processor, memory, system, compiler).
- **Dependencies**: Python 3 with matplotlib for the graphs; `/usr/bin/time`.
- **Environment variables** of the script: `DATASET` (default `dataset`), `TRAIN` (20000), `TEST`
  (5000), `SEED` (1: the draw and the resampling), `EPOCHS` (default: the 20 of the
  configurations; a smaller value is for a quick check only).

## Reproduce

With the dataset extracted at the root of the repository:

```sh
sh scripts/bench_imbalance.sh
```

An optional argument gives the output directory (default `docs/performance/data`). Without the
dataset the script prints an explanation on stderr and exits with status 1.

| Output | Content |
| --- | --- |
| `data/imbalance_runs.csv` | one row per network: option, init seed, training size, accuracy, macro F1, F1 of the five labels, F1 of the three families, training seconds, peak memory (KB) |
| `data/imbalance_f1.png` | F1 per class and macro F1, for each option |
| `data/imbalance_family_f1.png` | F1 per family, for each option |
| `data/imbalance_logs/` | training logs, class counts before and after resampling, weights, test class counts |

## Results

To be measured on the real dataset. The table and the graphs will be added here, and the decision
taken in [ADR-0018](../adr/0018-class-imbalance.md).

| Option | F1 Nothing | F1 Check White | F1 Check Black | F1 Checkmate White | F1 Checkmate Black | Macro F1 |
| --- | --- | --- | --- | --- | --- | --- |
| None | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured |
| Loss weighting | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured |
| Oversampling | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured |
| Undersampling | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured |

## Limits

- One sample and three initial weights: no confidence interval; the draw can be changed with
  `SEED`.
- No early stopping in the comparison (ADR-0018); the training sample is 20,000 positions because
  `--train` keeps every encoded position in memory.
- Resampling is done by the script, not by the network file: a network trained on a resampled file
  is reproduced only with `scripts/bench_imbalance.sh` (or `resample_imbalance.py`).
- One-hot 768 encoding, the one the analyzer knows; another encoding changes the numbers.
