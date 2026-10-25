# Symmetry augmentation: with and without

Measurement for issue #38: does training on the mirror and the colour-swapped flip of every
position help? Related decision: [ADR-0019](../adr/0019-symmetry-augmentation.md) (Proposed).

**Results: not yet measured.** The real dataset was not available when the method was written. The
pipeline was run end to end on generated positions only, to check that it works; nothing from that
run is reported here. This document is completed when the command below has run on the dataset.

## Method

1. **Draw**: `TRAIN + TEST` positions (default 20,000 + 5,000) drawn without replacement from all
   `dataset/*/*.txt` with `random.Random(SEED).sample` (default seed 42). The first `TRAIN` go to
   `train.txt`, the last `TEST` to `test.txt`. **The split is made before any augmentation**: no
   mirror or flip of a test position is ever built from a training position.
2. **Augmentation** (`scripts/augment.cpp --all`): `train_augmented.txt` holds, for each training
   position, the original, its mirror, its flip and its mirror + flip (4 × `TRAIN` positions).
   `test.txt` is **never** augmented.
3. **Two arms**, same configuration [`configs/augmentation.conf`](../../configs/augmentation.conf)
   (768 → 128 → 5, ReLU, Softmax, cross-entropy, Adam 0.001, batches of 64, 20 epochs, **no early
   stopping and `validation=0`**) and the same initial weights (`init seed`), `RUNS` networks per
   arm (default 3, different initial weights):
   - `base`: `my_torch_analyzer --train` on `train.txt`;
   - `augmented`: `my_torch_analyzer --train` on `train_augmented.txt`.
   Validation is off on purpose: with a random hold-out, a validation position would have symmetric
   twins in the training part (ADR-0019). The number of epochs is fixed, so the augmented arm makes
   four times more updates; the comparison cannot separate "more distinct positions" from "more
   steps".
4. **Scoring** (`scripts/score_augmentation.py`): accuracy and recall of the five classes on
   `test.txt` (predicted by `my_torch_analyzer --predict` on the boards only), training time and peak
   memory, one row per network in `augmentation_runs.csv`.
5. **Graph** (`scripts/plot_augmentation.py`): grouped bars, median of the runs with the individual
   runs as dots.

Also written: `augmentation_split.txt` (sizes of the files and how many distinct test positions,
compared on board, side, castling and en passant, also appear in the training files: expected
`0`; a non-zero value is a leak to read before trusting the result) and `augmentation_logs/` (the
epoch lines of every training).

## Reproduce

```
DATASET=dataset TRAIN=20000 TEST=5000 SEED=42 sh scripts/bench_augmentation.sh docs/performance/data
```

Requires the dataset (`dataset.zip`, given by the school) extracted at the root, g++, Python 3 and
matplotlib. Without the dataset the script prints an error on stderr and exits 1. Expected outputs in
`docs/performance/data/`: `augmentation_runs.csv` (6 rows with the default `RUNS`),
`augmentation_split.txt`, `augmentation_logs/`, `augmentation.png`.

The `--train` memory limit (ADR-0014) applies: the augmented arm encodes four times as many
positions as the base arm.

## Checking the symmetries themselves

The labels written by the tool are checked against the rule-based oracle, independently of the unit
tests:

```
sh bonus/scripts/check_symmetries.sh 2000      # 2,000 generated positions, or a chess file
```

It builds the tool and the oracle, augments, and exits non-zero unless the oracle reports
`disagree: 0` on every original, mirrored and flipped position.

## Results

Not yet measured.

## Limits

- One draw (one seed) and 3 initial weights per arm: no confidence interval beyond the spread of the
  runs.
- 20,000 training positions only, because `--train` keeps every encoded position in memory (#17).
- The baseline encoding `fen-onehot-768` is the only one used; the encodings of #14 may react
  differently to the symmetries.
- The mirror clears the castling rights (ADR-0019).
