# ADR-0021: Training data as a batch source, positions encoded on demand

- **Status**: Accepted
- **Date**: 2026-10-26
- **Deciders**: Ridwane777
- **Issue**: #17

## Context

`Network::fit` took its examples as two matrices (ADR-0009), and `my_torch_analyzer --train`
encoded every position of the training file into one matrix before training (ADR-0014). The final
network (#17) trains on the whole dataset with `fen-attacks-901` (ADR-0016), possibly augmented four
times (ADR-0019): 723,019 positions × 901 inputs × 8 bytes is 5.2 GB for the inputs alone, held
twice by the old code, and four times more with augmentation, on machines with 14 to 15 GB. A batch
only ever uses 64 positions at a time.

## Options considered

| Option | Pros | Cons |
| --- | --- | --- |
| **A. `fit` takes a `DataSource` that builds each batch on demand; the analyzer keeps the positions and encodes a batch when it is asked for** | Memory no longer grows with the encoded width; works with every encoding, present and future; nothing else changes (files, commands, results) | Positions are encoded again at every epoch; `fit` gains an interface |
| B. Store the inputs as `float` | Simple | Halves memory only: still about 10 GB with augmentation; against ADR-0003 (`double`) |
| C. Store the inputs as sparse (only the non-zero entries) | Very compact | Tied to sparse encodings (`fen-attacks-901` is denser); needs a sparse product in `Dense` |
| D. Train on part of the dataset | Nothing to write | Throws away data; the first measurement shows how much data matters |

## Decision

`my_torch` gains the generic `DataSource` interface (`size`, `inputWidth`, `targetWidth`, and `fill`,
which writes the requested examples into zeroed matrices) and `MatrixSource`, which wraps a
`Dataset`. `fit` draws every batch from a source; the old `fit(Dataset…)` remains and goes through
`MatrixSource`. The validation set is evaluated 4,096 examples at a time (`EVALUATION_CHUNK`), each
chunk's mean loss weighted by its size. The analyzer's `PositionSource` keeps the parsed positions,
their class and the encoding, and the training and validation parts are two lists of indices into
the same positions.

## Consequences

- ✅ Peak memory of one epoch on 100,000 positions: 1,338 MB before, 109 MB after; the whole dataset
  with `fen-attacks-901` trains in 198 MB, 180 s per epoch (`docs/performance/11-training-memory.md`).
- ✅ Same results: the networks trained before and after are byte-identical, and a source that
  builds its rows from a formula trains bit for bit like the same rows in matrices (tested).
- ✅ The library stays generic: `DataSource` knows nothing about chess, and `docs/library.md` shows
  how another project implements one.
- ❌ Each epoch encodes every position again; the measured epoch time did not grow (21.4 s against
  22.7 s on 100,000 positions), but a slow encoding would pay this cost every epoch.
- ❌ A validation set above 4,096 examples is evaluated in chunks, so its reported loss can differ
  from a single-block computation in the last bits; with early stopping, that could in principle
  move the stopping epoch.
- ❌ `Network.hpp`, the shared contract, changes again (one interface, one class, one overload).
- Follow-up: the training pipeline of #17 uses this source for the full and augmented dataset.
