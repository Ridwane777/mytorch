# ADR-0017: Dataset split: dedupe-aware stratified split by position key

- **Status**: Proposed
- **Date**: 2026-10-25
- **Deciders**: The pair (issue #12)
- **Issue**: #12

## Context

The dataset (723,019 lines, `docs/reading-a-fen.md`) must be cut into a training set, a validation
set and a test set that are never mixed: a network scored on positions it was trained on says
nothing about unseen ones. Two properties of the data make a plain random cut unsafe:

- The line holds the half-move and full-move clocks after the position. The same position can appear
  more than once, with different clocks or in several files, and a random line split would then put
  copies of it in training and in test.
- The labels are not balanced (`Nothing` is the largest class by far, `Checkmate Black` the
  smallest), so a plain random split can leave a small class thin in the validation or test set.

Whether the dataset really holds duplicates or contradictions is **not measured yet**: the real
`dataset/` is not available where this was written. The decision below is therefore Proposed; the
script prints these numbers (`docs/performance/dataset.md`, "pending measurement") and the pair
confirms or changes the decision when they are known.

## Options considered

| Option | Pros | Cons |
| --- | --- | --- |
| **Group lines by position key, drop exact duplicates, exclude contradictions, split each label (stratified) by group with a fixed seed** | No position can be in two sets; every class is represented in each set in the same proportion; same dataset and seed give the same files | Drops the contradictory positions; the key definition is a choice; small classes are not cut exactly in the ratios |
| Naive random split of the lines | Simplest | Duplicates leak from training into test and inflate the test score; small classes can be under-represented |
| Split by source file (`10_pieces`, `20_pieces`, `many_pieces`) | No leak between files if positions never cross files (to be checked) | The sets would differ in distribution (the files are made by piece count), so the test would measure something else; the classes are not balanced across sets |
| Keep contradictory positions as noise | Keeps every line | The same input with two answers: the network cannot be right for both, and the test score cannot be clean |
| Key including the clocks | Fewer lines considered duplicates | The same position with other clocks stays in two sets: leak, since the network input does not use the clocks |

## Decision

A position key is the first four FEN fields (board, side to move, castling, en passant); the clocks
are ignored. `scripts/dataset_split.py` groups the lines by key. Lines with the same key and the
same label keep one line (the first read, files in sorted path order); keys with several labels are
excluded from every set and listed in `contradictions.txt`. Within each of the five labels, the
sorted keys are shuffled with `random.Random(seed)` (default seed 42) and cut 80 % / 10 % / 10 %
(`--train`, `--validation`, `--test`, which must sum to 1). A key goes to exactly one set. The
files are written to `build/dataset_split/` with a manifest (seed, ratios, counts, SHA-256) and are
never committed.

## Consequences

- ✅ The test score is not inflated by copies of training positions, and every label has about the
  same share in each set.
- ✅ Reproducible: the same dataset, seed and ratios give byte-identical files (checked on a
  synthetic dataset; the manifest holds the SHA-256 of each file to compare two machines).
- ❌ The contradictory positions are lost to training: if there are many, the network loses data
  and the real dataset's label noise is hidden from the test.
- ❌ The guarantee depends on the key definition: two boards that are the same position written
  differently (for example an en passant square that is set but not capturable) count as different,
  so a leak of that kind is not detected.
- ❌ The ratios are not exact for small strata (rounding to whole positions per label), and the
  split is over the five labels only: nothing is balanced inside a label (for example by number of
  pieces).
- Follow-up: run `sh scripts/dataset_split.sh` on the real dataset, fill in
  `docs/performance/dataset.md` and switch the status to Accepted (or supersede this ADR) once the
  duplicate and contradiction counts are known; the training script (#17) and the scoring of #18
  must use these files.
