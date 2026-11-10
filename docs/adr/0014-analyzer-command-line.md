# ADR-0014: Analyzer command line, encoding table, label matching and validation hold-out

- **Status**: Accepted
- **Date**: 2026-10-24
- **Deciders**: Ridwane777
- **Issue**: #15

## Context

`my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE` is the program the
evaluation harness runs. In `--predict` it must print one prediction per position, in file order; in
`--train` it trains the network of LOADFILE and saves it to SAVEFILE or back to LOADFILE. The
specification's own examples show three networks answering at three levels of detail (`Check`,
`Checkmate`, `Checkmate Black`). The network file (ADR-0012) already stores the label names, the
name of the input encoding and the training settings. Four points remain: how the analyzer finds
the encoding, how expected outputs map to a network with fewer labels, where the validation set
for early stopping comes from, and how large files fit in memory.

## Options considered

**Input encoding**

| Option | Pros | Cons |
| --- | --- | --- |
| **A table of encodings looked up by the name stored in the network file** | The analyzer depends on no particular encoding; #14 adds encodings without touching the command line; an unknown name is a clear error | One more indirection |
| One encoding hard-wired in the analyzer | Simplest | A network trained with another encoding would give wrong answers silently |

The baseline `fen-onehot-768` (12 piece planes of 64 squares) is written here so the analyzer works
end to end; the variants and their comparison stay in #14.

**Expected outputs and network labels**

| Option | Pros | Cons |
| --- | --- | --- |
| **Exact label, else the first word of the expected output** | The 5-class dataset trains networks with 3 classes (`Check White` → `Check`) or 5 classes, as in the specification's examples | A rule to know; an output matching nothing is an error |
| Exact label only | Strict | A network without colors cannot be trained on the dataset |

**Validation set for early stopping**

| Option | Pros | Cons |
| --- | --- | --- |
| **A seeded fraction of CHESSFILE, set by `validation=` in the network's `train` line** | No extra command-line option (the requirements fix them); reproducible; stored with the other training settings | Not stratified by class; the held-out positions are not trained on |
| A second file on the command line | Explicit | Not in the required interface |

**Memory**

| Option | Pros | Cons |
| --- | --- | --- |
| **`--predict` encodes and predicts 4,096 positions at a time** | Memory bounded whatever the file size, order kept | — |
| Encode the whole file at once | Simplest | 723,019 positions × 768 inputs × 8 bytes ≈ 4.4 GB |

`--train` still encodes the whole file, because `fit` takes its data as one matrix.

## Decision

The analyzer parses its arguments strictly (exactly one mode, `--save` only with `--train` and with
one file, two positional files, no unknown option), reads and checks every input before printing
or writing anything, and finds the encoding by its name in the network file, checking that its
width matches. `--predict` prints the label of the most probable output, one line per position,
4,096 positions at a time. `--train` maps each expected output to a label (exact, else first word),
holds out `validation × count` positions chosen by the training seed when `validation > 0` (required
when `patience > 0`), builds the loss, optimizer and scheduler from the network file, trains, prints
one line per epoch, and saves to SAVEFILE or LOADFILE.

## Consequences

- ✅ End-to-end tests: predictions in file order, optional expected outputs ignored, more positions
  than one chunk, train then predict recovers the answers, a network without colors answers without
  colors, `--save` leaves LOADFILE untouched and no `--save` overwrites it, early stopping with the
  held-out fraction, and 17 error cases exiting 84; a malformed line prints nothing at all.
- ✅ By hand on 2,700 dataset positions: training in 4.8 s, prediction of all of them in 0.2 s.
- ❌ `--train` keeps every encoded position in memory: the full dataset would need about 4.4 GB.
  To solve in #17 (encode each batch on the fly, or a compact input type), with its own ADR.
- ❌ The epoch lines are printed after training, not live: `fit` has no progress callback.
- ❌ The validation hold-out is random, not stratified: with few positions a rare class may be
  missing from it.
- Follow-up: #14 registers its encodings in the table and compares them; #17 trains the final
  network and solves the memory limit.
