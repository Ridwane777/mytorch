# ADR-0045: Ensembles of networks in the library, in one binary file

- **Status**: Accepted (2026-11-05)
- **Date**: 2026-11-05
- **Deciders**: Ridwane777 (issue #40, assigned to Meric-BG); accepted by Ridwane777 and merged
  without a GitHub review from Meric-BG of the change to `include/my_torch/` (to be reviewed after
  the fact)

## Context
Issue #40 asks to combine several networks (mean of the probabilities, vote, stacking), to load an
ensemble in the analyzer, and to compare it with a single model. We have two trained networks that
read the same positions differently: the pushed dense network (96.15 % on the test split) and the
CNN snapshot (98.07 %). Three choices: where the code lives, how an ensemble is stored, and how the
analyzer is given one, knowing that the subject's command line
(`--predict LOADFILE CHESSFILE`) must not change and that the first defense criticized network
files kept as plain text.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| **`Ensemble` in the library, one binary file holding every member (chosen)** | Generic (any networks with the same labels, any encodings); one self-contained file to copy; signature, version and checksum like the network files (ADR-0042); the analyzer recognizes it by its first bytes, same command line | A change to the public headers (`include/my_torch/`), reviewed by both members; the members are copied into the file (2.7 MB for ours) |
| A text file listing the paths of the members and their weights | Tiny, readable, no copy of the networks | Plain text again; breaks when a member is moved or renamed; a second file format to parse |
| Several LOADFILEs on the command line | No new file | Changes the subject's command line; the combination and the weights would need options too |
| Combination in the analyzer only (`src/analyzer/`) | No change to the library's contract | The library cannot serve it to other programs (the Python bindings, ADR-0044); the generic part is chess-free and belongs to the library (ADR-0041) |

## Decision
`include/my_torch/Ensemble.hpp`: `Ensemble` holds networks with the same labels, a weight each,
and a `Combination`: `Average` (weighted mean of the probabilities), `Vote` (weighted vote, a tie
going to the class with the highest mean probability) or `Stack` (a combiner network that reads
the members' probabilities side by side). `predict` takes one input matrix per member, so members
may use different encodings; `classify` applies the tie rule. `save`/`load` use a binary file:
signature `MYTORENS`, version 1, the combination, each member as a complete binary network file
(ADR-0042) with its weight, the combiner for `Stack`, and an FNV-1a 64 checksum. `Network` gains
`friend class Ensemble` and a `parse` that reads bytes instead of a path; its public API does not
change. `my_torch_analyzer --predict` loads an ensemble when LOADFILE starts with its signature.
`bonus/my_torch_ensemble` builds the files and trains the `Stack` combiner
(`configs/ensemble_stack.conf`, a logistic regression on the 10 probabilities) on labelled
positions the members did not learn.

## Consequences
- ✅ Measured on the 71,555 test positions
  ([performance/27-ensembles.md](../performance/27-ensembles.md)): the stacked ensemble reaches
  98.26 % (CNN alone 98.07 %), the mean with the weights chosen on validation (dense 1, CNN 2)
  98.23 %; both gains over the CNN are significant (paired test, p < 1e-7).
- ✅ Same command line; one file; a corrupted, truncated or foreign file is refused.
- ❌ The plain mean (1:1) and the vote do not beat the CNN alone (+16 positions out of 71,555,
  p = 0.59): with one member much weaker than the other, equal weights only dilute the better one.
  With two members, the vote gave exactly the predictions of the 1:1 mean.
- ❌ An ensemble costs the time and memory of all its members: prediction 1.2 times the CNN's
  time, 535 MB peak memory against 416 MB, a 2.7 MB file against 0.6 MB for the CNN alone.
- ❌ A change to `include/my_torch/` (a new header, a friend and a private function in `Network`).
- Follow-up: more and more diverse members (CNNs from several seeds, ADR-0039) would test whether
  the gain grows; stacking on cross-validated predictions would use all the training data.
