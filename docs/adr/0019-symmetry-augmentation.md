# ADR-0019: Symmetry augmentation, offline tool and two label-preserving board symmetries

- **Status**: Accepted
- **Date**: 2026-10-25
- **Deciders**: Meric-BG (author), Ridwane777 (accepted)
- **Issue**: #38

## Context

The dataset holds 723,019 labelled positions and a network trained on a sample of it has to learn
by itself that a check along the a-file and the same check along the h-file are the same thing.
Chess has two symmetries that keep the answer, so each training position can give more positions:

- **Left-right mirror** (file a ↔ h): attacks, checks and mates do not depend on the files. The
  castling rights do not survive it (the king would no longer stand on its starting square), so
  they are cleared; the oracle ignores castling anyway (ADR-0006) and castling can neither answer a
  check nor create one, so the label is unchanged.
- **Top-bottom flip with the colours swapped** (rank r ↔ 9 − r, every piece changes colour, the
  side to move changes, K ↔ k and Q ↔ q, the en passant square goes from rank 3 to rank 6 or back):
  the same game seen from the other side. The label changes colour: `Check White` reads "White
  gives check" (docs/reading-a-fen.md), so it becomes `Check Black`, and `Checkmate White` becomes
  `Checkmate Black`. `Nothing` and `Stalemate` have no colour and do not change.

Together they give four positions per input. Three questions: where the symmetries live, how they
reach the training, and how to know they are right. The constraints: the analyzer and the generator
only include `include/my_torch/` and the network file format (ADR-0012) is a contract between the
two work packages (AGENTS.md); `--train` already holds out a seeded validation fraction from its
input (ADR-0014); the gain was measured on the real dataset on 2026-10-26
([performance/05-augmentation.md](../performance/05-augmentation.md)): accuracy 59.0 % to 67.1 %, recall of
`Check Black` 32.0 % to 50.3 % and of `Checkmate Black` 59.9 % to 72.6 %.

## Options considered

| Option | Pros | Cons |
| --- | --- | --- |
| **A. Offline tool (`scripts/augment.cpp`) on top of `src/analyzer/Symmetry.cpp`, writing an augmented chess file** | No change to `--train`, `my_torch_analyzer`, the network file or `include/my_torch/`; the augmented file can be inspected, counted and checked by the oracle; any partner can ignore it | One more step before training; a validation hold-out made by `--train` sees the augmented file (see below) |
| B. A `train` setting in the network file (`augment=all`), applied in `Cli.cpp` after the validation hold-out | Training only: the hold-out never contains a twin of a training position; one command | Changes the network file contract (ADR-0012) and `Cli.cpp`: needs both partners' sign-off and its own migration of the format |
| C. Augmentation always on inside `Cli.cpp`, no setting | Simplest for the user | Silent change of what `--train` means for every network; not reproducible from the network file; impossible to compare with and without |
| D. No augmentation | Nothing to maintain | Leaves a free, label-preserving source of positions unused; a 4x data effect is untested |

## Decision

Option A. `src/analyzer/Symmetry.{hpp,cpp}` implements `mirrorFiles`, `flipColors`,
`flipLabel` and `augment` (order per input: original, mirror, flip, mirror + flip), tested against
hand-written expected FENs. `scripts/augment.cpp` (`augment [--mirror | --flip | --all] IN OUT`,
errors on stderr, exit 84) exposes it; `src/analyzer/Fen.cpp` gains `formatFen` to write positions
back. Correctness is checked against an independent implementation: the rule-based oracle, through
`bonus/scripts/check_symmetries.sh` (the oracle is never referenced from `src/analyzer`). The
measurement is `scripts/bench_augmentation.sh`: the split into train and test is made **before**
augmentation, only the training file is augmented, both arms use the same configuration
(`configs/augmentation.conf`: fixed epochs, no early stopping, `validation=0`) and the same initial
weights, and the test file is never touched.

Option B stays on the table if the benchmark shows a gain: it needs its own ADR and the agreement of
both partners.

## Consequences

- ✅ Nothing changes for the analyzer, the generator, the network file or the public headers: the
  contract between the two work packages is untouched.
- ✅ The symmetries are verified twice: by unit tests with expected FENs written by hand (a wrong
  axis passes a "twice is the identity" test but not these) and by the oracle on 2,000 generated
  positions, 8,000 after augmentation (2,400 of them stalemates): `disagree: 0`
  (`sh bonus/scripts/check_symmetries.sh 2000`).
- ✅ Reproducible: seeded draw, a command line in `docs/performance/05-augmentation.md`.
- ❌ **Validation leak when `--train` holds out its own fraction.** An offline file puts the mirror
  and the flip of a position next to it; `--train` then draws its validation positions at random
  from the file, so most twins of a validation position land in the training part and the early
  stopping curve looks better than it is (optimistic early stopping). Option B would avoid it. The
  benchmark avoids it with `validation=0` and a fixed number of epochs; anyone training with
  patience on an augmented file accepts the bias.
- ❌ Memory: the augmented file has four times the positions and `--train` keeps every encoded
  position in memory (ADR-0014: about 4.4 GB for the whole dataset), so augmentation multiplies the
  limit by four until #17 solves it.
- ❌ The mirror clears the castling rights, so the augmented positions are biased toward "no
  castling" compared with the dataset; the baseline encodings do not read the rights, but a future
  encoding that does would see that bias.
- ❌ With a fixed number of epochs the augmented arm makes four times more updates than the base
  arm: a gain measured this way mixes "more distinct positions" and "more steps". The benchmark
  cannot separate them.
- ❌ The gain is measured on 20,000 positions, one seed and one encoding, and it mixes more distinct
  positions with more updates; it costs 4.1 times the time and 3.8 times the memory. Test leakage
  through duplicates and symmetric copies is 23 of 5,000 test positions, not 0.
- Follow-up: the gain is real under this protocol; the offline file (option A) is what is used, and
  the in-process option B gets its own ADR only if the validation leak becomes a problem; compare with the three-class
  canonical variant (side to move always White) before choosing, and repeat with the final
  encoding of #17. Mark this ADR Abandoned only if a later measurement removes the gain.
