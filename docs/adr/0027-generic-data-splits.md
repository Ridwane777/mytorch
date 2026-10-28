# ADR-0027: Generic data splits in the library

- **Status**: Accepted (2026-10-28)
- **Date**: 2026-10-27
- **Deciders**: Ridwane777

## Context
The only split of the project is `scripts/dataset_split.py` (ADR-0017): dedupe-aware and stratified,
in Python, and specific to chess positions. The next steps need splits inside a C++ program: the
hyperparameter search (#10) scores its finalists, and k-fold cross-validation (#39) reports a mean
and a standard deviation. Training on one fold of a `PositionSource` (ADR-0021) also needs a view on
some of its rows without copying them.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| Keep splits in Python only | Nothing to do | Every C++ program that needs a fold writes files and reloads them; no k-fold |
| Rewrite the official split in C++ and replace `dataset_split.py` | One implementation | A different random generator draws other positions: the test split would change and every published measurement (96.2 % included) would no longer be comparable |
| **A generic module beside it, `my_torch/Data.hpp` (chosen)** | Reusable by any network (scikit-learn's stratified splitters, PyTorch's `Subset`); the official split is untouched | Two split codes in the project, with different jobs |

## Decision
`stratifiedSplit(classes, proportions, seed)` and `stratifiedKFold(classes, k, seed)` work on class
indices only; their randomness is the library's portable `permutation` (ADR-0009). `SubsetSource`
is a `DataSource` over chosen rows of another source, and `targetClasses` reads the class of every
row of a source. `scripts/dataset_split.py` stays the split of record for the chess dataset.

## Consequences
- ✅ The search (#10) and k-fold cross-validation (#39) can be written in C++ on any `DataSource`.
- ✅ Tested: every index in exactly one part or validation fold, class shares within one example,
  same seed same split, and training on a `SubsetSource` gives exactly the weights of training on
  a copy of its rows.
- ❌ Two split implementations: the Python one makes the official train/validation/test files, the
  library one only cuts what a program already holds. The documentation says which is which.
- ❌ The library split does not deduplicate: a caller with duplicated examples must remove them
  first, as `dataset_split.py` does with positions.
- ❌ `my_torch.hpp`, the public header, includes one more file: review by both members.
