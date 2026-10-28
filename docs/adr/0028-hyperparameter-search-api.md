# ADR-0028: Hyperparameter search in the library

- **Status**: Proposed
- **Date**: 2026-10-28
- **Deciders**: Ridwane777 (issue #10)

## Context
Until now every hyperparameter of the chess network was chosen by hand or by a few separate trial
series (ADR-0023, `docs/performance/09-trials.md`): learning rate 0.001, batches of 64, hidden layers
256 → 128, no dropout. Issue #10 asks to check these choices by a search over several
hyperparameters at once. A search trains one network per combination, 9 to 25 minutes each on the
development laptop at 10 % of the data, so a run lasts hours and must survive a stop (a closed lid,
a crashed session). The code that picks combinations does not depend on chess: only the function
that trains and scores a combination does.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| A shell script looping over configuration files | No C++ to write; the pattern of `scripts/trial_runs.sh` | One file per combination; no resume; the logic cannot be tested or reused by another network |
| Search hard-coded in a chess program | Quick | Same loop rewritten for the next network; not in the library |
| **Generic module `my_torch/Search.hpp`: search space, objective, strategy (chosen)** | Same split as Optuna and Ray Tune: the library chooses the combinations, the caller's objective scores each one; any network can use it; tested; a CSV log makes it resumable | One more public header to maintain; the objective is written by the caller |
| Successive halving or Hyperband (stop bad combinations early) | Less compute per search | The objective must report a score per epoch and be resumable mid-training; early stopping already ends most runs after 7 to 16 epochs |
| Bayesian optimisation (Gaussian process, or TPE as in Optuna) | Fewer trials on large spaces | A probabilistic model to write and test by hand (no library allowed); little gain on a 36-combination space |

## Decision
`SearchSpace` lists the values of each hyperparameter (numbers or labels). A strategy chooses grid
indices: `GridSearch` takes them all, `RandomSearch(count, seed)` draws `count` distinct ones with
the library's portable `permutation` (ADR-0009). `SearchStrategy::run` calls the objective once per
chosen combination and appends each score to a CSV log as soon as the trial ends; on a rerun, the
combinations already in the log are read back, not trained again. The chess objective is
`scripts/hyperparam_search.cpp`; the protocol and the results are in
`docs/performance/15-hyperparameters.md`.

## Consequences
- ✅ A search of several hours can be stopped and resumed without losing a trial; this happened on
  the first run, stopped after 6 of the 36 trials.
- ✅ The random search reuses the grid's log: comparing the two strategies costs no extra training.
- ✅ Tested (`tests/my_torch/test_search.cpp`): grid order, reproducible and distinct random draws,
  NaN never the best, a stopped search resumes with only the missing trials, a log of another
  space refused; also run under the sanitizers.
- ❌ Trials run one after the other: the matrix product already uses every thread (ADR-0024), so
  running trials side by side would not make the search faster on one machine.
- ❌ Only grid and random strategies. Hyperband, Bayesian optimisation, a genetic algorithm or
  simulated annealing would be other `SearchStrategy` subclasses, not written.
- ❌ The random search draws from the grid's values only, not from a continuous range such as a
  log-uniform learning rate.
- Follow-up: revisit if a space grows past a few hundred combinations, where Hyperband or a model
  of the score would pay for itself.
