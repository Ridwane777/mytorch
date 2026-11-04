# ADR-0034: Uncertainty by Monte Carlo dropout

- **Status**: Accepted (2026-11-04)
- **Date**: 2026-10-29
- **Deciders**: Ridwane777 (issue #26), approved by Meric-BG

## Context
Issue #26 asks for predictions with dropout active over N passes, an uncertainty measure, and an
analysis: are the errors the uncertain predictions? The library has `Dropout` (ADR-0010) and a
`forward(input, training)` that switches every layer at once, so dropout cannot be on while
BatchNorm (ADR-0033) stays in prediction mode. The pushed network has no dropout; the candidate of
the hyperparameter search (ADR-0030, abandoned) has 0.2 and is trained on all the data.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| **Monte Carlo dropout: N passes with only the dropout layers active (chosen, the subject's suggestion)** | No new training; any network with dropout | N times the prediction cost; only for networks with dropout |
| Ensemble of independently trained networks (#40) | Usually the best uncertainty | N full trainings (hours each) |
| Plain confidence: the top probability of one pass | Free, every network | Not a model uncertainty; taken as the baseline to beat |
| Calibration (temperature scaling) | Better probabilities | Needs a fit on the validation split; changes no ranking of the positions |
| **Three measures returned (entropy, mutual information, disagreement) (chosen)** | Each answers a different question; the analysis compares them | More output to explain |

## Decision
`Network::predictMonteCarlo(input, passes, seed)` runs `passes` predictions with only the `Dropout`
layers in training mode, each seeded from `seed`, restores the network's own generators, and
returns the mean probabilities with the entropy, mutual information and disagreement of each row.
It throws without a dropout layer. The analysis (`docs/performance/18-uncertainty.md`) uses the
candidate network on the test split and compares the three measures with the plain confidence.

## Consequences
- ✅ Reproducible (same seed, same result) and without side effect on the network (tested in
  `tests/my_torch/test_uncertainty.cpp`, with BatchNorm kept in prediction mode).
- ✅ Answers the issue: the errors are the uncertain predictions (AUC 0.93; the most uncertain tenth
  holds about 70 % of the errors).
- ❌ MC dropout does not beat the plain confidence (AUC 0.938) on this network and split, for 30
  times the cost; its own measures, mutual information (0.759) and disagreement (0.837), are the
  weakest. The uncertainty to show is therefore the plain top probability.
- ❌ Useless with the pushed network, which has no dropout.
- Follow-up: compare with an ensemble (#40), and on positions unlike the training data.
