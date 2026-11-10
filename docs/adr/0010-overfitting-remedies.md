# ADR-0010: Overfitting remedies, per-layer L1/L2, dropout layer and early stopping in fit

- **Status**: Accepted
- **Date**: 2026-10-24
- **Deciders**: Ridwane777
- **Issue**: #7

## Context

A project requirement is the program to avoid overfitting. Validation loss is already recorded each
epoch (#4, ADR-0009) and AdamW provides decoupled weight decay (#5, ADR-0008). Issue #7 adds L1/L2
regularization, dropout and early stopping with restoration of the best model, configurable per
layer where it makes sense. Three placements have to be decided, and they touch the public headers
shared by both work packages.

## Options considered

**Where do L1/L2 live?**

| Option | Pros | Cons |
| --- | --- | --- |
| **In each `Dense` layer: coefficients l1, l2; backward adds `l1·sign(w) + 2·l2·w` to dW; `Layer::penalty()` reports the term** | Per-layer strength; biases are excluded naturally; future layers (convolution) do the same; the gradient stays checkable by finite differences on loss + penalty | One more virtual method on `Layer`; `fit` sums the penalties to report them |
| In the loss | One value for the whole network | `Loss` does not see the weights: its interface would have to change |
| In the optimizer | Same place as AdamW's weight decay | Not per layer; duplicates AdamW |

**How is dropout added?**

| Option | Pros | Cons |
| --- | --- | --- |
| **A `Dropout(rate, seed)` layer, inverted dropout** | Inserted anywhere, so per layer by construction; same model as activations (ADR-0004) and `nn.Dropout`; uses the existing `training` flag | One more layer in the list |
| A rate inside `Dense` | Fewer layers | Every future layer would reimplement it |

**Where does early stopping live?**

| Option | Pros | Cons |
| --- | --- | --- |
| **`TrainConfig`: `patience` (0 = off), `minDelta`, `restoreBest`** | Plain settings; `fit` owns the epoch loop, the validation loss and the weights | Another change to `Network.hpp` |
| A callback hook in `fit` | General | More API than needed now |
| The caller stops training itself | Nothing to change | It must run one epoch at a time and copy the weights by hand |

## Decision

`Dense` takes an optional `Regularization {l1, l2}`; its backward adds `l1·sign(w) + 2·l2·w` to the
weight gradient (never to the bias) and `penalty()` returns `l1·Σ|w| + l2·Σw²`. `Layer` gains a
virtual `penalty()` returning 0 by default; `fit` reports the mean penalty of the epoch in
`EpochStats::penalty`, while `trainLoss` stays the data loss so that it can be compared with
`validationLoss`. `Dropout(rate, seed)` is a layer: in training it zeroes each value with probability
`rate` and divides the others by `1 - rate`, its mask drawn from `std::mt19937_64` with a portable
uniform; in prediction it is the identity. Early stopping is in `TrainConfig`: it needs a validation
set, stops after `patience` epochs without beating the best validation loss by more than `minDelta`,
and restores the best weights when `restoreBest` is true.

## Consequences

- ✅ On the overfitting benchmark (`docs/performance/08-overfitting.md`, 5 seeds, test set never used
  for a choice), every remedy lowers the median test loss from 0.758: early stopping 0.267, dropout
  0.325, L2 0.488, L1 0.579, elastic net 0.580.
- ✅ L1 drives the weight of a useless input to below 0.01 while keeping the useful one (tested).
- ✅ The penalty gradient is checked by finite differences, and the test fails when the factor 2 of
  L2 is removed on purpose; biases are never penalized (tested).
- ✅ Early stopping restores exactly the weights a run stopped by hand at the best epoch would have
  (tested to the bit); dropout masks are reproducible from their seed.
- ❌ Three public headers change (`Layer.hpp`, `Dense.hpp`, `Network.hpp`): both members review.
- ❌ L1 is not differentiable at 0; `sign(0) = 0` is used, so a weight that reaches exactly 0 may
  leave it again. No proximal step is implemented.
- ❌ With Adam, L2 is rescaled by the adaptive step; decoupled weight decay (AdamW) behaves
  differently and the two are not compared yet.
- ❌ The usable L2 range is narrow on the benchmark (λ = 0.001 already underfits at about 60 %
  accuracy), and L1 costs accuracy there (83 % at its selected λ); the strengths must be measured
  again on the chess data (#17).
- ❌ Early stopping keeps a full copy of the weights; cheap for the planned network sizes, to
  revisit for very large models.
- Follow-up: `save`/`load` (#8) must store each layer's regularization and dropout rate; checkpoints
  (#35) must store the early-stopping state if training is resumed.
