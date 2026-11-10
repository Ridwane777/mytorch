# ADR-0009: Training loop, scheduler parameter and portable shuffling

- **Status**: Accepted
- **Date**: 2026-10-24
- **Deciders**: Ridwane777
- **Issue**: #4

## Context

`Network::fit` (#4) runs mini-batch training with the losses of #3 (ADR-0007) and the optimizers and
schedulers of #5 (ADR-0008). Three points are not settled by ADR-0004:

- the `fit` signature fixed in ADR-0004 has no place for a learning-rate scheduler, which arrived
  later with #5;
- the specification requires a reproducible training, and #4 requires the same result for the same seed;
- the epoch loss must stay correct when the last batch is smaller than the others.

## Options considered

**Where does the scheduler go?**

| Option | Pros | Cons |
| --- | --- | --- |
| **Optional parameter of `fit`**: `const Scheduler *scheduler = nullptr` | Explicit; existing calls still compile unchanged | Changes the public header shared by both work packages |
| Inside `TrainConfig` | `fit` signature unchanged | `TrainConfig` stops being a plain, copyable set of settings |
| No scheduler in `fit`: the caller runs one epoch at a time | Nothing to change | The epoch loop and the history are rebuilt by hand by every caller |

**How to shuffle reproducibly?**

| Option | Pros | Cons |
| --- | --- | --- |
| `std::shuffle` with `std::mt19937_64` | One line | `std::shuffle` and `std::uniform_int_distribution` are implementation-defined: the same seed can give another order with another standard library (libstdc++ against libc++) |
| **Own Fisher-Yates on `std::mt19937_64`, with rejection sampling for the bounded draw** | The engine is fully specified by the standard, so the order is the same everywhere | About 20 lines of code to own and test |

**How to average the epoch loss?**

| Option | Pros | Cons |
| --- | --- | --- |
| Mean of the batch losses | Simple | A last batch of 2 examples weighs as much as a full batch of 32 |
| **Batch loss × batch size, divided by the number of examples** | Exactly the mean over every example of the epoch | One multiplication per batch |

## Decision

`fit` takes an optional `const Scheduler *scheduler` (default `nullptr`) and applies
`optimizer.setLearningRate(scheduler->rate(epoch))` at the start of each epoch. Each epoch shuffles
the example order with an in-house Fisher-Yates on `std::mt19937_64` seeded once per `fit` with
`TrainConfig::seed`. For each batch, the loop zeroes the gradients, runs the forward pass in training
mode, adds the batch loss weighted by its size, backpropagates from the last layer to the first
(skipping the final Softmax when the loss fuses it, ADR-0007) and calls `optimizer.step`. The
validation loss, if a validation set is given, is computed at the end of each epoch with `predict`.

## Consequences

- ✅ Same seed, same weights and same history to the bit (tested), and the shuffle does not depend on
  the standard library.
- ✅ The epoch loss is the true mean over the examples, partial last batch included (tested with a
  zero learning rate: equal to the loss of the whole set).
- ✅ The gradient used by `fit` is checked against finite differences through a full network with
  label smoothing and class weights; a test fails if the Softmax skip is removed.
- ✅ XOR is learned with Softmax + `CrossEntropy` and with Sigmoid + `MSE` (milestone of #4).
- ❌ Changes the public `Network.hpp`, so both members must review the PR.
- ❌ Reproducibility is per machine only for the weight initialisation: `Dense` still draws its
  weights with `std::uniform_real_distribution`, which is implementation-defined; to fix in #6.
- ❌ Each batch copies its rows into new matrices; acceptable now, to measure on the full dataset
  (performance, #33).
- ❌ Without the gradient test, a wrong backward can go unnoticed: with the Softmax skip removed on
  purpose, XOR was still learned.
- Follow-up: metrics per epoch (#11) extend `EpochStats`; early stopping (#7) and checkpoints (#35)
  will need hooks in this loop, through a new ADR if the signature changes again.
