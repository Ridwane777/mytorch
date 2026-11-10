# ADR-0008: Optimizers keep their own state; schedulers are stateless functions of the epoch

- **Status**: Accepted
- **Date**: 2026-10-24
- **Deciders**: Meric-BG (author), Ridwane777 (accepted)
- **Issue**: #5

## Context

Training moves every weight against its gradient. How far, and with what memory of past gradients,
is the optimizer's job (SGD, Adam...). How the step size changes over time is the scheduler's job.
The specification asks for faster learning and escaping local minima. ADR-0004 fixed the `Optimizer`
interface (`step(parameters)`, `setLearningRate`); two questions were left open: where an
optimizer keeps its memory, and how a schedule is expressed.

Measured on two interleaved spirals ([07-optimizers.md](../performance/07-optimizers.md), 5 seeds, 400
epochs, best of 9 learning rates): Adam reaches a median training loss of 0.0048 (99.5 % accuracy),
SGD with momentum 0.0113 (98.5 %) but only at one rate out of nine, and plain SGD stays near 0.23
(57 %). Between a rate that works and one that diverges there is a factor 3 for Adam.

## Options considered

**Where does an optimizer keep its memory (velocity, running averages)?**

| Option | Pros | Cons |
| --- | --- | --- |
| **In the optimizer, in matrices shaped like the parameters, indexed by position** | `Layer` stays unaware of optimizers; swapping the optimizer needs no change to layers; one place to save for #35 | The same parameters must be passed in the same order at every step (checked: `my_torch::Error` otherwise) |
| In each layer, next to its weights | No ordering assumption | Every new layer (#29 convolution, #30 normalisation) must carry optimizer state for every optimizer |
| Map keyed by the parameter's address | No ordering assumption | A moved or reallocated matrix silently starts from empty memory |

**How is a learning-rate schedule expressed?**

| Option | Pros | Cons |
| --- | --- | --- |
| **`Scheduler::rate(epoch)`, a pure function; the loop calls `optimizer.setLearningRate(...)`** | Trivially testable (value for a given epoch); nothing to save to resume a run (#35); an optimizer does not need to know about schedules | Cannot react to the training (no reduce-on-plateau); granularity is the epoch |
| A scheduler object that owns and updates the optimizer each step | Per-step schedules (one-cycle) | Couples the two classes; stateful, so it must be saved and restored |

## Decision

- Each optimizer keeps its memory itself (`std::vector<Matrix>` created at the first step, one
  matrix per parameter) and rejects parameters whose count or shapes change.
- Schedulers are pure functions of the epoch (`ConstantRate`, `StepDecay`, `ExponentialDecay`,
  `CosineAnnealing`, and a `Warmup` wrapper around any of them). The training loop applies
  `optimizer.setLearningRate(scheduler.rate(epoch))` at the start of each epoch.
- `setLearningRate` accepts **0** (constructors still require a positive rate): a cosine schedule
  returns its minimum, 0 by default, after its last epoch, and a decaying exponential can
  underflow. Negative rates still throw.
- Algorithms: SGD (momentum, Nesterov), Adam (with AMSGrad), AdamW (decoupled weight decay),
  RMSProp, Adagrad. Adam is the recommended default for the chess network, based on the benchmark
  above, to be confirmed on the real dataset.

## Consequences

- ✅ Adding an optimizer or a scheduler touches no layer and no existing optimizer.
- ✅ Schedulers are reproducible and need no saved state.
- ✅ Every optimizer is tested on hand-computed steps, on a quadratic with curvatures 1 and 10, and on
  XOR through a real network.
- ❌ Optimizer memory is tied to parameter order: a network whose layers change between steps (or a
  future graph architecture, #31) breaks it loudly. Checkpoints (#35) must save the optimizer memory
  as well as the weights; the format is not decided here.
- ❌ Epoch granularity: no per-step schedules (one-cycle, per-batch warm-up), and no
  validation-driven schedule (reduce on plateau needs state, so it would need a new ADR).
- ❌ The measurement is on a toy problem with training loss only; the best rates found (Adam 0.1,
  momentum 3) will not transfer to the chess data, and Adam's best rate sits a factor 3 below
  divergence. No scheduler beat a constant rate in the benchmark, and the differences are within seed
  noise, so the schedulers are shipped on principle, not on evidence.
- ❌ Not implemented: Adadelta, Nadam, one-cycle, warm restarts, reduce on plateau, gradient
  clipping (#34).
- Follow-up: repeat the comparison with `Network::fit` on the real positions with a validation set;
  `Network::fit` (#4) applies the scheduler and zeroes gradients after each step.
