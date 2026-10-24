# ADR-0007: Cross-entropy coupled to Softmax, Softmax gradient skipped in backpropagation

- **Status**: Proposed
- **Date**: 2026-10-24
- **Deciders**: Meric-BG (awaiting Ridwane777, who writes backpropagation in #4)
- **Issue**: #3

## Context

The analyzer's network ends with a `Softmax` layer, which turns scores `z` into probabilities `p`
(positive, summing to 1); the loss then compares `p` with the target `t` (a one-hot vector: 1 for
the right class, 0 elsewhere). For cross-entropy, `L = -Σ t_i·log p_i`.

Backpropagation applies the chain rule: the gradient of the loss with respect to `p` is computed
first, then each layer turns it into a gradient for its own input. For cross-entropy,
`dL/dp_i = -t_i / p_i`, and the `Softmax` backward then multiplies it by roughly `p_i`. When the
network is badly wrong, the right class has `p_i ≈ 0`, so the division by `p_i` must be guarded
(`p_i` is bounded to ε = 1e-15, otherwise it divides by zero) and the guard destroys the gradient.

Measured by [scripts/softmax_ce_gradient.cpp](../../scripts/softmax_ce_gradient.cpp) (two classes,
the right class pushed down by `gap` score points, naive path `-t/max(p, ε)` through
`Softmax::backward`, compared with the exact value `p - t`):

| gap | p of the right class | naive gradient | coupled gradient | exact `p - t` |
| --- | --- | --- | --- | --- |
| 10 | 4.54e-05 | -0.999955 | -0.999955 | -0.999955 |
| 30 | 9.36e-14 | -1.000000 | -1.000000 | -1.000000 |
| 40 | 4.25e-18 | **-0.004248** | -1.000000 | -1.000000 |

So the naive path is exact while `p > ε`, and loses the gradient abruptly below ε: at gap 40 it
returns -0.004 instead of -1, with no error and no NaN, exactly when the network is most wrong.
Without the guard it would instead return NaN once `p` underflows to 0 (not measured here).

Mathematically the two steps simplify: `dL/dz_i = p_i - t_i`. That form never divides by `p_i`.
ADR-0004 already announced this coupling without fixing the mechanism.

## Options considered

| Option | Pros | Cons |
| --- | --- | --- |
| **`CrossEntropy` takes the Softmax output, returns `dL/dz` and declares itself "fused" (`fusesSoftmax()`); backpropagation then skips the final Softmax backward** | Stable gradient; the network keeps its visible `Softmax` layer, so `predict` returns probabilities as in ADR-0004's example; no change to the layer list | One special case in backpropagation; `gradient()` no longer returns `dL/dprediction` for this loss, which must be documented |
| `CrossEntropy` takes the raw scores `z` (logits) and applies softmax internally; the network has no Softmax layer | No special case: this is PyTorch's `nn.CrossEntropyLoss` model; stable by subtracting the maximum | `predict` no longer returns probabilities (a separate softmax is needed at prediction time); ADR-0004's example and the network files change |
| Keep `dL/dp = -t/p` and bound `p` to ε | No API change | The gradient is wrong when `p < ε`, which is exactly the crushed right class; a band-aid |

## Decision

We take the first option. `Loss` gains a virtual `fusesSoftmax()` (false by default).
`CrossEntropy` returns true: its `value()` reads probabilities (bounded to 1e-15 before the
logarithm) and its `gradient()` returns the gradient with respect to the **input** of the final
Softmax, `(p·W - w⊙q) / batch_size` (`W = Σ w_c·q_c`), i.e. `(p - t) / batch` without weights or
smoothing. When `loss.fusesSoftmax()` is true, `Network::fit` (#4) must skip the `backward` of the
last layer, which must be a `Softmax`, and must **throw `my_torch::Error` if it is not**: otherwise
the gradient would silently be the wrong one. The full-network gradient test
(`tests/my_torch/gradient_check.hpp`) already does the skip and the check, and proves by finite
differences that the result is correct; one test also verifies that it detects a forgotten skip.

Two options of the same class, with no API impact: **label smoothing** `s` replaces the target with
`q = (1 - s)·t + s/classes` (the network is no longer pushed toward probabilities of 0 or 1, i.e.
infinite scores), and **class weights** `w_c` multiply the loss of the examples of class `c`, for
class imbalance (#16). The mean is still divided by the batch size, not by the sum of the weights.

## Consequences

- ✅ Exact, stable gradient even when the right class has a probability close to 0.
- ✅ `predict` still returns probabilities; ADR-0004's API does not change.
- ✅ Smoothing and class weights are covered by the same gradient test (relative error under 1e-6
  through a full network).
- ❌ A special case to remember in backpropagation: if #4 forgets it, the gradient is wrong with no
  visible error (only the full-network gradient test catches it).
- ❌ `gradient()` does not mean the same thing for every loss (`dL/dp` for MSE, `dL/dz` for
  `CrossEntropy`); documented in `Loss.hpp`.
- ❌ A network whose last layer is not a `Softmax` cannot use `CrossEntropy`: the gradient would be
  that of another computation. `BinaryCrossEntropy` exists for Sigmoid outputs.
- ❌ *Focal loss* is not implemented: its gradient with respect to the scores does not have this
  simple form and would need its own ADR.
- Follow-up: #4 calls `fusesSoftmax()` in `Network::fit`, throws on a mismatch, and adds a test of a
  network learning XOR with Softmax + `CrossEntropy`. To revisit if a layer is ever placed after the
  Softmax (not planned).
