# ADR-0031: Training stability, gradient clipping and a stop on non-finite values

- **Status**: Accepted (2026-10-29)
- **Date**: 2026-10-28
- **Deciders**: Ridwane777 (issue #34), approved by Meric-BG

## Context
A full training of the chess network lasts hours (6,285 s for the final network,
`docs/performance/10-final-network.md`). An exploding gradient makes one step move the weights by a
huge amount; once a weight is infinite or NaN, every later output is NaN. Before this decision `fit`
did not notice: training went on to the end with ruined weights, and nothing told at which epoch, in
which layer, it went wrong. Issue #34 asks for gradient clipping and a clear stop.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| Clip by value only (each gradient in [-v, v]) | Simplest; bounds every component | Changes the direction of the step when some components are clamped and others not |
| Clip by global norm only (rescale all gradients so that their L2 norm is at most c) | Keeps the direction, as `clip_grad_norm_` in PyTorch; the usual choice | One threshold for the whole network: a large gradient in one layer shrinks every layer's step |
| **Both, each off by default, value first then norm (chosen)** | The two tools of the issue; old configurations and network files behave exactly as before | Two settings to understand; clipping hides a too-high learning rate instead of fixing it |
| Clip per layer | Each layer bounded separately | Not asked; a threshold per layer to choose |
| On a non-finite value: skip the batch and go on | The run survives a single bad batch | Hides the problem; a run that diverges skips batch after batch without saying so |
| **On a non-finite value: stop with an `Error` naming the epoch, the batch and the layer (chosen)** | The cause is visible at once, through the usual error path (stderr, exit code 84) | Hours of training are lost when it happens late, with no checkpoint to resume from |
| Make the check optional | No cost when off | A silent ruined run is exactly the failure to avoid; the check is cheap |

## Decision
`TrainConfig` gets `clipValue` and `clipNorm` (`clip_value=` and `clip_norm=` on the `train` line of
a configuration or network file), 0 meaning off. In each batch of `fit`, after `backward`: the
gradients are checked, clamped by value, rescaled by global norm, then the optimizer steps and the
weights are checked. The loss of the batch is checked as well. The first non-finite loss, gradient or
weight throws `my_torch::Error`, for example
`fit: non-finite gradient in layer 2 (dense) at epoch 3, batch 17; lower the learning rate or set clip_norm / clip_value`
(layers counted from 1, as in the configuration file). The check is always on.

## Consequences
- ✅ A diverging run stops at the batch where it diverges, with the layer at fault, instead of
  running for hours on NaN weights.
- ✅ Clipping keeps a deliberately diverging case finite (tests in `tests/my_torch/test_stability.cpp`:
  a linear regression that plain SGD makes overflow, a layer with a NaN gradient, the norm and the
  direction after clipping, saving and loading the settings, negative thresholds refused; also run
  under the sanitizers).
- ✅ Off by default: the final network and its configuration are unchanged; older network files,
  without the two keys, still load.
- ❌ The checks read every gradient and every weight once per batch. Their cost is too small to
  measure: on a 3-epoch synthetic training of the final architecture on an idle laptop, `fit` took
  3.93 s ± 0.38 with them against 4.02 s ± 0.21 without (5 runs each,
  `docs/performance/16-benchmarks.md`). A first measurement, 3 % to 7 % slower, was taken while
  another training ran and is replaced by this one.
- ❌ Stopping loses the run: there is no checkpoint to resume a training from.
- ❌ No per-layer clipping, and no automatic choice of the threshold.
- Follow-up: if a larger network makes the checks measurable, check the weights once per epoch
  instead of once per batch.
