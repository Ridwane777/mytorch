# ADR-0033: Normalization layers, BatchNorm and LayerNorm

- **Status**: Accepted (2026-10-29)
- **Date**: 2026-10-29
- **Deciders**: Ridwane777 (issue #30), approved by Meric-BG

## Context
Issue #30 asks for batch normalization (with running statistics and a training / prediction mode)
and layer normalization, saved in the network file, checked by finite differences, and measured
with and without normalization. Normalization rescales each feature to mean 0 and variance 1, then
lets two trained parameters per feature (gamma, beta) choose the scale; it is the usual way to
train deep and convolutional networks (#29) at larger learning rates. The `Layer` interface already
receives the training flag, and since ADR-0032 a layer can save a state that is not a parameter.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| **BatchNorm and LayerNorm as two layers (chosen)** | The two of the issue; they compose with the others in a configuration; LayerNorm needs no batch | Two classes, two backward passes to check |
| BatchNorm only | The classic choice for convolutional networks | Depends on the batch size; nothing for a batch of one or for the attention of #32 |
| Normalization as an option of `Dense` | No new layer line | Mixes two roles in one class; no LayerNorm after other layers |
| **After the Dense, before its activation (chosen, as in the original paper)** | The usual placement; the ReLU receives centred values | Some prefer after the activation; not measured here |
| **Running averages with a momentum of 0.1 and the unbiased variance (chosen, as PyTorch)** | Same results as the reference library for the same settings | The prediction differs slightly from training on the same batch (biased variance in training) |
| A cumulative average of all batches | No setting | Early batches, from untrained weights, weigh as much as the last ones |

## Decision
`BatchNorm(features, momentum = 0.1, epsilon = 1e-5)` and `LayerNorm(features, epsilon = 1e-5)` in
`my_torch/Normalization.hpp`, written `batch_norm <features>` and `layer_norm <features>` in
configuration and network files (gamma, beta, and for BatchNorm the running mean and variance).
BatchNorm uses the batch statistics in training and the running averages in prediction; a training
batch of one row normalizes to 0 and does not change the averages. The running averages are the
layer's state (`saveState`), saved in checkpoints and, with the best weights, by early stopping.
The final network does not use them (`docs/performance/17-normalization.md`).

## Consequences
- ✅ Both backward passes match finite differences on the input, gamma and beta
  (`tests/my_torch/test_normalization.cpp`); in prediction a position gives the same answer alone or
  in a batch; save and load, and a resumed training, are bit-identical; also under the sanitizers.
- ✅ The measurement found a bug: early stopping restored the best weights with the running
  averages of the last epoch (0.798 instead of 0.857 for BatchNorm on the final architecture). The
  layers' state is now saved and restored with the best weights, also in checkpoints (ADR-0032,
  format version 2), with a regression test.
- ✅ Ready for the deeper networks of #29 and #31.
- ❌ No gain on the chess network at 10 % of the data: equal scores on the final architecture
  (less overfitting of the validation loss only), lower ones on a six-layer network.
- ❌ One seed only, one learning rate: the larger learning rates normalization allows were not tried.
- ❌ BatchNorm depends on the batch: a batch of one row in training learns nothing through it.
- Follow-up: measure again on the convolutional network of #29, with a larger learning rate.
