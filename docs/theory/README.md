# Theory

The mathematics behind what `my_torch` implements, written for a reader who wants to understand the
library without reading all of its code. Every formula here matches the code it cites (paths are
relative to the repository root); a technique that the library does not implement is not described
as if it were. Measured numbers are never stated here without a link to the report in
[docs/performance/](../performance/README.md) that produced them.

## Chapters

| # | Chapter | What it covers |
| --- | --- | --- |
| 1 | [The problem](01-the-problem.md) | Classifying a FEN position into 5 labels, the input encodings, class imbalance |
| 2 | [Backpropagation](02-backpropagation.md) | Forward pass, loss, chain rule, the gradients of `Dense` in matrix form, the finite-difference gradient check of the tests |
| 3 | [Activations and losses](03-activations-and-losses.md) | ReLU, LeakyReLU, Sigmoid, Tanh, Softmax; MSE, MAE, Huber, binary cross-entropy, softmax cross-entropy with its fused gradient |
| 4 | [Optimization](04-optimization.md) | SGD, Adam, AdamW, RMSProp, Adagrad; schedulers; weight initialisation; L1/L2, dropout, BatchNorm, LayerNorm, gradient clipping |
| 5 | [Convolution and attention](05-convolution-and-attention.md) | `Conv2D` and pooling, residual blocks, multi-head self-attention, and what they measured on the chess data |
| | [References](references.md) | The papers behind each implemented technique, and where each one is used |

## Suggested reading order

- **Newcomer**: 1, then 2 (sections 1 to 3), then 3. Chapter 2 is the core idea of the whole library:
  every layer only needs a forward and a backward function.
- **Engineer**: 2 (the matrix form of `Dense` and the gradient check), 4 (update equations with the
  configuration keys of [configuration.md](../configuration.md)), then 5.
- **Researcher**: 1 (the encodings and the 5-label task), 3 (the fused softmax cross-entropy with
  class weights and label smoothing), 5, then [references.md](references.md).

## Notation

- A batch is a matrix with **one row per example** (`include/my_torch/Matrix.hpp`, `Scalar` is
  `double`). $X \in \mathbb{R}^{B \times n}$ has $B$ examples of $n$ features.
- $\odot$ is the element-wise product, $A^\top$ the transpose, $\mathbf{1}$ a column of ones.
- $L$ is the loss averaged over the batch; for a matrix $M$, $\partial L / \partial M$ (also written
  $dM$) has the shape of $M$.
- `snake_case` names in code font are configuration keys ([configuration.md](../configuration.md)).

Related: [glossary.md](../glossary.md) for terms, [architecture.md](../architecture.md) for the code
structure, [adr/](../adr/README.md) for the decisions.
