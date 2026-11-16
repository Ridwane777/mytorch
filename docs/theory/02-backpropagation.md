[Theory index](README.md) · previous: [The problem](01-the-problem.md) · next: [Activations and losses](03-activations-and-losses.md)

# 2. Backpropagation

`my_torch` has no automatic differentiation: every layer writes its own derivative by hand. This
chapter explains why that is enough, and how the tests check that each hand-written derivative is
right.

## 1. Layers, forward and backward

A network is a sequence of layers $f_1, \dots, f_K$ (`include/my_torch/Layer.hpp`,
`src/my_torch/Network.cpp`). On a batch $X$ (one row per example), the **forward pass** computes

$$A_0 = X, \qquad A_k = f_k(A_{k-1}) \quad (k = 1, \dots, K), \qquad P = A_K.$$

Each layer has two methods:

- `forward(input, training)`: computes $A_k$ from $A_{k-1}$ and keeps what its derivative will need
  (its input, its output, a dropout mask, ...);
- `backward(gradOutput)`: receives $G_k = \partial L / \partial A_k$, **adds** the gradients of its
  own parameters to their gradient matrices, and returns $G_{k-1} = \partial L / \partial A_{k-1}$.

Parameters are exposed as `Parameter` pairs (value, gradient) by `parameters()`, which is all an
optimizer sees ([chapter 4](04-optimization.md)).

## 2. The loss

A `Loss` (`include/my_torch/Loss.hpp`) gives a value $L(P, T)$ and its gradient
$\partial L / \partial P$ for a prediction $P$ and targets $T$ of the same shape. Every loss of the
library averages over the batch: with $\ell$ the per-example loss,

$$L = \frac{1}{B} \sum_{i=1}^{B} \ell(p_i, t_i),$$

so the gradient carries a factor $1/B$ (`src/my_torch/Losses.cpp`). The losses are listed in
[chapter 3](03-activations-and-losses.md).

## 3. The chain rule, layer by layer

Since $L$ depends on $A_{k-1}$ only through $A_k = f_k(A_{k-1})$, the chain rule gives, element by
element,

$$\frac{\partial L}{\partial (A_{k-1})_{ij}} = \sum_{a,b} \frac{\partial L}{\partial (A_k)_{ab}} \, \frac{\partial (A_k)_{ab}}{\partial (A_{k-1})_{ij}}.$$

So the gradient flows backwards: start from $G_K = \partial L / \partial P$, then call
`backward` on layers $K, K-1, \dots, 1$. Each layer only needs its local derivative, never the rest
of the network. This is backpropagation (Rumelhart, Hinton and Williams, 1986). One training step
of `Network::fit` (`src/my_torch/Network.cpp`) is:

1. forward pass in training mode;
2. `grad = loss.gradient(P, T)`;
3. for $k = K, \dots, 1$: `grad = layers[k]->backward(grad)` (the last Softmax is skipped when the
   loss fuses it, [chapter 3](03-activations-and-losses.md));
4. check that every gradient is finite, clip the gradients if configured, `optimizer.step`.

Gradient matrices **accumulate** (`+=`), and the training loop zeroes them before each batch.

## 4. The Dense layer in matrix form

`Dense` (`src/my_torch/Dense.cpp`) holds $W \in \mathbb{R}^{n \times m}$ (inputs by outputs) and a
bias row $b \in \mathbb{R}^{1 \times m}$. With $X \in \mathbb{R}^{B \times n}$:

$$Z = X W + \mathbf{1} b \qquad (\texttt{input.matmul(\_weights).addRowVector(\_bias)}).$$

Given $G = \partial L / \partial Z \in \mathbb{R}^{B \times m}$, since $Z_{ij} = \sum_k X_{ik} W_{kj} + b_j$:

$$\frac{\partial L}{\partial W} = X^\top G, \qquad \frac{\partial L}{\partial b} = \mathbf{1}^\top G \ \text{(sum of the rows)}, \qquad \frac{\partial L}{\partial X} = G\, W^\top.$$

In the code these are `_input.transposeMatmul(gradOutput)`, `gradOutput.sumRows()` and
`gradOutput.matmulTranspose(_weights)`: the transposes are never materialised. The batch average is
already inside $G$ (the $1/B$ of the loss), so $X^\top G$ is the mean over the batch of the
per-example outer products $x_i^\top g_i$.

When the layer has `l1` or `l2` set, the objective gets a penalty
$\lambda_1 \sum |w| + \lambda_2 \sum w^2$ (`Dense::penalty`) and `backward` adds its derivative

$$\frac{\partial}{\partial W}\Big(\lambda_1 \textstyle\sum |w| + \lambda_2 \sum w^2\Big) = \lambda_1\, \operatorname{sign}(W) + 2 \lambda_2 W,$$

with $\operatorname{sign}(0) = 0$. The penalty is not divided by the batch size. Biases are not
penalised.

A two-layer example, `dense → relu → dense → softmax` with cross-entropy, therefore computes:
$G_3 = (P - T)/B$ (fused softmax), $dW_2 = A_2^\top G_3$, $G_2 = G_3 W_2^\top$, then
$G_1 = G_2 \odot \mathbb{1}[Z_1 > 0]$ (ReLU), $dW_1 = X^\top G_1$.

## 5. Checking the derivatives: finite differences

A wrong derivative does not crash: training just goes worse. The tests therefore compare every
hand-written gradient with a numerical one (`tests/my_torch/gradient_check.hpp`).

**Centred difference.** For a scalar function $L(\theta)$ and a step $h$,

$$\frac{L(\theta + h) - L(\theta - h)}{2h} = L'(\theta) + O(h^2),$$

more accurate than the one-sided $(L(\theta + h) - L(\theta))/h$, whose error is $O(h)$. The tests
use $h = 10^{-6}$ in double precision.

**Testing one layer.** A layer's output is a matrix, not a scalar. The check builds the scalar
$L = \sum_{ij} \text{output}_{ij} R_{ij}$ with $R$ a fixed random matrix, so that
$\partial L / \partial \text{output} = R$ exactly. It passes $R$ to `backward`, then moves each cell
of the tested matrix (a weight, a bias, or the input) by $\pm h$ and compares the measured slope
with the analytic gradient through the relative error

$$\varepsilon = \frac{|a - \tilde a|}{\max(1, |a| + |\tilde a|)}$$

(`maxError`; `inputError` does the same for $\partial L / \partial X$). The tolerance is
$10^{-6}$. `lossError` checks a loss's gradient against its value, and `networkError` checks every
parameter of a complete network, with the same backward order as `Network::fit`. Every
parameterised layer of the library is tested this way, including convolution and attention
(`tests/my_torch/test_dense.cpp`, `test_convolution.cpp`, `test_attention.cpp`, ...).

The same gradients were also compared with PyTorch's autograd on identical weights and batches
([22-pytorch-reference.md](../performance/22-pytorch-reference.md),
[ADR-0040](../adr/0040-pytorch-reference.md)).
