[Theory index](README.md) · previous: [Backpropagation](02-backpropagation.md) · next: [Optimization](04-optimization.md)

# 3. Activations and losses

Only what the library implements: the activations of `include/my_torch/Activations.hpp`
(`src/my_torch/Activations.cpp`) and the losses of `include/my_torch/Losses.hpp`
(`src/my_torch/Losses.cpp`).

## Activations

Without a non-linearity between them, stacked `Dense` layers collapse into one linear map
($XW_1W_2 = X(W_1W_2)$). The activations are parameter-free layers applied element by element
(except Softmax, applied per row). In backward, an element-wise activation $y = \sigma(z)$ returns
$G \odot \sigma'(Z)$.

| Layer (config) | $\sigma(z)$ | $\sigma'(z)$ as computed | Kept from forward |
| --- | --- | --- | --- |
| `relu` | $\max(0, z)$ | $1$ if $z > 0$, else $0$ (so $0$ at $z = 0$) | input |
| `leaky_relu slope=a` (0.01) | $z$ if $z > 0$, else $a z$ | $1$ if $z > 0$, else $a$ | input |
| `sigmoid` | $s = 1/(1 + e^{-z})$ | $s(1 - s)$, at most 0.25 | output |
| `tanh` | $t = \tanh z$ | $1 - t^2$ | output |

Sigmoid and Tanh compute their derivative from their **output**, which they keep instead of the
input. Both saturate: for large $|z|$ the derivative is close to 0, which slows learning in deep
stacks; ReLU does not saturate for $z > 0$, which is why He initialisation pairs with it
([chapter 4](04-optimization.md)).

### Softmax

`softmax` turns a row of scores $z \in \mathbb{R}^C$ into probabilities:

$$p_j = \frac{e^{z_j - \max_k z_k}}{\sum_c e^{z_c - \max_k z_k}}.$$

Subtracting the row maximum does not change the result (it cancels between numerator and
denominator) but keeps every exponent $\le 0$, so $e^{\cdot}$ never overflows. Its Jacobian is
$\partial p_j / \partial z_i = p_j(\delta_{ij} - p_i)$, so for an incoming gradient $g$:

$$\frac{\partial L}{\partial z_i} = \sum_j g_j\, p_j(\delta_{ij} - p_i) = p_i \Big(g_i - \sum_j p_j g_j\Big),$$

which is what `Softmax::backward` computes per row, in $O(C)$ without building the Jacobian.

## Losses

All losses average over the batch: the value is the sum over every cell divided by the batch size
$B$, and the gradient is the per-cell derivative divided by $B$. $p$ is the prediction, $t$ the
target, $d = p - t$.

| Config | Per-example value | Gradient w.r.t. $p_j$ (before $/B$) |
| --- | --- | --- |
| `mse` | $\sum_j (p_j - t_j)^2$ | $2(p_j - t_j)$ |
| `mae` | $\sum_j \lvert p_j - t_j\rvert$ | $\operatorname{sign}(p_j - t_j)$, $0$ at $p_j = t_j$ (subgradient) |
| `huber delta=δ` (1) | $\sum_j \tfrac12 d_j^2$ if $\lvert d_j\rvert \le \delta$, else $\delta(\lvert d_j\rvert - \tfrac12\delta)$ | $\operatorname{clamp}(d_j, -\delta, \delta)$ |
| `binary_cross_entropy` | $-\sum_j [t_j \log p_j + (1 - t_j)\log(1 - p_j)]$ | $\dfrac{p_j - t_j}{p_j(1 - p_j)}$ |
| `cross_entropy` | $-\sum_j w_j q_j \log p_j$ (see below) | fused with Softmax (see below) |

Note that `mse` sums over the output columns and divides by the batch size only, not by the number
of columns. Binary cross-entropy clamps $p$ to $[\varepsilon, 1 - \varepsilon]$ with
$\varepsilon = 10^{-15}$ before the logarithm and the division; it expects probabilities, i.e. a
final `sigmoid`.

## Softmax cross-entropy, fused

The chess networks end with `softmax` and use `cross_entropy`. Applied naively, the chain would
compute $\partial L / \partial p_j = -w_j q_j / p_j$, then go through the Softmax backward. The
division by $p_j$ blows up when the right class has a probability close to 0. The library instead
couples the two ([ADR-0007](../adr/0007-softmax-cross-entropy-coupling.md)): `CrossEntropy::fusesSoftmax()`
returns true, `Network::fit` requires a final `softmax` and skips its backward, and
`CrossEntropy::gradient` returns the gradient with respect to the **Softmax input** $z$ directly.

**Label smoothing and class weights.** With `smoothing` $s \in [0, 1)$ and $C$ classes, the target
becomes $q = (1 - s)\,t + s/C$; with `weights` $w_1, \dots, w_C$ (all 1 when omitted), the
per-example loss is

$$\ell = -\sum_{c} w_c\, q_c \log p_c, \qquad p = \operatorname{softmax}(z),$$

with $\log p_c$ computed as $\log \max(p_c, 10^{-15})$. Using $\partial \log p_c / \partial z_j = \delta_{cj} - p_j$:

$$\frac{\partial \ell}{\partial z_j} = -\sum_c w_c q_c (\delta_{cj} - p_j) = p_j\, W - w_j q_j, \qquad W = \sum_c w_c q_c.$$

`CrossEntropy::gradient` returns exactly $(p_j W - w_j q_j)/B$. Without weights, $W = \sum_c q_c = 1$
and this reduces to the familiar

$$\frac{\partial L}{\partial z} = \frac{p - q}{B},$$

which is bounded by $1/B$ in absolute value whatever the probabilities. The mean is still divided
by the batch size, not by the sum of the weights. The tests check this fused gradient against
finite differences of the loss value through the whole network (`networkError` in
`tests/my_torch/gradient_check.hpp`).

Why cross-entropy for classification: it is the negative log-likelihood of the true label under
the predicted distribution, so minimising it is maximum-likelihood estimation; and its gradient
$p - t$ stays large when the network is confidently wrong, where MSE on a softmax would have a
gradient damped by the softmax Jacobian.

The pushed network uses `cross_entropy` without smoothing or weights (`configs/final_network.conf`);
the weighted variant was measured in [04-imbalance.md](../performance/04-imbalance.md).
