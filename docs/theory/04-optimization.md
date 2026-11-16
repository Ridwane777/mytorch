[Theory index](README.md) · previous: [Activations and losses](03-activations-and-losses.md) · next: [Convolution and attention](05-convolution-and-attention.md)

# 4. Optimization

Backpropagation gives a gradient $g = \partial L / \partial w$ for every weight $w$. This chapter
covers how the weights start, how they move, and what keeps them from overfitting, with the
hyperparameter names of the configuration keys ([configuration.md](../configuration.md)). Decisions:
[ADR-0008](../adr/0008-optimizers-and-schedulers.md), [ADR-0020](../adr/0020-weight-initialisation.md),
[ADR-0010](../adr/0010-overfitting-remedies.md), [ADR-0031](../adr/0031-training-stability.md),
[ADR-0033](../adr/0033-normalization.md).

## Optimizers

`include/my_torch/Optimizers.hpp`, `src/my_torch/Optimizers.cpp`. Updates are element-wise; the
state (velocity, averages) has the shape of each parameter and starts at 0. $\eta$ is `lr`.

**`sgd lr momentum nesterov`** (0.01, 0, 0):

$$v \leftarrow \text{momentum}\cdot v + g, \qquad w \leftarrow w - \eta \cdot \begin{cases} v & \text{nesterov} = 0\\ g + \text{momentum}\cdot v & \text{nesterov} = 1 \end{cases}$$

With `momentum` 0 this is plain gradient descent, $w \leftarrow w - \eta g$. The velocity
accumulates past gradients, which speeds up travel along consistent directions and damps
oscillations; the Nesterov form looks one step ahead with the updated velocity.

**`adam lr beta1 beta2 epsilon amsgrad`** (0.001, 0.9, 0.999, 1e-8, 0), with $k$ the step count:

$$m \leftarrow \beta_1 m + (1 - \beta_1) g, \qquad v \leftarrow \beta_2 v + (1 - \beta_2) g^2,$$

$$w \leftarrow w - \eta\, \frac{m / (1 - \beta_1^k)}{\sqrt{\hat v / (1 - \beta_2^k)} + \epsilon}, \qquad \hat v = \begin{cases} v & \text{amsgrad} = 0\\ \max(\hat v_{\text{prev}}, v) & \text{amsgrad} = 1. \end{cases}$$

The divisions by $1 - \beta^k$ correct the bias of averages that start at 0. Since $m$ and
$\sqrt v$ have the scale of $g$, each weight moves by about $\eta$ per step whatever its gradient's
scale. With `amsgrad`, the running maximum is taken on the uncorrected $v$ and corrected afterwards.

**`adamw lr weight_decay beta1 beta2 epsilon amsgrad`** (`weight_decay` 0.01, the rest as Adam):
before the Adam update of the same step,

$$w \leftarrow w - \eta \cdot \text{weight\_decay} \cdot w.$$

The decay is **decoupled** from the gradient. An L2 term added to the gradient of Adam would be
divided by $\sqrt{\hat v}$ like the rest and lose its effect on weights with large gradients
(Loshchilov and Hutter). Plain `adam` has a weight decay of 0.

**`rmsprop lr decay epsilon`** (0.001, 0.9, 1e-8): $s \leftarrow \text{decay}\cdot s + (1 - \text{decay})\, g^2$,
$w \leftarrow w - \eta\, g / (\sqrt s + \epsilon)$. No bias correction, no momentum.

**`adagrad lr epsilon`** (0.01, 1e-10): $s \leftarrow s + g^2$, $w \leftarrow w - \eta\, g / (\sqrt s + \epsilon)$.
The sum only grows, so each step only shrinks.

The pushed network uses Adam with `lr=0.001` (`configs/final_network.conf`); the optimizers were
compared in [07-optimizers.md](../performance/07-optimizers.md).

## Learning-rate schedulers

`include/my_torch/Schedulers.hpp`, `src/my_torch/Schedulers.cpp`. A scheduler is a pure function of
the epoch $e$ (from 0); at the start of each epoch the training loop calls
`optimizer.setLearningRate(scheduler.rate(e))`, so when a scheduler is set, its `base` (0.001 by
default) replaces the optimizer's `lr`. `scheduler none` keeps `lr`.

| Config | $\eta(e)$ |
| --- | --- |
| `constant base` | $\text{base}$ |
| `step base step gamma` (10, 0.1) | $\text{base}\cdot \text{gamma}^{\lfloor e / \text{step}\rfloor}$ |
| `exponential base gamma` (0.95) | $\text{base}\cdot \text{gamma}^{e}$ |
| `cosine base epochs min` (10, 0) | $\text{min} + (\text{base} - \text{min})\,\frac{1 + \cos(\pi e / \text{epochs})}{2}$ for $e < \text{epochs}$, then $\text{min}$ |
| any of them with `warmup=N` | $\eta_{\text{inner}}(0)\cdot \frac{e + 1}{N}$ for $e < N$, then $\eta_{\text{inner}}(e - N)$ |

The cosine form is the annealing of SGDR (Loshchilov and Hutter) without restarts. Warm-up avoids
large first steps while Adam's averages are still empty. Being stateless, a scheduler needs nothing
saved to resume a training.

## Weight initialisation

`src/my_torch/Dense.cpp` (`lawOf`, `fillOrthogonal`); same names for `conv2d` and `token_dense`.
With $n_{\text{in}}$ inputs and $n_{\text{out}}$ outputs, weights have mean 0 and variance $\sigma^2$;
a uniform law $U(-a, a)$ with $a = \sqrt{3\sigma^2}$ has that variance. Biases start at 0.

| `init` | $\sigma^2$ | Rationale |
| --- | --- | --- |
| `uniform` (default of `dense`) | $1/(3 n_{\text{in}})$, i.e. $U(\pm\sqrt{1/n_{\text{in}}})$ | the original behaviour, kept for reproducibility |
| `lecun_uniform`, `lecun_normal` | $1/n_{\text{in}}$ | keeps the variance of $z$ equal to that of the inputs |
| `xavier_uniform`, `xavier_normal` | $2/(n_{\text{in}} + n_{\text{out}})$ | balances forward and backward variance, for `tanh`, `sigmoid` (Glorot and Bengio, 2010) |
| `he_uniform`, `he_normal` (default of `conv2d`, `token_dense`) | $2/n_{\text{in}}$ | ReLU zeroes half its inputs, so the variance is doubled (He et al., 2015) |
| `orthogonal` | orthonormal columns or rows (modified Gram-Schmidt on a Gaussian matrix) | preserves norms through deep linear stacks |

The argument behind LeCun, Xavier and He: for $z = \sum_{k=1}^{n_{\text{in}}} w_k x_k$ with
independent zero-mean terms, $\operatorname{Var}(z) = n_{\text{in}} \operatorname{Var}(w) \operatorname{Var}(x)$.
Choosing $\operatorname{Var}(w) = 1/n_{\text{in}}$ keeps activations at a constant scale from layer
to layer; the factor 2 compensates for ReLU. Measured in [06-initialisation.md](../performance/06-initialisation.md).
The attention projections use Xavier uniform and the `positional` table $N(0, 0.1^2)$ ([chapter 5](05-convolution-and-attention.md)).

## Regularisation

**L1 / L2 per layer** (`dense l1= l2=`): adds $\lambda_1 \sum|w| + \lambda_2 \sum w^2$ to the
objective, gradient $\lambda_1 \operatorname{sign}(w) + 2\lambda_2 w$ ([chapter 2](02-backpropagation.md)).
Under SGD, L2 shrinks each weight towards 0 at every step; under Adam, prefer `adamw`.

**Dropout** (`dropout rate=`, `src/my_torch/Dropout.cpp`, Srivastava et al., 2014). Inverted
dropout: in training, a mask $M$ has entries $0$ with probability `rate` and $1/(1 - \text{rate})$
otherwise; $Y = X \odot M$ and backward returns $G \odot M$. The scaling keeps
$\mathbb{E}[Y] = X$, so in prediction the layer is the identity. Monte Carlo dropout keeps the masks
on at prediction time to estimate uncertainty ([ADR-0034](../adr/0034-mc-dropout.md)).

**Batch normalization** (`batch_norm`, `src/my_torch/Normalization.cpp`, Ioffe and Szegedy, 2015).
Per column $j$, over the $m$ rows of the batch: $\mu_j$, $\sigma_j^2$ (biased, divided by $m$),
$\hat x = (x - \mu_j)/\sqrt{\sigma_j^2 + \epsilon}$, $y = \gamma_j \hat x + \beta_j$. Backward, with
$\hat d = g \odot \gamma$ and $r_j = 1/\sqrt{\sigma_j^2 + \epsilon}$:

$$d\gamma_j = \sum_i g_{ij}\hat x_{ij}, \qquad d\beta_j = \sum_i g_{ij}, \qquad dx_{ij} = \frac{r_j}{m}\Big(m\,\hat d_{ij} - \sum_k \hat d_{kj} - \hat x_{ij}\sum_k \hat d_{kj}\hat x_{kj}\Big).$$

Running averages, used in prediction, follow
$\text{running} \leftarrow (1 - \text{momentum})\,\text{running} + \text{momentum}\cdot\text{batch}$
(`momentum` 0.1, `epsilon` 1e-5), with the unbiased variance (divided by $m - 1$).

**Layer normalization** (`layer_norm`, Ba, Kiros and Hinton, 2016): the same formulas over the
features of each **row** instead of the rows of each column, identical in training and prediction.
`token_norm` applies it to each token ([chapter 5](05-convolution-and-attention.md)). Both were
measured in [17-normalization.md](../performance/17-normalization.md).

**Early stopping** (`train patience= min_delta= restore_best=`): stop when the validation loss has not
improved by `min_delta` for `patience` epochs, and restore the best epoch's weights
([ADR-0009](../adr/0009-training-loop.md)). The pushed network uses `patience=3`.

## Gradient clipping

`train clip_value=` clamps each gradient to $[-c, c]$; then `clip_norm=` rescales all gradients by
$\min(1, \tau / \lVert g \rVert_2)$, where $\lVert g \rVert_2$ is the norm over **every** parameter
of the network (`clip` in `src/my_torch/Network.cpp`). Both are off by default; a non-finite loss,
gradient or weight stops training with an error that suggests them.
