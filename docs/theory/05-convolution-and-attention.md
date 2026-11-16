[Theory index](README.md) · previous: [Optimization](04-optimization.md) · next: [References](references.md)

# 5. Convolution and attention

A dense layer treats its 901 inputs as an unordered list: it does not know that square e4 is next
to e5. Convolution and attention give the network a structure over the 64 squares. Both are built
from the same idea as `Dense` (a matrix product forward, its transposes backward) and are checked
by the same finite differences ([chapter 2](02-backpropagation.md)). Decisions:
[ADR-0037](../adr/0037-convolutional-layers.md), [ADR-0041](../adr/0041-generic-planes-layer.md),
[ADR-0046](../adr/0046-residual-blocks.md), [ADR-0047](../adr/0047-attention-layers.md).

## From a flat row to planes or tokens

A `Matrix` row is still one example. An image-like example is stored **channel-major**: value of
channel $c$ at row $y$, column $x$ in column $(c \cdot H + y)\cdot W + x$
(`include/my_torch/Convolution.hpp`). The parameter-free `to_planes` reshapes the 901 inputs of
`fen-attacks-901` into **19 planes of 8 × 8**: the 12 piece planes and the 2 attack planes, plus one
constant plane per scalar (side to move and the 4 castling rights), each scalar repeated on its 64
cells. In backward, a cell gets its own gradient and a scalar the sum over its plane.

`to_tokens` does the same for attention (`include/my_torch/Attention.hpp`): token $t$ (one per
square, 64 tokens) gets cell $t$ of every plane, then the 5 scalars, so **64 tokens of 19
features**, stored token-major (feature $f$ of token $t$ in column $t \cdot 19 + f$).

## Convolution (`conv2d`)

`src/my_torch/Convolution.cpp`. $F$ filters of size $C \times k \times k$ slide over the input with
stride $S$ and zero padding $P$; output size $H' = \lfloor (H + 2P - k)/S \rfloor + 1$ (same for the
width). Like every library, it computes a cross-correlation:

$$Y_{f,y,x} = b_f + \sum_{c=1}^{C}\sum_{u,v=0}^{k-1} W_{f,c,u,v}\; X_{c,\;yS + u - P,\;xS + v - P}.$$

It is implemented as one matrix product (**im2col**). Each output position becomes a row of the
patch matrix $\Pi$ of shape $(B H' W') \times (C k^2)$; the weights are stored as
$W \in \mathbb{R}^{F \times C k^2}$. Then

$$Z = \Pi\, W^\top + \mathbf{1} b, \qquad dW \mathrel{+}= dZ^\top \Pi, \qquad db \mathrel{+}= \mathbf{1}^\top dZ, \qquad d\Pi = dZ\, W,$$

and **col2im** adds each row of $d\Pi$ back to the input cells it was copied from (a cell that
belongs to several patches receives the sum). These are exactly the `Dense` formulas with $\Pi$ in
the role of $X$: weight sharing just means that one weight's gradient sums over every position.
Samples are processed by blocks to bound the memory of $\Pi$.

**Pooling**, no parameters, no padding: `max_pool2d` sends each gradient to the position of the
window's maximum (the first one in reading order on ties); `avg_pool2d` shares it equally among the
$\text{size}^2$ cells; `global_avg_pool` averages each channel, $\partial / \partial x = g / (HW)$.

## Residual blocks and branches

`src/my_torch/Containers.cpp`. `residual` computes $y = F(x) + x$ for a sequence $F$ that keeps the
width, so backward returns $\partial L/\partial x = G + J_F^\top G$: the identity term lets the
gradient reach early layers unchanged. `concat` runs branches on the same input, puts their outputs
side by side, cuts the gradient into one slice per branch, and sums the branches' input gradients.

## Multi-head self-attention (`self_attention`)

`src/my_torch/Attention.cpp`. For one example, $X \in \mathbb{R}^{T \times d}$ ($T$ tokens of
dimension $d$), four projections with weights $d \times d$ and biases:

$$Q = XW_q + b_q, \quad K = XW_k + b_k, \quad V = XW_v + b_v.$$

They are split into $h$ heads of $d_h = d/h$ columns. For each head,

$$S = \frac{Q K^\top}{\sqrt{d_h}}, \qquad P = \operatorname{softmax}_{\text{rows}}(S), \qquad C = P V,$$

the heads' contexts are put side by side, and the output is $C W_o + b_o$ (Vaswani et al., 2017).
There is no mask: every square attends to every square. Backward, per head, with
$G$ the incoming gradient:

$$dC = G W_o^\top, \quad dV = P^\top dC, \quad dP = dC\, V^\top, \quad dS = P \odot \big(dP - \operatorname{rowsum}(dP \odot P)\big),$$

$$dQ = \frac{dS\, K}{\sqrt{d_h}}, \qquad dK = \frac{dS^\top Q}{\sqrt{d_h}}, \qquad dX = dQ\, W_q^\top + dK\, W_k^\top + dV\, W_v^\top,$$

and each projection gets $dW = X^\top dQ$ (resp. $dK$, $dV$) and $db$ the column sums. $dS$ is the
Softmax backward of [chapter 3](03-activations-and-losses.md), row by row. The $1/\sqrt{d_h}$ keeps
the scores at unit scale when $Q$ and $K$ have unit-variance entries, so the softmax does not
saturate. The bias $b_k$ gets a zero gradient: adding the same vector to every key shifts each row
of $S$ by a constant, which the softmax ignores (the gradient tests compare it in absolute terms,
`tests/my_torch/gradient_check.hpp`).

The other token layers: `token_dense` applies one `Dense` to every token (the batch is viewed as
$BT$ rows of $d$ features); `positional` adds a learned $T \times d$ table drawn from $N(0, 0.1^2)$,
whose gradient is $G$ summed over the batch; `token_norm` is `LayerNorm` on each token. A
transformer block in `configs/attention/` is built from these and `residual` (pre-norm: norm, then
attention or MLP, inside a residual).

## What was measured

- **CNN against dense** ([20-cnn.md](../performance/20-cnn.md)): `to_planes`, two `conv2d` 3 × 3,
  max pooling, then two `dense`. Trained on the same 10 % of the data, the CNN scores 0.9684 test
  accuracy against 0.8577 for the dense network, with 3.6 times fewer parameters but twice the time
  per epoch. The report warns that this dense network overfits on 10 % of the data; against the
  pushed dense network trained on all the data (0.9589 on the same positions), the gap is about 1
  point, with one seed. A CNN trained later on the full data reaches 0.9807 on the test split
  ([26-comparison.md](../performance/26-comparison.md)).
- **Residual blocks** ([28-residual.md](../performance/28-residual.md)): at the reduced budget, six
  convolutions improve on two, with or without shortcuts.
- **Mini transformer** ([29-attention.md](../performance/29-attention.md)): on 10 % of the training
  data and 10 epochs, one block scores 0.9167 and two blocks 0.9180, against 0.9417 for the CNN on
  the same budget, while taking 3 to 6 times longer to train and up to 2.9 GiB of memory. The
  attention layers stay in the library, tested; whether attention wins with the full data was not
  measured.

The convolution's inductive bias (local patterns, shared over the board) fits chess attacks, which
are local or along lines; attention has to learn that structure from the data, which is plausibly
why it needs more of it. That last point is an interpretation, not a measurement.
