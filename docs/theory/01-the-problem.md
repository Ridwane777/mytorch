[Theory index](README.md) · next: [Backpropagation](02-backpropagation.md)

# 1. The problem

## The task

`my_torch_analyzer` reads one chess position per line, in FEN notation, and answers `Nothing`,
`Check` or `Checkmate`, with the side that gives it. The networks of the project are therefore
**5-class classifiers**, not 3-class ones: the labels of every configuration are, in order,

| Index | Label |
| --- | --- |
| 0 | `Nothing` |
| 1 | `Check White` |
| 2 | `Check Black` |
| 3 | `Checkmate White` |
| 4 | `Checkmate Black` |

(`configs/final_network.conf`). "Check White" means White gives check. How to read a FEN line and
the chess rules that matter are explained in [reading-a-fen.md](../reading-a-fen.md); the parser is
`src/analyzer/Fen.cpp` ([ADR-0005](../adr/0005-fen-parsing.md)).

Formally, the network is a function $f_\theta : \mathbb{R}^n \to \Delta^4$ from an input vector
$x$ to a probability distribution $p$ over the 5 labels (its last layer is a softmax, see
[chapter 3](03-activations-and-losses.md)). The predicted label is $\arg\max_c p_c$. Training looks
for parameters $\theta$ that minimise the average cross-entropy between $p$ and the one-hot target
$t$ (the vector with a 1 at the index of the true label) over the training positions.

## From a FEN to an input vector

A network does not read text: an `Encoding` turns a parsed `Position` into a row of numbers
(`src/analyzer/Encoding.hpp`, `src/analyzer/Encoding.cpp`, [ADR-0016](../adr/0016-input-encoding.md)).
The network file names its encoding (`input fen-attacks-901 901`), and the analyzer looks it up by
name. Squares are indexed as in `Position`: $s = 0$ is a8, then along the rank, $s = 63$ is h1.

| Name | Width | Content |
| --- | --- | --- |
| `fen-onehot-768` | 768 | 12 planes of 64 squares, order `P N B R Q K p n b r q k`: $x_{64k+s} = 1$ when piece $k$ is on square $s$ |
| `fen-onehot-773` | 773 | the 768 above, then $x_{768} = 1$ when White is to move, $x_{769..772}$ = castling rights `K Q k q` |
| `fen-signed-385` | 385 | 6 planes `P N B R Q K`, $+1$ for a white piece, $-1$ for a black one; then $x_{384}$ = White to move |
| `fen-attacks-901` | 901 | the 773 above, then 64 inputs "square attacked by White" and 64 "square attacked by Black" |

In `fen-attacks-901`, a square is attacked when a piece could capture there, pins ignored, and a
sliding piece attacks up to and including the first piece in its way. This is **feature
extraction**: it hands the network part of the rules (who attacks what) but not the verdict, since
whether a check is a checkmate still depends on every legal reply. The clocks and the en passant
square are not encoded.

The one-hot encodings are sparse binary vectors: with $P$ pieces on the board, exactly $P$ of the
768 piece inputs are 1. This matters for chapter 2: in the first `Dense` layer, only the weight rows
of the active inputs receive a non-zero gradient for that example.

The encoding was chosen by measurement on the validation split
([03-encodings.md](../performance/03-encodings.md)): `fen-attacks-901` scored best of the four, and it
is the input of the pushed network (`fen-attacks-901 → 256 → 128 → 5`,
[ADR-0023](../adr/0023-final-architecture.md)). Two label-preserving board symmetries are used to
augment the training set ([ADR-0019](../adr/0019-symmetry-augmentation.md),
`src/analyzer/Symmetry.cpp`); [10-final-network.md](../performance/10-final-network.md) measures
their effect.

For the convolutional and attention networks, the flat 901-wide row is reshaped inside the network
(`to_planes`, `to_tokens`) rather than by another encoding: see [chapter 5](05-convolution-and-attention.md).

## The data and its imbalance

The dataset has 723,019 lines, 715,559 positions after removing duplicates, split 80 / 10 / 10 per
label into training (572,446), validation (71,558) and test (71,555) sets that share no position
([01-dataset.md](../performance/01-dataset.md), [ADR-0017](../adr/0017-dataset-split.md)).

The classes are **imbalanced**, but moderately: `Nothing` is 41.5 % of the lines and
`Checkmate Black`, the rarest, 11.0 %, a ratio of 3.8 to 1 ([01-dataset.md](../performance/01-dataset.md)).
A model that always answered `Nothing` would score about 41.5 %.

The library supports one loss-side remedy, **class weights** in the cross-entropy
(`loss cross_entropy weights=w1,w2,…`, [chapter 3](03-activations-and-losses.md)). The
project also compared oversampling and undersampling of the data. On a 20,000-position sample, no
option clearly beat doing nothing (macro F1 54.9 without handling, 54.3 with weighting, with
overlapping ranges over three runs), so the pushed network uses no class weights
([04-imbalance.md](../performance/04-imbalance.md), [ADR-0018](../adr/0018-class-imbalance.md)).

Because accuracy alone hides the minority classes, the reports also give per-class precision,
recall and F1 and their unweighted mean, the **macro F1** (`src/my_torch/Metrics.cpp`,
[ADR-0011](../adr/0011-classification-metrics.md)). For class $c$, with $TP_c$, $FP_c$, $FN_c$ the
true positives, false positives and false negatives:

$$\text{precision}_c = \frac{TP_c}{TP_c + FP_c}, \qquad \text{recall}_c = \frac{TP_c}{TP_c + FN_c}, \qquad F1_c = \frac{2\,\text{precision}_c\,\text{recall}_c}{\text{precision}_c + \text{recall}_c}.$$

## What the final network achieves

The pushed dense network scores **0.9615 accuracy and 0.9478 macro F1** on the 71,555 test
positions; most of its errors confuse a check with a checkmate of the same side
([10-final-network.md](../performance/10-final-network.md)). That confusion is the expected hard
case: telling them apart requires knowing whether the side in check has any legal move, a global
property of the position that the inputs do not give directly.
