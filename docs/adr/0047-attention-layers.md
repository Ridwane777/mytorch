# ADR-0047: Attention layers for a mini transformer, built from the residual container

- **Status**: Accepted (2026-11-08)
- **Date**: 2026-11-08
- **Deciders**: the pair (issue #32); accepted by both members after the review of PR #112. The code
  changes `include/my_torch/` and the network file, so it goes through a reviewed PR.

## Context
Issue #32 asks for a small transformer: token layers, a positional embedding and multi-head
self-attention. The question it answers is whether a model that relates the squares of the board to
each other classifies the positions better than the dense network and the CNNs we have.

What the current networks cannot do well. A king in check is attacked by a piece that can be on the
other side of the board: the relation that decides `Check` or `Checkmate` links squares far apart.
A convolution (ADR-0037) sees a 3 × 3 window per layer, so two layers still see a 5 × 5 area at
most; a dense network (ADR-0023) sees every input but ignores that the 64 squares share the same
structure, and has to learn each square separately. Self-attention lets every square read every
other square with the same weights, whatever their distance.

What exists that the change must keep working: the `Layer` interface (ADR-0004), the sequential
`Network`, the container layers `residual` and `concat` (ADR-0046), checkpoints (ADR-0032), the
saliency of ADR-0035 (a backward after a prediction forward), the fused softmax and cross-entropy
(ADR-0007), the binary format (ADR-0042) and every network file already written. The encoding
`fen-attacks-901` stays as it is (ADR-0016): columns [0, 768) hold 12 piece planes of 64 squares,
[768, 773) five scalars (side to move, four castling rights), [773, 837) the squares attacked by
White and [837, 901) the squares attacked by Black.

The width of a row stays the only thing a layer receives: a `Matrix` row is one sample (ADR-0037).
A sequence of tokens is therefore stored flattened, token-major: the value of feature `f` of token
`t` sits in column `t·F + f`, and the width inside the network is always `tokens × dim`.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| **A. Five small layers, the block built from `residual` (chosen)**: `to_tokens`, `token_dense`, `positional`, `self_attention`, `token_norm` | Each layer does one thing and has its own finite-difference test; a pre-norm block is `residual[token_norm, self_attention]` then `residual[token_norm, token_dense, relu, token_dense]`, with no new container; depth, width and heads are configuration; the layers can serve other models | Several layers and several lines per block in a configuration; five new words in the binary format |
| B. One `transformer_block` layer | Short configuration, one record per block | One large class with its own backward (attention, norms, MLP and both shortcuts), hard to test piece by piece; duplicates `residual`, `layer_norm` and `dense`; any variation (post-norm, no MLP) is new code |
| C. Sinusoidal or 2D positions instead of learned ones | No parameters; 2D (rank and file) encodes the geometry directly | A fixed pattern cannot adapt to what chess needs (the same offset means different things for a knight and a rook); the 64 squares are fixed, so a learned table has no generalisation to lose |
| D. Mean pooling of the tokens before the head | Fewer head parameters; the head does not depend on the number of tokens | The average erases which square holds the king or the attacker; the head `tokens·dim → 128 → 5` keeps the position of every token |
| E. Post-norm blocks (norm after the shortcut) | The original transformer | Harder to train without warm-up; the gradient crosses a norm on the shortcut path, which defeats the purpose of the identity shortcut (ADR-0046) |
| F. A new 64-token encoding in the analyzer (one vector per square) | No scalar repeated on every token inside the library | Changes `src/analyzer/Encoding.hpp`, a second encoding to keep, and the trained networks and the other experiments no longer share their input; the reshape is cheap and generic |

## Decision
Option A, with these choices:

1. **`to_tokens inputs=… scalars=… count=… tokens=…`**: the input row holds planes of `tokens`
   cells and `count` scalars starting at column `scalars`, with the same validation as `to_planes`
   (`cells = tokens`). Each token receives one value from every plane, then the scalars repeated:
   `F = (inputs - count) / tokens + count` features. For `fen-attacks-901`: 64 tokens of
   14 planes (12 pieces, 2 attack bits) + 5 scalars = 19 features per square. Backward: a cell gets
   its own gradient, a scalar the sum over all tokens. Parameter-free.
2. **`token_dense tokens=… inputs=… outputs=…`**: one `inputs × outputs` matrix and one bias shared
   by all tokens (a dense layer applied to every token), the usual initialisations with
   `fan_in = inputs`, the `l1`/`l2` penalty of `dense`. Implemented as the product of the
   `(batch·tokens) × inputs` view of the batch by the weights, reusing the parallel product.
3. **`positional tokens=… dim=…`**: a learned `tokens × dim` table added to every sample, drawn
   from N(0, 0.1²) with the seed of the layer. Learned rather than sinusoidal because the 64
   squares are fixed (option C).
4. **`self_attention tokens=… dim=… heads=…`**: `dim` divisible by `heads`; projections `Wq`, `Wk`,
   `Wv`, `Wo` (`dim × dim`) with biases; per head, scaled dot-product scores `QKᵀ / sqrt(dim/heads)`,
   a softmax per row, no mask (every square sees every square), the heads concatenated then
   projected by `Wo`. The backward is derived by hand (`dV`, `dP`, `dS = P ∘ (dP − rowsum(dP ∘ P))`,
   `dQ`, `dK`, the projections); in prediction the attention matrices are not kept and the backward
   recomputes them, which bounds the memory of the 4096-row evaluation chunks.
5. **`token_norm tokens=… dim=… epsilon=…`**: layer normalization over the `dim` features of each
   token, with `gamma` and `beta` of size `dim` (the code of `layer_norm` applied to the
   `(batch·tokens) × dim` view).
6. **Block and head**: pre-norm, from `residual` only: `residual[token_norm, self_attention]` then
   `residual[token_norm, token_dense dim→4·dim, relu, token_dense 4·dim→dim]`. The head flattens
   (the row is already `tokens·dim` wide): `dense tokens·dim → 128`, `relu`, `dense 128 → 5`,
   `softmax`. Baseline: `dim = 32`, `heads = 4`, one block; a two-block variant is compared.
7. **Files**: the layers are written as `key=value` lines like `conv2d` followed by their
   parameters (weights row by row, then biases). In the binary format the words `to_tokens`,
   `token_dense`, `positional`, `self_attention`, `token_norm`, `tokens`, `dim` and `heads` are
   appended to the keyword table; the version stays 2 and every file already written loads
   unchanged. A program from before this ADR refuses a file with these layers ("unknown layer
   type"), which is expected.

Each new gradient gets a finite-difference test (AGENTS.md). Whether the transformer beats the
dense network and the CNNs is measured afterwards in a performance report; this ADR only decides
what is built.

## Consequences
- ✅ A transformer without a new container, a new network type or a change to `fit`: the blocks are
  configuration, and nesting, checkpoints, saliency and the optimizers work as for any layer.
- ✅ The encoding, the analyzer and every file already written are untouched.
- ✅ Each layer is tested alone against finite differences, and the block as a whole inside a
  `residual`.
- ❌ Attention costs O(tokens² · dim) time and O(tokens²) memory per head and per sample in
  training; with 64 tokens, training will be slower than the dense network and the CNN for the same
  number of epochs, and the 4096-row evaluation chunks need the recomputation described above.
- ❌ The number of tokens is fixed by the configuration (64 for a board); a model trained for 64
  squares cannot read another size, and the learned positions do not transfer.
- ❌ The 5 game scalars are repeated on all 64 tokens (320 of the 1216 input values of the
  tokens), a crude way to give every square the side to move and the castling rights.
- ❌ Eight more words in the binary keyword table (`words.size()` goes from 91 to 99, below the
  256 allowed by the one-byte code) and a public header (`include/my_torch/Attention.hpp`) that
  both members must review; files with these layers cannot be read by earlier builds.
- ❌ A configuration needs about ten lines per block; a bad `tokens × dim` chain is an error at
  load time, not a shape the library fixes silently.
- Follow-up: a performance report (accuracy, time per epoch, memory) against the dense network and
  the CNNs, done in [29-attention.md](../performance/29-attention.md); revisit if training time
  makes the comparison unfair, or if a mean-pooled head or sinusoidal positions turn out
  equivalent and cheaper.
