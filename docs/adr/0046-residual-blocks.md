# ADR-0046: Residual connections and concatenation as container layers, the network stays a sequence

- **Status**: Accepted (2026-11-05)
- **Date**: 2026-11-05
- **Deciders**: Ridwane777 (issue #31); option A accepted by the pair before any code; the code
  that changes `include/my_torch/` and the network file goes through its own reviewed PR

## Context
Issue #31 asks for residual connections (`output = F(x) + x`) and concatenation, and for a
residual network that trains, saves and reloads. Residual blocks are what lets a network be deep:
the gradient reaches the first layers through the shortcut, and a block that has nothing to add
only has to bring its weights near zero. They are the base of ResNet and of the chess networks
of AlphaZero and Leela Chess Zero, and every transformer block of #32 contains two of them.

Today a `Network` is a sequence (`std::vector<std::unique_ptr<Layer>>`, ADR-0004): `forward` calls
the layers in order and `backward` in reverse, and the network file lists them one per record
(ADR-0012, binary since ADR-0042). A shortcut sends one output to two places, which a plain
sequence cannot express. What exists that a change must keep working: the `Layer` interface
(`forward`, `backward`, `parameters`, `penalty`, `save`, `saveState`/`loadState`, `outputWidth`),
checkpoints (ADR-0032), Monte Carlo dropout (ADR-0034), saliency (`inputGradient`, ADR-0035), the
fused softmax and cross-entropy (ADR-0007), and every network file already written.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| **A. Container layers in the sequence (chosen)**: a `Residual` layer holds a sub-sequence and returns `F(x) + x`; a `Concat` layer holds branches and returns their outputs side by side | `Network` stays a sequence: `fit`, the optimizers, the checkpoints and the summary see one more `Layer`; the forward and backward of a container are a few lines (backward: the gradient through the sub-sequence plus the gradient itself); old files unchanged; nesting comes for free (a container may hold containers) | Only shortcuts that start and end around one block: no connection from any layer to any other; the code that looks at the top-level layers only (dropout detection of Monte Carlo dropout, width checks of the normalizations) must look inside containers |
| B. A graph of layers: each layer names its inputs, forward runs them in a computed order | Any architecture (branches, skips across many blocks, several outputs) | `forward`, `backward` (gradients summed where an output feeds several layers), `fit`, the checkpoints and the file format all rewritten; a second file layout to read beside the sequential one; the largest risk of breaking what works, for architectures we do not plan to train |
| C. A fixed residual network type (a `ResNet` class with N blocks) | Least code | Not a library feature: no concatenation, no reuse in the transformer blocks of #32 |
| D. Nothing | No risk | The \"going further\" track of the specification and #32 stay out of reach; the depth of our networks stays limited |

## Decision
Option A, in four parts.

1. **`Residual`** (`include/my_torch/Residual.hpp`): owns a sequence of layers. Forward:
   `F(x) + x`, where `F` is the sub-sequence; the sub-sequence must give back the width it receives
   (`outputWidth(w) == w`, else `my_torch::Error` when the network is built or loaded): an identity
   shortcut only, no projection. Backward: `g` through the sub-sequence in reverse, plus `g` for the
   shortcut. `parameters`, `penalty`, `saveState`/`loadState` and `outputWidth` are those of the
   sub-sequence; `training` is passed down.
2. **`Concat`**: owns several branches (each a sequence) that read the same input; output: the
   branches' outputs side by side, width the sum of theirs. Backward: the gradient cut into one
   slice per branch, each through its branch, the input gradients summed.
3. **File format**: a container is one record followed by the records of its layers, which its
   count covers; `layers N` keeps counting the top-level layers only.
   ```
   layers 6
   to_planes inputs=901 scalars=768 count=5 height=8 width=8
   conv2d channels=19 height=8 width=8 filters=32 kernel=3 stride=1 padding=1
   relu
   residual layers=3
   conv2d channels=32 height=8 width=8 filters=32 kernel=3 stride=1 padding=1
   relu
   conv2d channels=32 height=8 width=8 filters=32 kernel=3 stride=1 padding=1
   concat branches=2
   branch layers=1
   …
   branch layers=2
   …
   …
   ```
   The same in configurations (ADR-0013). In the binary format (ADR-0042), the words `residual`,
   `concat` and `branch` are appended to the keyword table: files already written keep their codes
   and stay readable; a program from before this ADR refuses a file with a container ("unknown
   layer type"), which is expected.
4. **What must look inside containers**: the dropout detection of Monte Carlo dropout and the width
   checks of batch and layer normalization, through a way for a container to visit its layers
   (decided in the implementation, reviewed with it). The fused softmax stays a top-level last
   layer, as today.

Each new backward gets its finite-difference test (AGENTS.md), and a residual CNN is measured
against the current CNN (98.07 % on the test split) in a performance report.

## Consequences
- ✅ Residual networks and branches without rewriting `Network`, `fit` or the optimizers; every
  existing network file stays valid, bit for bit.
- ✅ The building block that #32 needs (attention + shortcut, MLP + shortcut).
- ✅ Nesting is allowed by construction (a residual block inside a branch, and so on).
- ❌ No connection between arbitrary layers (DenseNet-style skips across many blocks, several
  inputs or outputs): that would take option B and its own ADR.
- ❌ Identity shortcuts only: a block that changes the width (more filters) needs a projection on
  the shortcut, left for later; until then, the width changes happen between blocks.
- ❌ A change to `include/my_torch/` and to the network file: reviewed by both members, and network
  files with containers cannot be read by earlier builds.
- ❌ Code that walks the layers must now know containers (Monte Carlo dropout, normalization
  checks, the layer summary): a source of bugs, covered by tests on nested networks.
- Follow-up: the measurement of a residual CNN; a projection shortcut if a measured architecture
  needs it; option B only if an architecture we decide to train cannot be written with containers.
