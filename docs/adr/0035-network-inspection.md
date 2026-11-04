# ADR-0035: Network inspection and saliency per square

- **Status**: Accepted (2026-11-04)
- **Date**: 2026-10-29
- **Deciders**: Ridwane777 (issue #36), approved by Meric-BG

## Context
Issue #36 asks for a model summary, weight and gradient histograms per epoch, and a saliency map
per square ("which squares decided Check"), shown on real positions, to understand the network
and to show it at the defense. `fit` gave no access to the network between epochs, and the
library had no gradient with respect to the input. The chess GUI (ADR-0029) analyses one position
at a time and can draw over the board.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| **Epoch callback in `TrainConfig` (chosen)** | Any measurement without touching the training loop, as Keras callbacks | A `std::function` in the settings, not saved in network files |
| Histograms written by `fit` itself | Nothing to write for the caller | One more file format in `fit`; every other measurement would need the same |
| **Saliency by occlusion: remove each piece (chosen for the GUI)** | Readable ("without this piece, no check"); works on a saturated network | One prediction per piece (up to 32); a removed piece can make an impossible position |
| **Gradient × input (also computed)** | One backward pass; the textbook saliency | Nearly flat on confident answers (softmax saturation): agrees with the oracle on 76 % only |
| Integrated gradients | Fixes the saturation of the plain gradient | A baseline position to choose and 20 to 50 backward passes per map |
| **Saliency code in `bonus/gui` (chosen)** | Shared by the GUI and the analysis script; uses the analyzer's encodings | Not in the library: it knows the chess encodings |

## Decision
The library gets `modelSummary(network)`, `TrainConfig::onEpoch(epoch, stats)` called by `fit`
after each epoch, and `Network::inputGradient(input, output)` (the parameters' gradients are left
as they were; BatchNorm gets a backward after a prediction forward). `gui::saliency` gives, per
square, the occlusion and the gradient × input of the probability of the network's answer. The GUI
returns both in `/api/analyze`, draws occlusion over the board by default, and opens a position
given as `?fen=`. `scripts/inspect.sh` measures them on the test split
(`docs/performance/19-inspection.md`).

## Consequences
- ✅ The network looks at the right square: on 97.6 % of its right answers on the test positions in
  check or mate, the most important square by occlusion is the checking piece named by the oracle.
- ✅ The GUI shows it on any position, for the defense; the maps also show protection of the mating
  piece and blocked escape squares.
- ✅ `inputGradient` is checked by finite differences, the callback does not change the training
  (same losses as `my_torch_analyzer --train`), the GUI saliency is tested on the fool's mate.
- ❌ Gradient × input is a poor map on this confident network (75.7 %); it stays as a second
  option in the GUI, explained there.
- ❌ Occlusion can build positions no game reaches; it measures the network's reaction only.
- ❌ The histograms show the gradients of the last batch of each epoch, not an average.
- Follow-up: integrated gradients if the gradient map is needed; the same maps for the
  convolutional network of #29.
