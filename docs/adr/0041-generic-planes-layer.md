# ADR-0041: A generic planes layer, so that the library knows nothing about chess

- **Status**: Proposed
- **Date**: 2026-11-04
- **Deciders**: Ridwane777, after the first defense; review by Meric-BG (the change touches
  `include/my_torch/`)
- **Supersedes**: the `board_planes` layer of ADR-0037 (the rest of ADR-0037 stands)

## Context
`my_torch` must be a library independent of the chess project: the analyzer and the generator use
it through `include/my_torch/`, and `docs/library.md` says it "knows nothing about chess". The
`BoardPlanes` layer added with the CNN (ADR-0037) broke that promise: it hard-coded planes of 64
cells (an 8 × 8 board), and its comments described the `fen-attacks-901` encoding of the analyzer.
The documentation contradicted the code, and the question "is the module independent of the
project?" was asked at the first defense.

What the layer does is not specific to chess: it cuts a flat row into planes and repeats a few
scalars on planes of their own. Only the plane size (8 × 8) and the meaning of the columns are.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| **Generic `to_planes` with `height` and `width` in its line (chosen)** | No chess left in the library; the chess layout moves to the configuration, where it belongs; small change; the trained CNN keeps working | The configuration line is longer; the old name must still be read |
| Move the layer to the analyzer, registered as a custom layer | The library does not even contain a reshape it does not need | A registration API in `include/my_torch/` and in the file format, for one layer; `my_torch_analyzer` and the scripts would each have to register it |
| Keep `BoardPlanes`, document it as a chess helper | No work | The library stays tied to one project, and says the opposite |

## Decision
`BoardPlanes` becomes `ToPlanes`, line `to_planes inputs=… scalars=… count=… height=… width=…`:
the plane size is `height × width` instead of 64, and no comment of the library mentions chess. The
chess layout is written in `configs/cnn_network.conf` (`inputs=901 scalars=768 count=5 height=8
width=8`). The reader still accepts `board_planes inputs=… scalars=… count=…` as `to_planes` on
8 × 8, and writes the new name.

## Consequences
- ✅ `src/my_torch/` and `include/my_torch/` contain no chess concept: `docs/library.md` is true again.
- ✅ The layer serves any grid (a 3 × 4 case is tested), e.g. images or another board game.
- ✅ `bonus/networks/my_torch_network_cnn_epoch10.nn`, written with `board_planes`, still loads and
  predicts the same; saved again, it is written with `to_planes`.
- ❌ A public class is renamed (`BoardPlanes` → `ToPlanes`, one more constructor argument pair): code
  using the old name does not compile any more. Only `scripts/architecture.cpp` and the tests did.
- ❌ The reader keeps a compatibility branch for `board_planes`, to remove when no file uses it.
- Follow-up: the files written before keep the old name until they are saved again; the
  PyTorch twin (`bonus/comparaison/mytorch_net.py`) reads both names.
