# Licenses and origin of the data

## The project
**No license: all rights reserved** by its two authors ([ADR-0025](../adr/0025-project-license.md)).
Reading the repository does not give the right to copy, modify or redistribute it.

## Dependencies
The executables (`my_torch_analyzer`, `my_torch_generator`) depend only on the C++ standard library.
Every other tool is used to build, test or draw graphs, and never ships in the executables.

| Dependency | Use | License | Constraint on our code |
| --- | --- | --- | --- |
| libstdc++ (GCC's C++ standard library) | Linked into the executables | GPL v3 with the GCC Runtime Library Exception | None: the exception allows distributing the compiled program under any license |
| g++ / clang++ | Compilers | GPL v3 / Apache 2.0 with LLVM exception | None (tools, not linked) |
| GNU make | Build | GPL v3 | None (tool) |
| Criterion | Unit tests only | MIT | None (not in the executables) |
| clang-format | Formatting only | Apache 2.0 with LLVM exception | None |
| Python 3 | Scripts in `scripts/` and `bonus/` | PSF License | None |
| matplotlib | Graphs of the scripts | matplotlib license (PSF-based, BSD-style) | None |
| PyTorch (CPU wheel) and NumPy | Reference implementation of `bonus/comparaison/` only (ADR-0036, ADR-0040), in `bonus/comparaison/.venv`; never built into, linked to or called by the project | BSD-3-Clause style licenses | None for the executables (not linked, not distributed); the specification's ban on neural-network libraries is the real constraint, see ADR-0036 |
| scikit-learn, LightGBM (and SciPy, joblib, threadpoolctl pulled by scikit-learn) | Classical baselines of `bonus/comparaison/` only (ADR-0036, ADR-0043), in `bonus/comparaison/.venv` | BSD-3-Clause (scikit-learn, SciPy, joblib, threadpoolctl), MIT (LightGBM) | None for the executables (not linked, not distributed) |
| pybind11, NumPy | Python module of `bonus/python/` only (ADR-0044), in `bonus/python/.venv`; pybind11's headers are compiled into the module, never into the executables | BSD-3-Clause (both) | Keep pybind11's copyright notice if the module is ever distributed; nothing for the executables |
| chess.js 0.10.3 | Rules of the game in the bonus GUI page (`bonus/gui/web/vendor/`) | BSD-2-Clause | Keep its copyright notice, which is in the copied file |
| chessboard.js 1.0.0, jQuery 3.7.1 | Board drawing in the bonus GUI page | MIT | Keep the copyright notice, which is in each copied file |
| Chess piece images (Cburnett, Wikimedia Commons) | Pieces of the bonus GUI | CC BY-SA 3.0 | Credit the author and keep the license: stated in the page footer and here; the images are shared under the same license |

Versions, roles and replacements: [../dependencies.md](../dependencies.md).

## Data

| Data | Origin | License | What we publish |
| --- | --- | --- | --- |
| `dataset.zip` (723,019 labelled positions) | Provided by Epitech | None stated, upstream origin unknown | `dataset.zip`, redistributed with Epitech's authorization (risk R1) |
| Example and test positions in `tests/`, `docs/`, `bonus/` | Written by us or produced by our tools | Same as the project | Everything |
| `my_torch_network_final.nn` | Trained by us on the dataset | Same as the project | The weights. They are learned from the dataset but do not contain its positions |
