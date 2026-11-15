# Dependencies

For each dependency: role, tested version, license, and what happens if it disappears.
The executables have no runtime dependency other than the C++ standard library.

| Dependency | Role | Tested version | License | If it disappears |
| --- | --- | --- | --- | --- |
| g++ (C++20) | Compilation (ADR-0002) | 13.3.0 and 15.2.0 | GPL v3 with runtime exception | clang++ (21.1.8 builds the project and passes every test) |
| GNU make | Compilation (a project requirement) | - | GPL v3 | - |
| Criterion | Unit tests (dev only) | 2.4.1 and 2.4.2 | MIT | GoogleTest, Catch2 |
| clang-format | Code formatting (dev only) | 18.1.8, pinned: installed with pip in `.venv-format/` locally and with pipx in the CI (CONTRIBUTING.md) | Apache 2.0 with LLVM exception | Manual review against `.clang-format`; another version formats differently, so the CI check would fail |
| GNU time (`/usr/bin/time`) | Time and peak memory in the benchmark and training scripts (dev only) | - | GPL v3 | `getrusage` in a small wrapper |
| git | Version control; `train_final.sh` writes the commit into `run.txt`, `bench_training_memory.sh` builds an older commit | - | GPL v2 | - |
| Python 3 and matplotlib | Graphs of the benchmarks in `scripts/`, the architecture diagrams (`scripts/plot_architecture.py`, boxes drawn with matplotlib patches, ADR-0037) and data preparation (`scripts/dataset_split.py`, standard library only); never in the executables, and never computes a published metric: scores come from `scripts/score.cpp` and the library (ADR-0026) | Python 3.12.3 and 3.14.4, matplotlib 3.10.7 and 3.11.0 | PSF, matplotlib license (BSD-style) | Any plotting tool reading the CSV files |
| PyTorch (CPU wheel) and NumPy | **`bonus/comparaison/` only** (ADR-0036, ADR-0040): the PyTorch twin of the dense network and the CNN, the gradient check against autograd and the benchmarks of [performance/22-pytorch-reference.md](performance/22-pytorch-reference.md); installed in `bonus/comparaison/.venv` (git-ignored, `bonus/comparaison/requirements.txt`), never imported outside `bonus/comparaison/` (`scripts/check_comparison_isolation.sh`), never built or called by the Makefile; not needed to build, run or test the project | torch 2.15.0+cpu, numpy 2.5.4 (Python 3.12.3) | BSD-3-Clause style | The comparison cannot be re-run; the committed results stay valid as measurements of that date. Nothing else changes |
| scikit-learn and LightGBM | **`bonus/comparaison/` only** (ADR-0036, ADR-0043): the classical baselines of [performance/23-baselines.md](performance/23-baselines.md); same virtualenv, same isolation check, never built or called by the Makefile | scikit-learn 1.9.1, lightgbm 4.7.0 (Python 3.14.4) | BSD-3-Clause (scikit-learn), MIT (LightGBM) | The baselines cannot be re-run; the committed results stay valid as measurements of that date |
| pybind11 and NumPy | **`bonus/python/` only** (ADR-0044): build the Python module of the library and its tests and benchmark ([performance/25-python-bindings.md](performance/25-python-bindings.md)); installed in `bonus/python/.venv` (git-ignored, `bonus/python/requirements.txt`); pybind11 is header-only and used at build time; never built by the root Makefile | pybind11 3.1.0, numpy 2.5.4 (Python 3.14.4) | BSD-3-Clause (both) | The module cannot be built; a C API called with `ctypes` is the fallback (ADR-0044). Nothing else changes |
| chess.js 0.10.3, chessboard.js 1.0.0, jQuery 3.7.1 | Bonus chess GUI only (ADR-0029): legal moves and board drawing in the page; copied into `bonus/gui/web/vendor/`, so no network access is needed | 0.10.3, 1.0.0, 3.7.1 | BSD-2-Clause, MIT, MIT | The page can be rewritten with our own board; the server and the analysis do not depend on them |
| Wikipedia chess pieces (Cburnett) | Piece images of the GUI | - | CC BY-SA 3.0 | Any other piece set, or Unicode symbols |
| A web browser | Opening the GUI page | Chrome 2026 (headless screenshot) | - | Any recent browser |

Graphviz (`dot`) was considered for the architecture diagrams of `scripts/architecture.sh`: it is not
installed on the development machines and is not needed, matplotlib (already required) draws the
boxes and arrows, so the project gains no dependency.

Tested versions come from the two development machines (2026-10-24, updated 2026-10-27); the
benchmark documents state which machine produced each measurement.

## Services, secrets, quotas
None for the programs and for the scripts run on a laptop: they call no external API, need no account, key or token, and have no
quota. The only external services are GitHub (repository, issues, pull requests) and, for some
measurements, a shared VPS whose access is personal and never written in the repository.
Optionally, `scripts/aws_train.sh` ([ADR-0038](adr/0038-unattended-aws-training.md)) uses an AWS
project (EC2, S3, IAM) through the AWS CLI profile of the person who launches it; no key is
written in the repository.

## Licenses
Each license and its constraint on our code: [legal/licenses.md](legal/licenses.md). None of them
constrains the executables, which only link the C++ standard library (GCC Runtime Library Exception).

To check: the versions available on the target environment.
