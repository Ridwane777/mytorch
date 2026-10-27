# Dependencies

For each dependency: role, tested version, license, and what happens if it disappears.
The executables have no runtime dependency other than the C++ standard library.

| Dependency | Role | Tested version | License | If it disappears |
| --- | --- | --- | --- | --- |
| g++ (C++20) | Compilation (ADR-0002) | 13.3.0 and 15.2.0 | GPL v3 with runtime exception | clang++ (21.1.8 builds the project and passes every test) |
| GNU make | Compilation (required by the subject) | — | GPL v3 | — |
| Criterion | Unit tests (dev only) | 2.4.1 and 2.4.2 | MIT | GoogleTest, Catch2 |
| clang-format | Code formatting (dev only) | 21.1.8 (not installed on every machine) | Apache 2.0 with LLVM exception | Manual review against `.clang-format` |
| GNU time (`/usr/bin/time`) | Time and peak memory in the benchmark and training scripts (dev only) | — | GPL v3 | `getrusage` in a small wrapper |
| git | Version control; `train_final.sh` writes the commit into `run.txt`, `bench_training_memory.sh` builds an older commit | — | GPL v2 | — |
| Python 3 and matplotlib | Graphs of the benchmarks in `scripts/` (dev only, never in the executables); `scripts/dataset_split.py` itself needs only the Python 3 standard library | Python 3.12.3 and 3.14.4, matplotlib 3.10.7 and 3.11.0 | PSF, matplotlib license (BSD-style) | Any plotting tool reading the CSV files |

Tested versions come from the two development machines (2026-10-24, updated 2026-10-27); the
benchmark documents state which machine produced each measurement.

## Services, secrets, quotas
None. The programs and the scripts call no external API, need no account, key or token, and have no
quota. The only external services are GitHub (repository, issues, pull requests) and, for some
measurements, a shared VPS whose access is personal and never written in the repository.

## Licenses
Each license and its constraint on our code: [legal/licenses.md](legal/licenses.md). None of them
constrains the executables, which only link the C++ standard library (GCC Runtime Library Exception).

To check: the versions available on the autograder image ("the dump").
