# Dependencies

For each dependency: role, tested version, license, and what happens if it disappears.
The executables have no runtime dependency other than the C++ standard library.

| Dependency | Role | Tested version | License | If it disappears |
| --- | --- | --- | --- | --- |
| g++ (C++20) | Compilation (ADR-0002) | 13.3.0 and 15.2.0 | GPL v3 with runtime exception | clang++ (21.1.8 builds the project and passes every test) |
| GNU make | Compilation (required by the subject) | — | GPL v3 | — |
| Criterion | Unit tests (dev only) | 2.4.1 | MIT | GoogleTest, Catch2 |
| clang-format | Code formatting (dev only) | not installed on every machine | Apache 2.0 with LLVM exception | Manual review against `.clang-format` |
| Python 3 and matplotlib | Graphs of the benchmarks in `scripts/` (dev only, never in the executables); `scripts/dataset_split.py` itself needs only the Python 3 standard library | Python 3.12.3 and 3.14.4, matplotlib 3.10.7 and 3.11.0 | PSF, matplotlib license (BSD-style) | Any plotting tool reading the CSV files |

Tested versions come from the two development machines (2026-10-24); the benchmark documents state
which machine produced each measurement.

To check: the versions available on the autograder image ("the dump").
