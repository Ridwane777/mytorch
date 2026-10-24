# Dependencies

For each dependency: role, tested version, license, and what happens if it disappears.
The executables have no runtime dependency other than the C++ standard library.

| Dependency | Role | Tested version | License | If it disappears |
| --- | --- | --- | --- | --- |
| g++ (C++20) | Compilation (ADR-0002) | 13.3.0 | GPL v3 with runtime exception | clang++ 18 (tested locally) |
| GNU make | Compilation (required by the subject) | — | GPL v3 | — |
| Criterion | Unit tests (dev) | 2.4.1 | MIT | GoogleTest, Catch2 |
| clang-format | Code formatting (dev) | — | Apache 2.0 with LLVM exception | — |

To check: the versions available on the autograder image ("the dump").
