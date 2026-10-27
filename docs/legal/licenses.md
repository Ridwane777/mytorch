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

Versions, roles and replacements: [../dependencies.md](../dependencies.md).

## Data

| Data | Origin | License | What we publish |
| --- | --- | --- | --- |
| `dataset.zip` (723,019 labelled positions) | Given by the school with the subject | None stated, upstream origin unknown | Nothing of the files: statistics and graphs only (risk R1) |
| `my_torch-project.pdf` (subject) | the school | the school | Nothing (`.gitignore`) |
| Example and test positions in `tests/`, `docs/`, `bonus/` | Written by us or produced by our tools | Same as the project | Everything |
| `my_torch_network_final.nn` | Trained by us on the dataset | Same as the project | The weights. They are learned from the dataset but do not contain its positions |
