# ADR-0044: Python bindings of the library with pybind11

- **Status**: Accepted (2026-11-04)
- **Date**: 2026-11-04
- **Deciders**: Ridwane777 (question raised after the first review); accepted by the pair

## Context
After the first review, we were asked whether the library can be used from another language,
the way NumPy is written in C and used from Python. It could not: the library is a static C++
archive (`build/libmy_torch.a`) that only a C++ program can call. Its public API
(`include/my_torch/`) is already independent of chess (ADR-0041), so the question is only how to
expose it, not what to expose. A core project constraint forbids neural-network libraries, not binding tools;
the bonus belongs in `bonus/` and must not change what the root Makefile builds.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| **pybind11 module (chosen)** | C++ classes, exceptions and NumPy arrays map directly; one source file; the method used by PyTorch itself | A dependency at build time (header-only); a module per Python version |
| A C API (`extern "C"`) in a shared library, called with `ctypes` | No dependency; usable from any language that calls C (Rust, Go…) | Every class flattened into functions and handles; arrays and errors converted by hand on both sides; more code for the same features |
| Cython or nanobind | Cython: the classic for NumPy-style code; nanobind: smaller and faster to build | Cython: a third language in the repository; nanobind: needs CMake in practice, the project uses make |
| Nothing: files only | No new code; the PyTorch twin already reads our configurations (ADR-0040) | Python never calls our code: only the files are shared, the answer to the question stays no |

## Decision
`bonus/python/bindings.cpp` builds a module `my_torch` with pybind11, from the public API only (the
library is not changed): `Network.load`, `Network.from_config`, `save`, `save_text`, `predict`
(NumPy arrays in and out, copied in one block, GIL released during the computation), `fit` (the
loss, optimizer, scheduler and settings of the network file, with `epochs`, `batch`, `seed` and
`patience` as overrides; returns one dict per epoch), `summary`, and the `inputs`, `labels` and
`encoding` of the network. `my_torch::Error` becomes the Python exception `my_torch.Error`
(a `RuntimeError`). The module compiles the library sources position-independent itself
(`make -C bonus python`); it is not part of `make` or of `make -C bonus all`.

## Consequences
- ✅ The library is usable from Python like a NumPy-style extension: measured on the 71,555 test
  positions, the probabilities are the same bits as the C++ ones and the labels are those of
  `my_torch_analyzer` ([performance/25-python-bindings.md](../performance/25-python-bindings.md)).
- ✅ No change to the library or to `include/my_torch/`: the bindings follow the API, they do not
  shape it.
- ✅ Six tests without the dataset (`make -C bonus python_tests`), also run against a build with
  AddressSanitizer and UBSan.
- ❌ Prediction through the module is 12 % slower on the whole test split (1.94 s against 1.73 s,
  medians of 5 runs): the inputs are copied from NumPy into a `Matrix` and back.
- ❌ One more build dependency (pybind11 3.1.0) in `bonus/python/.venv`, and a module built for one
  Python version (3.14 here).
- ❌ Only `Network` is exposed: layers, optimizers and data sources stay reachable through the
  configuration file only, not as Python objects.
- Follow-up: a zero-copy path (a `Matrix` viewing the NumPy buffer) if the copy cost matters; layer
  objects in Python if someone wants to build a network without a configuration file.
