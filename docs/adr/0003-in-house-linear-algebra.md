# ADR-0003: In-house linear algebra, in double, stored row by row

- **Status**: Accepted
- **Date**: 2026-10-21
- **Deciders**: Ridwane777, Meric-BG
- **Issue**: #1

## Context

A neural network spends almost all its time in matrix products: inputs × weights
for each layer, forward as well as backward. A batch of 32 positions encoded
on 768 inputs, multiplied by a layer of 128 neurons, already represents about 3 million
multiplications. A core project constraint forbids neural-network libraries, but says nothing about
linear algebra libraries. We must also be able to check every gradient with finite
differences at a tolerance of 1e-6 (issue #3).

## Options considered

| Option | Pros | Cons |
| --- | --- | --- |
| **In-house `Matrix` class** | No ambiguity with respect to the specification; all the computation is written and explainable by the team; no dependency on the target environment | Slower than an optimized library until #33 (blocking, SIMD) is done; risk of bugs |
| Eigen (header-only C++ linear algebra library) | Very fast, vectorized, proven; expressive API | May be judged contrary to the spirit of the specification; dependency to install or bundle; hides part of the evaluated work |
| BLAS / OpenBLAS (standard matrix computation library) | Fastest possible matrix product, multicore | System dependency to check on the target environment; low-level C API; same risk of misinterpretation as Eigen |

For number precision:

| Option | Pros | Cons |
| --- | --- | --- |
| **`double`** (≈ 16 digits) | Gradient checking at 1e-6 is possible; fewer accumulated rounding errors | Twice the memory; vectorized computation about twice as slow |
| `float` (≈ 7 digits) | Faster, lighter | Rounding exceeds the 1e-6 tolerance: the gradient test would fail on correct code |

## Decision

An **in-house `my_torch::Matrix` class**, in **`double`** (`using Scalar = double`), stored in **a
single contiguous array, row by row** (row-major): cell (r, c) is at index `r × columns + c`.
A batch is a matrix with **one row per example**. The matrix product runs its loops in
i-k-j order to read memory in order.

Why a single array rather than a `vector<vector<double>>`: the processor loads memory in
64-byte blocks (cache lines). Neighboring cells in the matrix are neighbors in memory, so
already loaded; an array of rows scatters the rows in memory and costs one allocation per row.

## Consequences

- ✅ All the computation is ours and explainable in a review; no runtime dependency.
- ✅ Gradient checking (#3) is possible with a strict tolerance.
- ✅ Switching to `float` later only requires changing one line (`Scalar`).
- ❌ Naive matrix product slow at first: optimization (blocking, SIMD, threads) becomes a task
  in its own right (#33, #25), to be measured against the naive version.
- ❌ Twice the memory and time of `float`.
- ❌ Possible numerical bugs in our code: covered by the unit tests and gradient
  checking.
- Follow-up: decide whether Eigen or BLAS is acceptable under the core constraint. If so, and if #33 is
  not enough, a new ADR may compare our optimized `Matrix` with Eigen on a benchmark.
