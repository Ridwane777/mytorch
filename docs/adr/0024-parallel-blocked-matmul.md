# ADR-0024: Parallel and blocked matrix product, bit-identical results

- **Status**: Accepted (2026-10-28)
- **Date**: 2026-10-27
- **Deciders**: Ridwane777 (issues #25 and #33)

## Context
The final network (ADR-0023, 901 → 256 → 128 → 5) is trained on the whole dataset, and almost all
of the training time is spent in the three matrix products of each `Dense` layer: `X·W` (forward),
`Xᵀ·G` (weight gradient) and `G·Wᵀ` (input gradient). Before this ADR they were a plain triple loop
on one thread, and the two backward products first copied a transposed matrix. The training machine
is a laptop with 12 hardware threads (i7-1355U); a shared VPS is available but slower and noisy.
Constraints: no external library doing the work for us (subject), standard C++20 only, and every
result reproducible: a run must give the same network whatever the number of threads.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| **Cache blocking + rows of the result split between threads (chosen)** | Each output element is still summed by one thread, in the same order (k ascending): results bit-identical to the plain loop for any thread count; `std::thread` only | Gains capped by the batch size (64 rows to split); a thread pool to maintain |
| Data parallelism over the batch (each thread a part of the batch, gradients summed) | Fewer synchronisations | The order of the gradient sum depends on the thread count: networks no longer bit-identical, results not reproducible |
| OpenMP (`#pragma omp parallel for`) | Two lines of code | Compiler extension and `libgomp` dependency; static schedule must be forced to keep determinism; less control over the pool |
| Clustering with the VPS (MPI or sockets) | More cores in total | Network latency per batch far above the batch compute time (about 1 ms); the VPS is shared and slower; large complexity |
| GPU (the RunPod A5000) | Massive parallelism | CUDA is a heavy dependency, double precision is slow on that card, and a new code path to verify |

## Decision
Blocked product (blocks of 64 inner indices × 256 columns) whose result rows are split into
contiguous ranges, one per thread of a persistent pool (`my_torch/Parallel.hpp`). Dedicated
`transposeMatmul` and `matmulTranspose` avoid the transpose copies. Small products (under 2¹⁶
multiply-adds) stay on the calling thread. The thread count is the hardware count, overridable by
`MY_TORCH_THREADS` or `setThreadCount`.

## Consequences
- ✅ Measured on the laptop (docs/performance/12-matmul.md): products 4.2 to 4.7× faster on one
  thread, up to 23× with 12 threads; final network training 2.4× faster from 1 to 12 threads, and
  62.4 s → 18.4 s on the 1 % pipeline against the previous code.
- ✅ Networks byte-identical for 1, 2, 4, 8 and 12 threads and identical to the code before this
  ADR (tested in `tests/my_torch/test_parallel.cpp` and by the benchmark).
- ❌ Scaling is far from linear (2.4× with 12 threads): only 64 rows to split, the laptop has 2
  performance cores and 8 efficiency cores, and it throttles. A bigger batch would scale better but
  changes the training.
- ❌ More code (pool, three kernels) and a concurrency risk, checked with ThreadSanitizer.
- Follow-up: revisit with data parallelism if reproducibility across thread counts stops mattering,
  or with a GPU if the network grows by an order of magnitude.
