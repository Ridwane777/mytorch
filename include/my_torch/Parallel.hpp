#pragma once

#include <cstddef>
#include <functional>

namespace my_torch {

// A small pool of worker threads for the matrix products (ADR-0024). The threads are created once,
// on first use, and reused. Their number comes from the MY_TORCH_THREADS environment variable,
// else from the number of hardware threads. Results never depend on it: each task computes its
// own part of the output in the same order as a single thread would.

// Number of threads used by parallelFor (at least 1).
std::size_t threadCount();
// Changes it for the following calls (tests and benchmarks); 0 means the hardware default.
void setThreadCount(std::size_t count);

// Calls body(begin, end) on disjoint ranges covering [0, count), on up to threadCount() threads
// including the caller, and returns when all are done. body must not throw.
void parallelFor(std::size_t count, const std::function<void(std::size_t, std::size_t)> &body);

} // namespace my_torch
