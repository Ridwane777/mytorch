# ADR-0001: Python 3 with NumPy as the project language

- **Status**: Abandoned, superseded by [ADR-0002](0002-cpp-language.md)
- **Date**: 2026-10-21
- **Deciders**: Ridwane777

## Context

A project requirement is a language that runs on the target environment. Neural-network libraries
(PyTorch, TensorFlow…) are forbidden: forward propagation, backpropagation, optimizers and saving must be
written by the team. The deliverable also includes reproducible training scripts and benchmarks with charts.

## Options considered

1. **Python 3 + NumPy**: vectorized matrix computation, an ecosystem for charts (matplotlib), fast iterations.
2. **Pure Python 3** (no NumPy): no dependency, but matrix products in Python loops are very slow.
3. **C / C++**: maximum performance and memory control, but linear algebra, serialization and charts must
   be written or integrated, and the development cycle is longer.
4. **Rust**: performance and memory safety, but a learning curve and tooling less familiar to the team.

## Decision

**Python 3 with NumPy.** NumPy is used only as a linear algebra tool (arrays, matrix products); all the
neural-network logic (layers, activations, loss, backpropagation, optimizers, saving) is written by hand.
matplotlib is used only for benchmark charts, outside the evaluated executable.

## Consequences

✅ **Gained**
- Development speed: more time for experimentation (hyperparameters, regularization, benchmarks).
- Vectorized computation in C through NumPy: realistic mini-batch training without writing a BLAS.
- Charts and analysis scripts in the same language as the module.
- No compilation: the Makefile only makes the executables runnable.

❌ **Lost**
- Lower performance than optimized C/C++ on the non-vectorized parts (FEN parsing, training loops).
- External dependency (NumPy) to check on the target environment (local versions: Python 3.12.3,
  NumPy 1.26.4).
- Risk of misinterpretation: in a review we will need to show that NumPy is not a neural-network
  library and that no automatic differentiation is used.
- Parallelism (bonus) limited by the GIL; it will go through vectorization or multiprocessing.
