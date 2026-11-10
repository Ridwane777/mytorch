# ADR-0002: C++20 as the project language

- **Status**: Accepted
- **Date**: 2026-10-21
- **Deciders**: Ridwane777, Meric-BG
- **Supersedes**: [ADR-0001](0001-python-numpy-language.md) (Python + NumPy, which stayed at the proposal stage)

## Context

The specification accepts any language that runs on the target environment and forbids
neural-network libraries. All the computation (matrix products, backpropagation,
optimizers) must therefore be written by the team. Training covers 723,019 positions: computation
speed directly limits the number of possible experiments (hyperparameter search,
encoding comparisons). The specification also rewards parallelism as a bonus.

## Options considered

| Option | Pros | Cons |
| --- | --- | --- |
| Python + NumPy (ADR-0001) | Fast development, charts in the same language | NumPy dependency to get approved; NumPy does part of the computation for us; slow non-vectorized loops; parallelism limited by the GIL |
| Pure Python | No dependency | Matrix products in Python loops: training on 723,000 positions too slow |
| **C++20** | Native performance, control over memory and computation, standard threads (`std::thread`), compiler available on the image, every operation is written and explainable by the team | Linear algebra, serialization and parsing to write; possible memory errors; longer development cycle |
| Rust | Performance and memory safety | Tooling and language less mastered by the pair |

## Decision

**C++20**, built by a Makefile (a project requirement), with no runtime dependency other than the
standard library. The `my_torch` module is a static library (`libmy_torch.a`) linked into
`my_torch_analyzer` and `my_torch_generator`. Unit tests with Criterion (a lightweight C unit-testing framework).

Scripts outside the executables (benchmark charts, dataset analysis) can stay in Python:
they are not evaluated by the evaluation harness and are not part of the network.

## Consequences

- ✅ Fast training, so more experiments and benchmarks in the same time.
- ✅ No ambiguity about the library ban: all the computation is ours.
- ✅ Native parallelism possible for the bonus (`std::thread`).
- ❌ Matrix class and operations to write and optimize ourselves (cache, vectorization), with the
  risk of numerical bugs: hence the finite-difference gradient tests (issue #3).
- ❌ Possible memory errors: `make debug` builds with AddressSanitizer and UBSan.
- ❌ More code to write than in Python for the same result.
- Follow-up: check the g++ and Criterion versions on the target environment (locally: g++ 13.3,
  Criterion 2.4.1). The choice between a linear algebra library (Eigen…) and in-house code will
  be the specification of a separate ADR (issue #1).
