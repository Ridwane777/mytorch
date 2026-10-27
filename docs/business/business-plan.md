# Business plan: my_torch, a teaching library

> An exercise for the defense. Market figures are not given when we could not source them: the
> sizing below says what to measure, not invented numbers.

## 1. Summary
my_torch is a neural-network library written from scratch in standard C++20, small enough to be read
in full (about 3,700 lines for the library), with a complete worked example: a chess position
classifier that reaches 96.2 % accuracy on positions it has never seen
([final-network.md](../performance/final-network.md)). The product idea: sell it as **teaching
material** for deep-learning courses that want students to see every computation, from the matrix
product to backpropagation, instead of calling a black box.

## 2. The problem
- Deep-learning courses mostly teach with PyTorch or TensorFlow: students learn the API, not what
  happens inside (gradients, optimizers, initialisation, numerical stability).
- Writing a library from scratch is the classic exercise to learn it, but teachers must then build
  the subject, the tests and the reference solution themselves, and keep them correct.
- Existing from-scratch libraries are either tiny demos (a scalar autograd in a few hundred lines)
  or full frameworks too large to read.

## 3. The solution
- **The library**: dense layers, activations, losses, five optimizers, schedulers, regularisation,
  early stopping, metrics, a versioned network file, all in readable C++ with no dependency
  (`docs/library.md`).
- **The proofs**: every gradient is checked against finite differences, every design choice has an
  ADR with its trade-offs (25 so far), every performance claim has a reproducible script and a
  graph.
- **The worked example**: the chess analyzer, from raw FEN to a trained network, with the dataset
  analysis, the trials and the final evaluation.
- **What we would sell**: a course kit built on it: the library, step-by-step exercises (one layer,
  one optimizer at a time, each with its finite-difference test), and the teacher's solution.

## 4. Market and competition
**Customers**: engineering and computer-science schools, bootcamps, corporate training teams that
teach machine learning; individual learners as a secondary audience.

**Sizing, to measure before any investment**: number of schools teaching deep learning in France,
share of courses with a "from scratch" project, price paid today for course material. We have no
sourced figure for these: the first action of the roadmap is to ask the pedagogical teams of a few
schools, starting with the school.

| Alternative | Strength | Weakness for this use |
| --- | --- | --- |
| PyTorch, TensorFlow, JAX | Industry standards, huge ecosystem | Teach usage, hide the internals; far too large to read |
| micrograd (A. Karpathy) | Tiny, famous, free | Scalar values only, a demo rather than a course |
| tinygrad | Small real framework | Aimed at performance, not at teaching step by step |
| "From scratch" books and courses | Pedagogical | Usually in Python with NumPy; no tested reference library to build on |
| Teachers' own projects | Free, tailored | Cost of writing and maintaining subject, tests and solution |

**Our difference**: a complete, tested, documented reference in C++ with its decision records and
measurements, which is exactly what a teacher needs to grade a from-scratch project.

## 5. Business model
| Offer | For whom | Revenue |
| --- | --- | --- |
| Library and documentation | Everyone | Free (needs a license first, see ADR-0025) |
| Course kit: exercises, tests, teacher's solution | Schools, bootcamps | Yearly license per school |
| Workshops: two-day "build your own deep-learning library" | Companies, schools | Per session |

No price is set: it depends on the interviews of section 4. The costs to cover are in
[costs.md](costs.md): the product runs on the students' own machines, so there is no hosting cost.

## 6. SWOT
| Strengths | Weaknesses |
| --- | --- |
| Readable, dependency-free, tested (finite differences, sanitizers); every choice justified by an ADR and every claim measured | CPU only, double precision: slow for large networks (`docs/performance/matmul.md`); only dense layers today; two authors, no external user yet |
| **Opportunities** | **Threats** |
| Schools want projects that cannot be solved by calling a library; demand for explainable AI teaching | Free alternatives (micrograd, courses); AI assistants that generate such code; the name "my_torch" next to the PyTorch trademark ([risk-analysis.md](../legal/risk-analysis.md), R4) |

## 7. Roadmap
| Step | Content | Indicator |
| --- | --- | --- |
| 1. Validate | Interview the pedagogical teams of 3 to 5 schools | Number ready to test a kit |
| 2. License and name | Choose the license (new ADR superseding ADR-0025), rename the product | Done or not |
| 3. Course kit v1 | Exercises cut from the existing layers, optimizers and tests | One school pilot |
| 4. Extend | Bonus layers already planned (convolution #29, normalisation #30, residual #31) | Number of exercises |
| 5. Scale | Workshops, second school | Revenue covers the maintenance time |

## 8. Team
The two members of the pair, main author of each part from `git log`:
- **Ridwane777**: matrix and parallel product, dense layers, activations, dropout, training loop,
  metrics, network file and configuration, analyzer command line and FEN reading, generator, chess
  oracle and position generator (bonus), training pipeline and final network.
- **Meric-BG**: losses, optimizers and schedulers, input encodings, symmetry augmentation, dataset
  split, and the benchmarks of encodings, class imbalance and initialisation.
