# Architecture Decision Records

One file per decision, based on [TEMPLATE.md](TEMPLATE.md). An accepted ADR is never rewritten: if it changes, a new ADR supersedes it.

| ADR | Title | Status | Date |
| --- | --- | --- | --- |
| [0001](0001-python-numpy-language.md) | Python 3 with NumPy as the project language | Abandoned, superseded by ADR-0002 | 2026-10-21 |
| [0002](0002-cpp-language.md) | C++20 as the project language | Accepted | 2026-10-21 |
| [0003](0003-in-house-linear-algebra.md) | In-house linear algebra, in double, stored row by row | Accepted | 2026-10-21 |
| [0004](0004-public-api.md) | Public API of the my_torch library | Accepted | 2026-10-21 |
| [0005](0005-fen-parsing.md) | Reading FEN: format validation and board representation | Proposed | 2026-10-23 |
| [0006](0006-oracle-without-castling.md) | Rule-based oracle: move generation by copy and without castling | Proposed | 2026-10-23 |
| [0007](0007-softmax-cross-entropy-coupling.md) | Cross-entropy coupled to Softmax, Softmax gradient skipped in backpropagation | Proposed | 2026-10-24 |
| [0008](0008-optimizers-and-schedulers.md) | Optimizers keep their own state; schedulers are stateless functions of the epoch | Proposed | 2026-10-24 |
