# Architecture Decision Records

One file per decision, based on [TEMPLATE.md](TEMPLATE.md). An accepted ADR is never rewritten: if it changes, a new ADR supersedes it.

| ADR | Title | Status | Date |
| --- | --- | --- | --- |
| [0001](0001-python-numpy-language.md) | Python 3 with NumPy as the project language | Abandoned, superseded by ADR-0002 | 2026-10-21 |
| [0002](0002-cpp-language.md) | C++20 as the project language | Accepted | 2026-10-21 |
| [0003](0003-in-house-linear-algebra.md) | In-house linear algebra, in double, stored row by row | Accepted | 2026-10-21 |
| [0004](0004-public-api.md) | Public API of the my_torch library | Accepted | 2026-10-21 |
| [0005](0005-fen-parsing.md) | Reading FEN: format validation and board representation | Accepted | 2026-10-23 |
| [0006](0006-oracle-without-castling.md) | Rule-based oracle: move generation by copy and without castling | Accepted | 2026-10-23 |
| [0007](0007-softmax-cross-entropy-coupling.md) | Cross-entropy coupled to Softmax, Softmax gradient skipped in backpropagation | Accepted | 2026-10-24 |
| [0008](0008-optimizers-and-schedulers.md) | Optimizers keep their own state; schedulers are stateless functions of the epoch | Accepted | 2026-10-24 |
| [0009](0009-training-loop.md) | Training loop, scheduler parameter and portable shuffling | Accepted | 2026-10-24 |
| [0010](0010-overfitting-remedies.md) | Overfitting remedies: per-layer L1/L2, dropout layer and early stopping in fit | Accepted | 2026-10-24 |
| [0011](0011-classification-metrics.md) | Classification metrics in the library, conventions and export formats | Accepted | 2026-10-24 |
| [0012](0012-network-file-format.md) | Network file format: versioned self-describing text with a checksum | Accepted | 2026-10-24 |
| [0013](0013-configuration-format.md) | Generator configuration in the network file syntax, seeded and checked before writing | Accepted | 2026-10-24 |
| [0014](0014-analyzer-command-line.md) | Analyzer command line, encoding table, label matching and validation hold-out | Accepted | 2026-10-24 |
| [0015](0015-stalemate-positions.md) | Stalemate bonus: generated sparse positions with generated negatives | Accepted | 2026-10-24 |
| [0016](0016-input-encoding.md) | Input encoding of a position | Accepted | 2026-10-26 |
| [0017](0017-dataset-split.md) | Dataset split: dedupe-aware stratified split by position key | Accepted | 2026-10-25 |
| [0020](0020-weight-initialisation.md) | Weight initialisation chosen per layer in the configuration, original default kept | Accepted | 2026-10-25 |
