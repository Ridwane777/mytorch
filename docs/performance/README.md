# Performance and benchmarks

Each measurement states: method, machine, dataset, configuration (`configs/`), reproduction
script (`scripts/`), results **with graphs** (`data/`), thresholds and limits.

To write once the chess network is trained (#17, #18): per-class metrics and confusion matrix (tools
from #11), learning curves on the real dataset, comparison of configurations, training and
prediction time.

Done: [optimizers.md](optimizers.md) (optimizers and learning-rate schedulers, issue #5),
[overfitting.md](overfitting.md) (L1, L2, elastic net, dropout and early stopping against overfitting, issue #7),
[serialization.md](serialization.md) (network file size and save/load time, issue #8),
[stalemate.md](stalemate.md) (stalemate bonus: generated positions and a six-class network, issue #24),
[initialisation.md](initialisation.md) (LeCun, Xavier, He and orthogonal weight initialisation against the default, issue #6),
[encodings.md](encodings.md) (input encodings of a FEN position, issue #14),
[dataset.md](dataset.md) (dataset statistics and dedupe-aware train/validation/test split, issue #12),
[imbalance.md](imbalance.md) (class imbalance: none, loss weighting, oversampling, undersampling,
issue #16),
[augmentation.md](augmentation.md) (training with and without board-symmetry augmentation, issue #38),
[training-memory.md](training-memory.md) (training memory, whole file encoded against batch by batch, issue #17),
[eda.md](eda.md) (exploratory analysis: checking pieces, king squares, material, FEN fields, shortcuts),
[trials.md](trials.md) (architecture and initialisation of the final network, five seeds per finalist, issue #17).
