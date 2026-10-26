# Performance and benchmarks

Each measurement states: method, machine, dataset, configuration (`configs/`), reproduction
script (`scripts/`), results **with graphs** (`data/`), thresholds and limits.

To write once the chess network is trained (#17, #18): per-class metrics and confusion matrix (tools
from #11), learning curves on the real dataset, comparison of configurations, training and
prediction time.

Done: [optimizers.md](optimizers.md) (optimizers and learning-rate schedulers, issue #5),
[overfitting.md](overfitting.md) (L1, L2, elastic net, dropout and early stopping against overfitting, issue #7),
[serialization.md](serialization.md) (network file size and save/load time, issue #8),
[stalemate.md](stalemate.md) (stalemate bonus: generated positions and a six-class network, issue #24).
[encodings.md](encodings.md) (input encodings of a FEN position, issue #14).
