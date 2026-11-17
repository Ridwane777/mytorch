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
| [0012](0012-network-file-format.md) | Network file format: versioned self-describing text with a checksum | Superseded by ADR-0042 | 2026-10-24 |
| [0013](0013-configuration-format.md) | Generator configuration in the network file syntax, seeded and checked before writing | Accepted | 2026-10-24 |
| [0014](0014-analyzer-command-line.md) | Analyzer command line, encoding table, label matching and validation hold-out | Accepted | 2026-10-24 |
| [0015](0015-stalemate-positions.md) | Stalemate bonus: generated sparse positions with generated negatives | Accepted | 2026-10-24 |
| [0016](0016-input-encoding.md) | Input encoding of a position | Accepted | 2026-10-26 |
| [0017](0017-dataset-split.md) | Dataset split: dedupe-aware stratified split by position key | Accepted | 2026-10-25 |
| [0018](0018-class-imbalance.md) | Class imbalance: none, loss weighting, oversampling and undersampling compared on per-class F1 | Accepted | 2026-10-25 |
| [0019](0019-symmetry-augmentation.md) | Symmetry augmentation: offline tool and two label-preserving board symmetries | Accepted | 2026-10-25 |
| [0020](0020-weight-initialisation.md) | Weight initialisation chosen per layer in the configuration, original default kept | Accepted | 2026-10-25 |
| [0021](0021-training-data-source.md) | Training data as a batch source, positions encoded on demand | Accepted | 2026-10-26 |
| [0022](0022-validation-tail-and-training-pipeline.md) | Untouched validation set at the end of the training file, one-command training pipeline | Accepted | 2026-10-26 |
| [0023](0023-final-architecture.md) | Final network architecture, 901 → 256 → 128 → 5 with the default initialisation | Accepted | 2026-10-27 |
| [0024](0024-parallel-blocked-matmul.md) | Parallel and blocked matrix product, bit-identical results | Accepted | 2026-10-27 |
| [0025](0025-project-license.md) | No license, all rights reserved | Superseded by 0048 | 2026-10-27 |
| [0026](0026-scores-from-the-library.md) | Published scores computed by the library | Accepted | 2026-10-27 |
| [0027](0027-generic-data-splits.md) | Generic data splits in the library | Accepted | 2026-10-27 |
| [0028](0028-hyperparameter-search-api.md) | Hyperparameter search in the library: search space, objective, grid and random strategies, resumable log | Accepted | 2026-10-28 |
| [0029](0029-chess-gui.md) | Chess GUI: a page with chess.js and chessboard.js, served by a local C++ server | Accepted | 2026-10-28 |
| [0030](0030-hyperparameter-search-result.md) | Candidate configuration from the hyperparameter search: lr 0.001, batches of 32, 512 → 256, dropout 0.2; below the final network at 100 % on the test split | Abandoned | 2026-10-28 |
| [0031](0031-training-stability.md) | Training stability: gradient clipping by value and by norm, off by default; stop on the first non-finite value, naming the layer | Accepted | 2026-10-28 |
| [0032](0032-checkpoints.md) | Checkpoints: the whole training state saved after every epoch, resumed when the file exists, identical to an uninterrupted run | Accepted | 2026-10-28 |
| [0033](0033-normalization.md) | Normalization: BatchNorm and LayerNorm layers between a Dense and its activation, running averages in prediction; not added to the final network | Accepted | 2026-10-29 |
| [0034](0034-mc-dropout.md) | Uncertainty by Monte Carlo dropout: N passes with only the dropout layers active; no better than the plain top probability on the chess network | Accepted | 2026-10-29 |
| [0035](0035-network-inspection.md) | Network inspection: model summary, epoch callback, input gradient; saliency per square by occlusion, shown in the GUI | Accepted | 2026-10-29 |
| [0036](0036-comparison-framework.md) | Comparison framework with existing networks: PyTorch, scikit-learn, XGBoost and LightGBM only under `bonus/comparaison/` in a separate virtualenv, one common protocol, a guard script; accepted by the pair as a bonus | Accepted | 2026-10-29 |
| [0037](0037-convolutional-layers.md) | Convolutional layers (Conv2D by im2col, pooling, board_planes) and a CNN for the chessboard; `Layer::outputWidth` added to the contract; the CNN scores 0.9684 against 0.8577 for the dense network on 10 % of the data, the pushed network stays dense | Accepted | 2026-10-29 |
| [0038](0038-unattended-aws-training.md) | Unattended training of the CNN on one on-demand AWS instance (`scripts/aws_train.sh`, S3, scoped role, self-termination), `--progress-every` snapshots and the refusal to launch when the epochs are below the progress interval | Accepted | 2026-11-02 |
| [0039](0039-cross-validation.md) | Cross-validation helper: `crossValidate` calls a trainer per stratified fold and reports each metric as mean ± sample standard deviation; fixed epochs, no early stopping and no augmentation inside the folds | Accepted | 2026-11-04 |
| [0040](0040-pytorch-reference.md) | PyTorch reference in `bonus/comparaison/`: architecture read from the configuration, our weights and batches replayed in PyTorch, gradients checked against autograd to 1e-5, float64 for the equivalence and float32 for the realistic cost; accepted by the pair with ADR-0036 | Accepted | 2026-11-04 |
| [0041](0041-generic-planes-layer.md) | Generic `to_planes` layer (height and width in its line) instead of the chess-specific `board_planes`, so that the library knows nothing about chess; the old name is still read | Accepted | 2026-11-04 |
| [0042](0042-binary-network-file-format.md) | Binary network file format: signature, version, typed binary tokens (keyword codes, LEB128 integers, binary64 reals) and a checksum; text still read, `saveText` and `my_torch_nn_dump` for inspection | Accepted | 2026-11-04 |
| [0043](0043-classical-baselines.md) | Classical baselines (logistic regression, random forest, scikit-learn MLP, LightGBM): a small grid each, chosen on validation, test read once, no augmentation | Accepted | 2026-11-04 |
| [0044](0044-python-bindings.md) | Python bindings with pybind11 in `bonus/python/`: `Network` (load, from_config, predict, fit, save) with NumPy arrays, library errors as `my_torch.Error`; the library is not changed | Accepted | 2026-11-04 |
| [0045](0045-model-ensembles.md) | Ensembles of networks: `Ensemble` in the library (weighted mean, vote, stacked combiner), one binary file holding every member, read by `--predict` with the same command line | Accepted | 2026-11-05 |
| [0046](0046-residual-blocks.md) | Residual connections and concatenation as container layers (`residual`, `concat`) inside the sequential network; identity shortcut only; old network files unchanged | Accepted | 2026-11-05 |
| [0047](0047-attention-layers.md) | Attention layers for a mini transformer (`to_tokens`, `token_dense`, `positional`, `self_attention`, `token_norm`); pre-norm blocks built from `residual`, learned positions, no encoding change; old network files unchanged | Accepted | 2026-11-08 |
| [0048](0048-mit-license.md) | MIT license for the code, copyright of both authors; `dataset.zip` and third-party GUI files excluded | Accepted | 2026-11-17 |
