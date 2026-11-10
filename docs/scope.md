# Scope: required minimum and additions

my_torch is an Epitech school project (3rd year), built by a team of two and published with
authorization. The specification itself is not copied here; the requirements below are the ones
recorded in [AGENTS.md](../AGENTS.md) and in the ADRs that cite them.

## 1. Required by the specification

| Requirement | Status | Evidence |
| --- | --- | --- |
| Neural-network library written from scratch: no NN library, no automatic differentiation | Done | [ADR-0002](adr/0002-cpp-language.md), [ADR-0003](adr/0003-in-house-linear-algebra.md), `scripts/check_comparison_isolation.sh` |
| Supervised learning | Done | [ADR-0009](adr/0009-training-loop.md) |
| `my_torch_analyzer --train [--save] / --predict LOADFILE CHESSFILE`, FEN → `Checkmate` / `Check` / `Nothing` (with the side) | Done | [ADR-0014](adr/0014-analyzer-command-line.md), `src/analyzer/` |
| One prediction per position, in file order; errors on stderr, exit code 84 | Done | [ADR-0005](adr/0005-fen-parsing.md), `tests/analyzer/` |
| `my_torch_generator config_file nb [...]` creating blank networks | Done | [ADR-0013](adr/0013-configuration-format.md), `src/generator/` |
| Pushed pre-trained network `my_torch_network*` | Done: `my_torch_network_final.nn`, 96.2 % on 71,555 unseen positions | [10-final-network.md](performance/10-final-network.md) |
| Avoid overfitting; evaluate with several metrics; faster learning | Done: L1/L2, dropout, early stopping; metrics; optimizers and schedulers | [ADR-0010](adr/0010-overfitting-remedies.md), [ADR-0011](adr/0011-classification-metrics.md), [ADR-0008](adr/0008-optimizers-and-schedulers.md) |
| Reproducible training, every script and configuration kept | Done: seeded runs, `configs/`, `scripts/` | [ADR-0004](adr/0004-public-api.md), [ADR-0013](adr/0013-configuration-format.md) |
| Makefile with `re`, `clean`, `fclean`; bonuses in `bonus/` | Done | `Makefile`, `bonus/Makefile` |

## 2. What we added, and requirements we exceeded

| Addition | Why | Evidence |
| --- | --- | --- |
| Five optimizers (SGD with momentum/Nesterov, Adam, AdamW, RMSProp, Adagrad) and five schedulers, benchmarked over 5 seeds | Choose by measurement, not by habit | [ADR-0008](adr/0008-optimizers-and-schedulers.md), [07-optimizers.md](performance/07-optimizers.md) |
| Four input encodings compared, board-symmetry augmentation, class-imbalance strategies | The encoding decided most of the accuracy | [ADR-0016](adr/0016-input-encoding.md), [ADR-0019](adr/0019-symmetry-augmentation.md), [ADR-0018](adr/0018-class-imbalance.md) |
| Weight initialisation schemes (LeCun, Xavier, He, orthogonal) | Stable training of deeper networks | [ADR-0020](adr/0020-weight-initialisation.md), [06-initialisation.md](performance/06-initialisation.md) |
| Blocked, multi-threaded matrix product | Training time on the full dataset | [ADR-0024](adr/0024-parallel-blocked-matmul.md), [12-matmul.md](performance/12-matmul.md) |
| BatchNorm and LayerNorm | Deep networks | [ADR-0033](adr/0033-normalization.md), [17-normalization.md](performance/17-normalization.md) |
| Convolutional layers and a CNN: 98.07 % | Beat the dense network on the board planes | [ADR-0037](adr/0037-convolutional-layers.md), [20-cnn.md](performance/20-cnn.md) |
| Residual and concat blocks | Deeper CNNs | [ADR-0046](adr/0046-residual-blocks.md), [28-residual.md](performance/28-residual.md) |
| Multi-head self-attention, mini transformer | Compare attention with convolution | [ADR-0047](adr/0047-attention-layers.md), [29-attention.md](performance/29-attention.md) |
| Hyperparameter search (grid and random) | Systematic tuning | [ADR-0028](adr/0028-hyperparameter-search-api.md), [15-hyperparameters.md](performance/15-hyperparameters.md) |
| Stratified k-fold cross-validation | How much a score moves between splits | [ADR-0039](adr/0039-cross-validation.md), [21-cross-validation.md](performance/21-cross-validation.md) |
| MC dropout uncertainty | Flag the doubtful predictions | [ADR-0034](adr/0034-mc-dropout.md), [18-uncertainty.md](performance/18-uncertainty.md) |
| Ensembles: stack 98.26 % | Combine the dense network and the CNN | [ADR-0045](adr/0045-model-ensembles.md), [27-ensembles.md](performance/27-ensembles.md) |
| Gradient clipping, checkpoints with identical resumption | Long trainings that can stop and resume | [ADR-0031](adr/0031-training-stability.md), [ADR-0032](adr/0032-checkpoints.md) |
| Binary network file with checksum (text dump kept as a tool) | Size ÷ 2.55, integrity check | [ADR-0042](adr/0042-binary-network-file-format.md), [24-binary-format.md](performance/24-binary-format.md) |
| Rule-based chess oracle | Verify the dataset and the predictions | [ADR-0006](adr/0006-oracle-without-castling.md), `bonus/oracle/` |
| Stalemate positions and a six-class network | Bonus class | [ADR-0015](adr/0015-stalemate-positions.md), [14-stalemate.md](performance/14-stalemate.md) |
| Chess GUI with saliency maps and network inspection | See what the network looks at | [ADR-0029](adr/0029-chess-gui.md), [ADR-0035](adr/0035-network-inspection.md), [gui.md](gui.md) |
| Python bindings (pybind11) | Use the library from Python | [ADR-0044](adr/0044-python-bindings.md), [25-python-bindings.md](performance/25-python-bindings.md) |
| Comparison with PyTorch (same weights and batches) and classical models, isolated in `bonus/comparaison/` | Check our gradients and place our results | [ADR-0036](adr/0036-comparison-framework.md), [ADR-0040](adr/0040-pytorch-reference.md), [ADR-0043](adr/0043-classical-baselines.md), [26-comparison.md](performance/26-comparison.md) |
| Unattended training on a temporary AWS EC2 instance | Full-dataset trainings off the laptop | [ADR-0038](adr/0038-unattended-aws-training.md), `scripts/aws_train.sh` |
| Decision records, performance reports, legal, business and retrospective documents | Follow the work without reading the code | [docs/README.md](README.md) |
