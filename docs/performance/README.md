# Performance and benchmarks

Each measurement states: method, machine, dataset, configuration (`configs/`), reproduction
script (`scripts/`), results **with graphs** (`data/`), thresholds and limits.

Numbered in reading order: the data, then the training choices, then the final network, then the
performance of the code itself.

| # | Document | Question | Issue |
| --- | --- | --- | --- |
| **Data** | | | |
| 01 | [01-dataset.md](01-dataset.md) | What is in the dataset, and how is it split into train, validation and test without sharing a position? | #12 |
| 02 | [02-eda.md](02-eda.md) | Exploratory analysis: checking pieces, king squares, material, FEN fields, shortcuts | #12 (complement) |
| 03 | [03-encodings.md](03-encodings.md) | Which input encoding of a FEN position learns best? | #14 |
| 04 | [04-imbalance.md](04-imbalance.md) | Class imbalance: none, loss weighting, oversampling or undersampling? | #16 |
| 05 | [05-augmentation.md](05-augmentation.md) | Does board-symmetry augmentation help? | #38 |
| **Training** | | | |
| 06 | [06-initialisation.md](06-initialisation.md) | LeCun, Xavier, He and orthogonal initialisation against the default | #6 |
| 07 | [07-optimizers.md](07-optimizers.md) | Which optimizer and learning-rate scheduler? | #5 |
| 08 | [08-overfitting.md](08-overfitting.md) | L1, L2, elastic net, dropout and early stopping against overfitting | #7 |
| 09 | [09-trials.md](09-trials.md) | Architecture and initialisation of the final network, five seeds per finalist | #17 |
| 15 | [15-hyperparameters.md](15-hyperparameters.md) | Learning rate, batch size, hidden layers and dropout: grid search against random search | #10 |
| **Final network** | | | |
| 10 | [10-final-network.md](10-final-network.md) | How good is the pushed network on the test split, with and without augmentation? | #17 |
| 20 | [20-cnn.md](20-cnn.md) | Does a convolutional network on the board planes beat the dense network, and what do the two architectures look like? | #29 |
| 21 | [21-cross-validation.md](21-cross-validation.md) | How much does a score move from one split to another: stratified 5-fold cross-validation of the dense network and the CNN, mean ± standard deviation and 95 % interval | #39 |
| **Code performance** | | | |
| 11 | [11-training-memory.md](11-training-memory.md) | Training memory: whole file encoded against batch by batch | #17 |
| 12 | [12-matmul.md](12-matmul.md) | Blocked, threaded matrix product: speed and identical results | #25, #33 |
| 13 | [13-serialization.md](13-serialization.md) | Network file size and save/load time | #8 |
| 16 | [16-benchmarks.md](16-benchmarks.md) | Learning curves of the full trainings, accuracy per category of position, training and prediction times, cost of the checks and checkpoints | #18 |
| 17 | [17-normalization.md](17-normalization.md) | BatchNorm and LayerNorm against no normalization, on the final architecture and a deep network | #30 |
| 18 | [18-uncertainty.md](18-uncertainty.md) | MC dropout: are the errors the uncertain predictions, and does it beat the plain confidence? | #26 |
| 19 | [19-inspection.md](19-inspection.md) | What the network learned: summary, weight and gradient histograms, which squares decide its answer | #36 |
| **Comparison** | | | |
| 22 | [22-pytorch-reference.md](22-pytorch-reference.md) | Same architectures in PyTorch (`bonus/comparaison/`): gradients against autograd, our weights and batches replayed, time per epoch, memory and accuracy after an equal budget | #42 |
| 23 | [23-baselines.md](23-baselines.md) | Classical baselines on the same split and encoding: logistic regression 65 %, random forest 77 %, scikit-learn MLP 91.7 %, LightGBM 96.75 %, against our dense 96.15 % and CNN 98.07 % | #43 |
| 24 | [24-binary-format.md](24-binary-format.md) | Binary network file (ADR-0042) against text: size ÷ 2.55, save and load about 2 times slower, identical predictions on the test split | defense |
| 25 | [25-python-bindings.md](25-python-bindings.md) | Python module (ADR-0044) against the C++ library: probabilities identical bit for bit and every label the analyzer's on the test split, prediction 12 % slower because of the NumPy copies | defense |
| 26 | [26-comparison.md](26-comparison.md) | Final comparison (#44): rule-based oracle 100 %, our CNN 98.07 % first among learned models, LightGBM 96.75 %, our dense 96.15 %; same results as PyTorch, 3 to 6 times slower to train, 2 to 3 times less memory; lines of code | #44 |
| 27 | [27-ensembles.md](27-ensembles.md) | Ensembles of the dense network and the CNN: stack 98.26 % and mean 1:2 98.23 % against the CNN alone 98.07 % (significant); mean 1:1 and vote no gain; cost of both members | #40 |
| 28 | [28-residual.md](28-residual.md) | Residual blocks (10 % of the data, 10 epochs): 6 convolutions beat 2 (95.76 % against 94.17 %), but the 2 residual blocks do not beat the same depth without shortcuts (95.43 %, McNemar p = 1.1e-5) | #31 |
| 29 | [29-attention.md](29-attention.md) | Mini transformer (10 % of the data, 10 epochs): 91.67 % (1 block) and 91.80 % (2 blocks) against 94.17 % for the CNN (McNemar p < 1e-100), 3 to 6 times slower per epoch, 4 to 7 times the memory; the dense network reaches 76.97 % | #32 |
| **Bonus** | | | |
| 14 | [14-stalemate.md](14-stalemate.md) | Stalemate: generated positions and a six-class network | #24 |

The benchmarks of #18 are gathered in [16-benchmarks.md](16-benchmarks.md).
