# Project file — document map

Entry point to follow the work **without reading the code**. Writing rules:
[AGENTS.md](../AGENTS.md). Keep this table up to date with each document added.

Status as of 2026-10-26:

| Area | Document | Status |
| --- | --- | --- |
| Decisions | [adr/](adr/README.md): ADRs and index, [template](adr/TEMPLATE.md) | 25 ADRs: 0001 (Python) abandoned, 0002 to 0023 accepted, 0024 and 0025 proposed |
| Data | [reading-a-fen.md](reading-a-fen.md): useful chess rules, reading a dataset line field by field, label distribution | Written |
| Data | [verification-dataset.md](verification-dataset.md): dataset labels checked by the oracle (100% agreement) | Written |
| Usage | [usage.md](usage.md): generate, train and predict from the command line, reproduce the final network; [configuration.md](configuration.md): every key of a configuration file; [library.md](library.md): using `my_torch` in another project | Written |
| Architecture | [architecture.md](architecture.md): the four parts and their dependency rule, module map, main classes, generate/train/predict flows with diagrams, errors, tests | Written |
| Performance | [performance/](performance/README.md): [optimizers.md](performance/optimizers.md) (optimizers and schedulers), [overfitting.md](performance/overfitting.md) (L1, L2, dropout, early stopping), [serialization.md](performance/serialization.md) (network file size and load time), [stalemate.md](performance/stalemate.md) (stalemate bonus), [initialisation.md](performance/initialisation.md) (weight initialisation), [encodings.md](performance/encodings.md) (input encodings), [dataset.md](performance/dataset.md) (dataset statistics and split), [imbalance.md](performance/imbalance.md) (class imbalance), [augmentation.md](performance/augmentation.md) (symmetry augmentation), [training-memory.md](performance/training-memory.md) (training memory), [eda.md](performance/eda.md) (exploratory data analysis), [trials.md](performance/trials.md) (architecture trials), [matmul.md](performance/matmul.md) (parallel matrix product), [final-network.md](performance/final-network.md) (the pushed network) | Measured on test problems and on the chess dataset; the pushed network scores 0.9615 accuracy and 0.9478 macro F1 on the test split |
| Legal | [legal/](legal/README.md): [risk-analysis.md](legal/risk-analysis.md), [licenses.md](legal/licenses.md), [privacy-policy.md](legal/privacy-policy.md), [terms-of-use.md](legal/terms-of-use.md), [legal-notice.md](legal/legal-notice.md) | Written: student analysis, not legal advice; the dataset is never published |
| Dependencies | [dependencies.md](dependencies.md): compilers, test and plotting tools, tested versions, services and secrets (none) | Written; autograder versions still to check |
| Costs | `business/costs.md`: measured and estimated | To write |
| Commercial | [business/](business/README.md): business plan, SWOT, roadmap | To write |
| Retrospective | [retrospective/](retrospective/README.md) | To fill in during the project |
