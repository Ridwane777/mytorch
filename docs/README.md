# Project file — document map

Entry point to follow the work **without reading the code**. Writing rules:
[AGENTS.md](../AGENTS.md). Keep this table up to date with each document added.

Status as of 2026-10-24:

| Area | Document | Status |
| --- | --- | --- |
| Decisions | [adr/](adr/README.md): ADRs and index, [template](adr/TEMPLATE.md) | 16 ADRs: 0001 (Python) abandoned, 0002 to 0015 accepted, 0019 (symmetry augmentation) proposed |
| Data | [reading-a-fen.md](reading-a-fen.md): useful chess rules, reading a dataset line field by field, label distribution | Written |
| Data | [verification-dataset.md](verification-dataset.md): dataset labels checked by the oracle (100% agreement) | Written |
| Usage | [usage.md](usage.md): generate, train and predict from the command line; [configuration.md](configuration.md): every key of a configuration file | Written |
| Architecture | [architecture.md](architecture.md): the four parts and their dependency rule, module map, main classes, generate/train/predict flows with diagrams, errors, tests | Written |
| Performance | [performance/](performance/README.md): [optimizers.md](performance/optimizers.md) (optimizers and schedulers), [overfitting.md](performance/overfitting.md) (L1, L2, dropout, early stopping), [serialization.md](performance/serialization.md) (network file size and load time), [stalemate.md](performance/stalemate.md) (stalemate bonus) | Three measurements written on test problems; the chess network measurements come with #17 and #18 |
| Legal | [legal/](legal/README.md): licenses, data, risks | To write |
| Dependencies | [dependencies.md](dependencies.md): compilers, test and plotting tools, with tested versions | Written; autograder versions still to check |
| Costs | `business/costs.md`: measured and estimated | To write |
| Commercial | [business/](business/README.md): business plan, SWOT, roadmap | To write |
| Retrospective | [retrospective/](retrospective/README.md) | To fill in during the project |
