# Project file — document map

Entry point to follow the work **without reading the code**. Writing rules:
[AGENTS.md](../AGENTS.md). Keep this table up to date with each document added.

Status as of 2026-10-23:

| Area | Document | Status |
| --- | --- | --- |
| Decisions | [adr/](adr/README.md): ADRs and index, [template](adr/TEMPLATE.md) | 8 ADRs: 0001 (Python, abandoned), 0002 (C++20, accepted), 0003 (linear algebra) and 0004 (API) accepted, 0005 (reading FENs), 0006 (oracle), 0007 (cross-entropy coupled with Softmax) and 0008 (optimizers and schedulers) proposed |
| Data | [reading-a-fen.md](reading-a-fen.md): useful chess rules, reading a dataset line field by field, label distribution | Written |
| Data | [verification-dataset.md](verification-dataset.md): dataset labels checked by the oracle (100% agreement) | Written |
| Architecture | `architecture.md`: `my_torch` library, analyzer, training and prediction flow | To write |
| Performance | [performance/](performance/README.md): benchmarks, learning curves, scripts; [optimizers.md](performance/optimizers.md): optimizers and schedulers, measured over 5 seeds | Optimizers written, still to write for the real dataset |
| Legal | [legal/](legal/README.md): licenses, data, risks | To write |
| Dependencies | [dependencies.md](dependencies.md) | Skeleton |
| Costs | `business/costs.md`: measured and estimated | To write |
| Commercial | [business/](business/README.md): business plan, SWOT, roadmap | To write |
| Retrospective | [retrospective/](retrospective/README.md) | To fill in during the project |
