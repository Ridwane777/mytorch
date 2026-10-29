# Project file — document map

Entry point to follow the work **without reading the code**. Writing rules:
[AGENTS.md](../AGENTS.md). Keep this table up to date with each document added.

Status as of 2026-10-26:

| Area | Document | Status |
| --- | --- | --- |
| Decisions | [adr/](adr/README.md): ADRs and index, [template](adr/TEMPLATE.md) | 33 ADRs: 0001 (Python) and 0030 (search candidate) abandoned, 0002 to 0028 accepted, 0029 and 0031 to 0033 proposed |
| Data | [reading-a-fen.md](reading-a-fen.md): useful chess rules, reading a dataset line field by field, label distribution | Written |
| Data | [verification-dataset.md](verification-dataset.md): dataset labels checked by the oracle (100% agreement) | Written |
| Usage | [usage.md](usage.md): generate, train and predict from the command line, reproduce the final network; [configuration.md](configuration.md): every key of a configuration file; [library.md](library.md): using `my_torch` in another project | Written |
| Usage | [gui.md](gui.md): bonus chess GUI, a playable board analysed move by move by the network and the oracle (ADR-0029) | Written |
| Architecture | [architecture.md](architecture.md): the four parts and their dependency rule, module map, main classes, generate/train/predict flows with diagrams, errors, tests | Written |
| Performance | [performance/](performance/README.md): 14 numbered documents in reading order: data (01 dataset to 05 augmentation), training (06 initialisation to 09 trials), [10-final-network.md](performance/10-final-network.md), code performance (11 training memory, 12 matrix product, 13 network file), bonus (14 stalemate) | Measured on test problems and on the chess dataset; the pushed network scores 0.9615 accuracy and 0.9478 macro F1 on the test split |
| Legal | [legal/](legal/README.md): [risk-analysis.md](legal/risk-analysis.md), [licenses.md](legal/licenses.md), [privacy-policy.md](legal/privacy-policy.md), [terms-of-use.md](legal/terms-of-use.md), [legal-notice.md](legal/legal-notice.md) | Written: student analysis, not legal advice; the dataset is never published |
| Dependencies | [dependencies.md](dependencies.md): compilers, test and plotting tools, tested versions, services and secrets (none) | Written; autograder versions still to check |
| Costs | [business/costs.md](business/costs.md): measured and estimated, kept apart | Written: compute measured, energy estimated with its method, developer time not tracked |
| Commercial | [business/](business/README.md): [business-plan.md](business/business-plan.md), a teaching library: problem, market, competition, model, SWOT, roadmap | Written; market not sized (no sourced figure) |
| Retrospective | [retrospective/](retrospective/README.md): [retrospective.md](retrospective/retrospective.md), technical, by component: where the time goes (profile), linear algebra, training loop, encoding, architecture, data and evaluation, file and pipeline; what we keep, next actions | Written on 2026-10-27, to update until the defense |
| Defense | [defense/](defense/README.md): [demo.md](defense/demo.md) with `scripts/demo.sh`, [questions.md](defense/questions.md) linked to the ADRs, [glossary.md](defense/glossary.md) | Written; used for the defense (#23 closed) |
