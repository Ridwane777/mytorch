# ADR-0048: MIT license for the code

- **Status**: Accepted (2026-11-17)
- **Date**: 2026-11-17
- **Deciders**: Ridwane777, Meric Gbemetonou
- **Supersedes**: [ADR-0025](0025-project-license.md)

## Context
ADR-0025 kept the code unlicensed while the repository was private and asked for a new ADR before any
public release. The repository is now public. With no license, readers can see the code but may not
reuse any of it, which contradicts the teaching purpose of the business plan. The code is a work of
collaboration of the two authors (CPI art. L113-3): both agreed to the license below. The dataset is
not ours: Epitech provided it and authorized its redistribution, nothing more.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| **MIT (chosen)** | Short and well known; lets learners and teachers reuse the library; same license as the authors' other public projects | Granted for ever on the published versions; anyone may reuse the code, including in closed products, with no obligation to share changes |
| Apache 2.0 | Permissive, with an explicit patent grant | Longer; the patent clause has no use for this project |
| GPL v3 | Derived works must stay open | Heavier for a teaching library; discourages reuse in coursework or companies |
| Keep no license | Nothing granted | Public but unusable: blocks the teaching use, looks closed |

## Decision
The code and documentation are under the MIT license, copyright Ridwane777 and Meric Gbemetonou
(`LICENSE` at the root). `dataset.zip` and the third-party files of the GUI bonus are excluded and keep
their own terms, stated in `LICENSE` and `docs/legal/licenses.md`.

## Consequences
- ✅ Anyone may read, run, modify and reuse the library, which makes the teaching use of the business plan possible.
- ✅ Same license as the authors' other public repositories: one rule to explain.
- ❌ The grant cannot be withdrawn for the versions already published, even if the license changes later.
- ❌ Reuse in closed products is allowed with no return: the authors only get the attribution.
- ❌ The repository mixes licenses (MIT code, Epitech dataset, BSD/MIT/CC BY-SA GUI files): readers must check `LICENSE` before reusing the data or the GUI assets.
- Follow-up: any outside contribution is accepted under the same license (stated in `CONTRIBUTING.md`).
