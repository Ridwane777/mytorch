# Legal risk analysis

> Student analysis, **not legal advice**. Situation on 2026-10-27: the repository is **private**
> in the `Ridwane777` GitHub organisation, and the programs run only on the user's machine.

## 1. Map of the risks

| # | Risk | Reference text | Level, school project | Level, if it became a product | Measure |
| --- | --- | --- | --- | --- | --- |
| R1 | Redistributing the dataset given by the school | French intellectual property code (CPI), art. L341-1 (database producer's right, Directive 96/9/EC); art. L122-4 (reproduction without consent) | Low | **High** | `dataset/` and `dataset.zip` are in `.gitignore` and never pushed; only statistics and our own example positions are published |
| R2 | Ownership and license of the code | CPI art. L113-2 and L113-3 (work of collaboration, two authors); the school rules on student projects | Medium | **High** | No license granted, all rights reserved (ADR-0025); the ownership under the the school rules is to check before any release |
| R3 | Breaking the subject's ban on neural-network libraries (academic integrity) | the school subject and rules | **High** if broken | — | All computation is ours (ADR-0002, ADR-0003); no NN library in the build (`docs/dependencies.md`); PyTorch and scikit-learn appear only in the bonus comparison plan (#42, #43), outside the graded executables |
| R4 | The name "my_torch" next to the PyTorch trademark | Trademark law (CPI book VII); PyTorch is a trademark of The Linux Foundation | Low (name imposed by the subject, not used commercially) | **High** | Rename before any commercial use |
| R5 | A wrong prediction used to make a decision | Civil code, art. 1240 (liability for fault) | Low | Medium | Accuracy published honestly (96.2 % on unseen positions, [final-network.md](../performance/final-network.md)); terms of use: provided as is, no warranty |
| R6 | Personal data | GDPR (Regulation (EU) 2016/679) | None | Low | The programs read local files, have no network access and log nothing ([privacy-policy.md](privacy-policy.md)) |
| R7 | Rules on artificial intelligence | AI Act (Regulation (EU) 2024/1689) | None | Low | A chess position classifier is not among the high-risk uses (Annex III); no obligation beyond the transparency already given by the documentation |
| R8 | Licenses of the dependencies | GPL v3 with the GCC Runtime Library Exception; MIT; PSF; matplotlib license | Low | Low | The executables only link the C++ standard library, which the runtime exception allows under any license ([licenses.md](licenses.md)) |

## 2. The main risks in detail

### R1: the dataset
- the school gave the dataset as `dataset.zip` (723,019 positions, `docs/performance/dataset.md`). It
  comes with no license file and no statement of its origin.
- A chess position is a fact, but a **collection** of positions gathered with a substantial
  investment can be protected by the database producer's right (CPI art. L341-1). Without a license,
  we assume we may use it for the project and nothing else.
- What we publish: counts, statistics and graphs measured on it, and positions we wrote ourselves or
  produced with our own tools (`bonus/positions`). What we never publish: the files themselves.

### R2: ownership of the code
- The code was written by the two members of the pair (`git log`): it is a work of collaboration,
  owned jointly (CPI art. L113-3), and every release needs the agreement of both.
- With no license file, nobody else gets any right to copy or reuse it: this is the default of
  copyright law, and the choice of ADR-0025.
- To check before any release: what the the school rules say about the rights on student projects.

### R4: the name
"my_torch" is imposed by the subject and refers to PyTorch on purpose. Fine inside a school
project; a product would need its own name to avoid any confusion with a registered trademark.

## 3. Summary
For the school project the only real risk is R3, and it is under control: it is checked by the
build itself (no NN library is linked). The other risks only matter if the project leaves the school:
then R1, R2 and R4 must be settled first (data, ownership, name).
