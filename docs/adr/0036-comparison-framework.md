# ADR-0036: Comparison framework with existing networks

- **Status**: Proposed (until the teaching staff approve; see Follow-up)
- **Date**: 2026-10-29
- **Deciders**: Meric-Gbemetonou (issue #41)

## Context
Issue #41 asks to compare our network with existing ones, to know whether 0.9615 accuracy and
0.9478 macro F1 on the test split (`docs/performance/10-final-network.md`) are good or only
look good. The subject forbids neural-network libraries (PyTorch, TensorFlow, Keras,
scikit-learn, JAX…) and automatic differentiation in the project (`AGENTS.md`). A comparison
needs exactly those libraries to run the other side. The question is therefore not only which
tools to use, but where they may live without touching the deliverable: `my_torch_analyzer`,
`my_torch_generator`, the library, the Makefile and the pushed network.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| Published results only | No library in the repository; no risk with the subject | No comparison on our data and our split: numbers from other datasets prove nothing |
| **Separate `comparison/` + its own virtualenv (chosen)** | Same data, split, encoding and metrics on both sides; nothing in the deliverable depends on it; one script can check the isolation | A second toolchain (Python and its packages); the directory still holds code that uses forbidden libraries |
| Put it in `bonus/` | Bonuses are already a place for extra work | `bonus/` is built by the project and graded as ours; a forbidden library there is much harder to defend |
| Drop the comparison | No risk at all | Nothing tells us whether our network is close to what a library gets; the "decision-making" criteria lose their reference |

Tools, all used only under `comparison/`:

| Tool | Role |
| --- | --- |
| PyTorch | Same-architecture reference: 901 → 256 → 128 → 5 (ADR-0023), same optimizer and schedule as `configs/final_network.conf`. Shows what our own implementation costs or gains against a mature one |
| scikit-learn | Classical baselines (for example logistic regression, random forest) on the same encoded input |
| XGBoost | Gradient-boosted trees baseline |
| LightGBM | Gradient-boosted trees baseline, faster to train than XGBoost on 901 features |

## Decision
The comparison lives in `comparison/`, with a virtualenv `comparison/.venv` (git-ignored) and a
requirements file in `comparison/`. Nothing in `comparison/` is linked into, built by or called
from the Makefile or any executable; the deliverable does not need it to build, run or test.
`scripts/check_comparison_isolation.sh` fails when PyTorch, TensorFlow, Keras, scikit-learn,
XGBoost, LightGBM or JAX is imported outside `comparison/`.

Every model is compared under the same protocol, so that a difference comes from the model and
not from the setup:

1. **Same dataset**: the one given by the school, read as by `scripts/dataset_split.sh`.
2. **Same split**: `SEED=42 sh scripts/dataset_split.sh` (ADR-0017, 0027), which writes
   `train.txt`, `validation.txt` and `test.txt` to `build/dataset_split/`. The test split is read
   once per model, after its choices are made on the validation split.
3. **Same encoding**: `fen-attacks-901` (ADR-0016), the same 901 values per position.
4. **Same architecture** for the neural reference: the layers, loss, optimizer, learning rate,
   batch size, epochs and early stopping of `configs/final_network.conf` (ADR-0023), with the same
   seed. The classical baselines only share items 1 to 3 and 5 to 6; their hyperparameters are
   chosen on the validation split.
5. **Same metrics**: every model writes one label per line, in the order of the test file, and
   `scripts/score.cpp` scores it with the library's metrics (ADR-0011, ADR-0026): accuracy, macro
   F1, precision, recall and F1 per class, confusion matrix. No second implementation of the
   metrics.
6. **Same machine**, recorded in the results (CPU, cores, memory, library versions), with training
   time measured on it. A result without its machine is not reported.

The results go to `docs/performance/` with a graph and the script that produces both, as every
other measurement (AGENTS.md). Measured numbers are kept apart from published ones.

Risk, stated plainly: the subject forbids neural-network libraries in the project. This layout
keeps them out of everything that is built, run, tested and pushed as the project, and the guard
makes that checkable, but it cannot guarantee that the graders accept a directory that imports
them at all. That is a judgment of the teaching staff, not a technical property, so their
approval is an explicit open item below and this ADR stays Proposed until it is given.

## Consequences
- ✅ We learn how our network stands against a mature implementation with the same architecture
  and against classical models, on our data and our split.
- ✅ The deliverable is independent from the comparison: removing `comparison/` changes nothing in
  the build, the tests or the pushed network.
- ✅ The isolation is checked by a command, not by promise:
  `scripts/check_comparison_isolation.sh` (exit 0 when clean, exit 1 with the offending lines on
  stderr otherwise).
- ❌ The staff may refuse any use of these libraries in the repository; in that case
  `comparison/` is not pushed and the comparison falls back to published results only.
- ❌ A second toolchain to install and keep working (Python, a virtualenv, four large packages).
- ❌ Results depend on the library versions and on the machine; they must be re-measured, not
  copied, when either changes.
- ❌ The guard reads tracked files only (`git grep`): a new file must be `git add`ed to be seen.
  It looks at `*.py *.cpp *.hpp *.h *.sh`, `Makefile` and `requirements*.txt`, and does not catch
  an indirect use (a library called through a subprocess or a dynamic import).
- Follow-up:
  - **Open: approval of the teaching staff** before any code using these libraries is pushed;
    the ADR moves to Accepted (or is abandoned) with their answer.
  - Run `scripts/check_comparison_isolation.sh` before each pull request, and in the CI if one
    exists.
  - The implementation of the comparison itself is separate work, outside this ADR.
  - Revisit if the staff set another rule, or if the library versions change the results.
