# How we work on this repository

## The main rule

**Nobody pushes directly to `main`.** Changes get in through a Pull Request reviewed by the other
member of the pair. `main` is always the deliverable state.

## The cycle

```bash
git checkout main && git pull
git checkout -b <type>/<short-topic>      # e.g. feat/fen-parser, docs/adr-encodage
# ... commits ...
git pull --rebase origin main
git push -u origin <type>/<short-topic>
```

Branch and commit types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `perf`.
Messages: `type(scope): description` (e.g. `feat(my_torch): dense layer and backpropagation`).
Link the issue in the PR (`Closes #12`).

## Review

One approval is enough, then the author merges. Short branches: one PR = one idea.

## Before asking for a review

```bash
make re
make tests_run
find include src tests -name '*.[ch]pp' | xargs .venv-format/bin/clang-format --dry-run --Werror
```

The formatter is pinned: clang-format **18.1.8**, the version of the CI. Two versions do not format
the same way, so use this one, installed once in a git-ignored virtualenv:

```bash
python3 -m venv .venv-format && .venv-format/bin/pip install clang-format==18.1.8
find include src tests -name '*.[ch]pp' | xargs .venv-format/bin/clang-format -i   # to fix
```

The CI (`.github/workflows/ci.yml`) runs the same checks. A red PR is not merged.

## Decisions and documentation

A structural choice (board encoding, network architecture, loss function, optimizer, save
format…) is written in `docs/adr/` in the same PR. A training result cited in the docs must be
reproducible with a script from `scripts/` and a config from `configs/`.

## Project constraints, never to be broken

- **No neural-network library** (PyTorch, TensorFlow, Keras, scikit-learn, JAX...),
  and no automatic differentiation.
- Supervised learning is mandatory.
- `my_torch_analyzer` at the root; errors on stderr, exit code **84**; in `--predict`, one
  prediction per position, **in file order**.
- A pre-trained network named `my_torch_network*` is pushed and loads with the analyzer.
- Keep every training script and configuration so that training can be reproduced.
- Bonuses in `bonus/`. Makefile with `re`, `clean`, `fclean`.

## C++ conventions

- C++20, `-Wall -Wextra -Werror`, no warning tolerated. Format: `.clang-format`.
- The analyzer and the generator include only the headers in `include/my_torch/`: this is the contract
  between the two work packages, and any change goes through a PR reviewed by both members.
- No manual `new`/`delete`: standard containers and smart pointers.
- Errors are reported by exceptions, caught in `main`, which prints to stderr and returns 84.
- Every gradient computation is covered by a finite-difference test.
- `make debug` (sanitizers) must pass without errors before a PR that touches memory.
- Everything is written in English: documentation, code comments, commit messages, identifiers.

## Project file: decisions, impacts, hindsight

The documents in `docs/` record the decision-making (justification with trade-offs, impact
assessment, concrete alternative) and let someone follow the work **without reading the code**.
Map: [`docs/README.md`](docs/README.md), to be kept up to date whenever a document is added.

1. **Decisions: ADR** (`docs/adr/`): one file per decision, from `TEMPLATE.md`, with its line in the
   index. Written **when the decision is made**. "Consequences" contains at least one ❌.
   An accepted ADR is **never rewritten**: a new ADR supersedes it.
2. **Impacts**: performance (`docs/performance/`, method, machine, results, graphs and a
   reproducible script in `scripts/`), legal (`docs/legal/`), costs (`docs/business/costs.md`,
   measured kept separate from estimated), dependencies (`docs/dependencies.md`).
3. **Alternatives: global retrospective** (`docs/retrospective/`).
4. **Business** (`docs/business/`).

Rules: documentation **aligned with the code**, every number checked by a measurement, never an
estimate presented as a measurement, weaknesses written honestly.

## License

Contributions are accepted under the project's [MIT license](LICENSE) (ADR-0048).
