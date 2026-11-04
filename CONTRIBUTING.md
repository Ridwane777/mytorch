# How we work on this repository

## The main rule

**Nobody pushes directly to `main`.** Changes get in through a Pull Request reviewed by the other
member of the pair. The autograder evaluates from `main`, and the defense is done from it.

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
See `AGENTS.md`.
