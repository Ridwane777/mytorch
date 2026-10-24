Closes #

## What it changes

<!-- One or two sentences: what a reviewer must understand before reading the code. -->

## Related decision

<!-- The ADR (docs/adr/) that this PR applies or calls into question, or "none". -->

## How to check it

<!-- The command to run, or what to look at. -->

## Before asking for a review

- [ ] `make re` builds without warnings
- [ ] `make tests_run` passes
- [ ] `make debug` then the tested case: no sanitizer errors
- [ ] `clang-format --dry-run --Werror` reports nothing
- [ ] the branch is up to date with `main` (`git pull --rebase origin main`)
- [ ] no neural-network library (PyTorch, TensorFlow…) is imported
- [ ] if the PR settles a structural choice, the ADR is written and added to the index
- [ ] if the PR changes a measured result, `docs/performance/` is up to date with its script
