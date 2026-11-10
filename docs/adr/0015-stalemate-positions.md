# ADR-0015: Stalemate bonus, generated sparse positions with generated negatives

- **Status**: Accepted
- **Date**: 2026-10-24
- **Deciders**: Ridwane777
- **Issue**: #24

## Context

Handling stalemates is a bonus of the specification. The oracle (ADR-0006) detects them, but the dataset
contains none (checked on all 723,019 lines), so a network cannot learn the class without new
positions. Three things must be decided: how to produce stalemate positions, how to keep the
network from learning a shortcut, and how the class reaches the analyzer.

## Options considered

**How to produce stalemates**

| Option | Pros | Cons |
| --- | --- | --- |
| **Sparse random boards (two kings and 1 to 6 pieces), kept when legal, labeled by the oracle** | Stalemates are mostly endgames with few pieces, so they appear (0.10 % of legal boards; the whole experiment runs in 72 s); reproducible from a seed | Not positions from real games |
| Play random moves from dataset positions until a stalemate | Positions closer to real games | Stalemates almost never occur: far too slow |

**What else to generate**

| Option | Pros | Cons |
| --- | --- | --- |
| **The same number of sparse non-stalemates, drawn the same way, labeled by the oracle** | The number of pieces alone no longer gives the answer | Adds sparse positions unlike the dataset |
| Stalemates only | Simplest | The network can learn "few pieces → stalemate" |

**How the class reaches the analyzer**

| Option | Pros | Cons |
| --- | --- | --- |
| **A sixth label `Stalemate` accepted by the FEN parser; the network's labels say whether it is used** | Nothing else changes: a 6-label network answers `Stalemate`, a 5-label one never does | The parser accepts a label the specification does not list |
| A separate analyzer for the bonus | Main code untouched | Duplicates the analyzer |

## Decision

`bonus/my_torch_positions COUNT SEED [SHARE]` draws sparse boards with SplitMix64 from SEED, keeps
the legal ones, and returns COUNT positions labeled by the oracle, a SHARE of them stalemates and the
rest other sparse positions, interleaved in a seeded order. The FEN parser accepts `Stalemate` as a
sixth label and the oracle maps its stalemate verdict to it. The experiment is reproduced by
`bonus/scripts/stalemate_experiment.sh`.

## Consequences

- ✅ A six-class network trained on 20,000 dataset and generated positions recognizes 69.9 % of
  unseen test stalemates (`docs/performance/14-stalemate.md`).
- ✅ Every generated position is legal and its label equals the oracle's verdict; the same seed
  gives the same positions (tested).
- ❌ The network also calls 200 non-stalemates stalemates (precision 55.5 %): the generated
  negatives reduce the sparse-board shortcut without removing it.
- ❌ The baseline encoding has no side to move, which a stalemate depends on; to measure again with
  the encodings of #14.
- ❌ The parser now accepts `Stalemate`, a label the specification does not list; a file using it is only
  meaningful for a network that has this label.
- Follow-up: measure again with an encoding that includes the side to move (#14) and on more data
  once the memory limit is solved (#17).
