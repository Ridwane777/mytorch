# ADR-0016: Input encoding of a position

- **Status**: Accepted
- **Date**: 2026-10-26
- **Deciders**: Meric-BG (author), Ridwane777 (accepted)

## Context
ADR-0014 gave the analyzer a table of named encodings and a first one, `fen-onehot-768`. Issue #14
adds three variants so that the choice of the input of the final network (#17) rests on a
measurement. The comparison is in [performance/03-encodings.md](../performance/03-encodings.md): same
network, same 20,000 training and 5,000 test positions, three runs per encoding.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| `fen-onehot-768` | Simplest, no hand-made feature | 57.2 % accuracy, cannot tell who is to move |
| `fen-onehot-773` | +5 inputs, 62.0 % accuracy | Still no knowledge of attacks |
| `fen-signed-385` | Half the time and memory (50 s, 147 MB) | 57.6 % accuracy, weak on `Checkmate White` |
| `fen-attacks-901` | Best: 67.2 % accuracy, macro F1 0.651 | Needs attack computation, 1.5× the time of 768, heuristic without pins |

## Decision
Use `fen-attacks-901` for the final network (#17), and keep the other three in the table for
comparison and as fallbacks. `configs/basic_network.conf` keeps `fen-onehot-768` as the baseline
example; the final network gets its own configuration in #17.

## Consequences
- ✅ +10 points of accuracy over the baseline, with the three runs of each encoding separated.
- ✅ The encoding table stays open: the other encodings remain selectable by name.
- ❌ Training is 1.5 times slower and uses 17 % more memory than `fen-onehot-768`.
- ❌ The attack map is hand-made knowledge in the input, less general than raw planes, and ignores pins.
- ❌ Measured on 20,000 positions with one network size; the gap may change with the full dataset
  or a larger network.
- Follow-up: repeat the comparison on the final configuration of #17; revisit if the gap drops below
  2 points.
