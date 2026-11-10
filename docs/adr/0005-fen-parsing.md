# ADR-0005: Reading FEN, format validation and board representation

- **Status**: Accepted
- **Date**: 2026-10-23
- **Deciders**: Ridwane777
- **Issue**: #13 (PR #48)

## Context

`my_torch_analyzer` reads files of positions in FEN notation, followed by a label (`--train`) or not
(`--predict`). A project requirement is invalid input to produce an error and exit code 84. The
dataset has 723,019 lines; it is read again at every training. Guide to the notation:
[`docs/reading-a-fen.md`](../reading-a-fen.md).

Two choices must be made: **how far to validate** a line, and **how to represent** the board in
memory before encoding it into network inputs (#14).

A position is **malformed** if the text does not follow the notation (a rank of 9 squares, an
unknown letter). It is **illegal** if the text is correct but the situation cannot
occur in a real game: a pawn on the 1st or 8th rank, two kings on adjacent squares,
nine white pawns, or the side not to move already in check.

## Options considered

**How far to validate?**

| Option | Pros | Cons |
| --- | --- | --- |
| Format only | The simplest; clear rule | Accepts a board with no king, on which "check" and "mate" make no sense |
| **Format + exactly one king per color** | Guarantees a usable line for classification; a one-line check | Lets the other illegal positions through |
| Format + full legality | Rejects everything that cannot occur in a game | Requires move generation (the oracle's job, #37); slower; a real game is never needed to classify a position |

**How to represent the board?**

| Option | Pros | Cons |
| --- | --- | --- |
| **64 `char` in FEN order** (square 0 = a8, 63 = h1; `'.'` = empty) | Direct copy of the text, readable in a debugger, testable square by square; filled with no computation | Finding the pieces of a type requires scanning the 64 squares |
| *Bitboards*: 12 64-bit integers, one per piece type and color, one bit per square | The chess-engine representation; attacks computed in a few bitwise operations; very fast for the oracle | Harder to read, debug and defend; useless for reading alone |
| 8 × 8 grid (`char[8][8]`) | Natural access by (rank, file) | Same content as the 64 squares with two indices to handle; no gain |

## Decision

The parser checks **the format** of the 6 fields and the label, plus **exactly one king of each
color**. It does not check chess legality, which is left to the oracle (#37). The board is an array
of 64 `char` in FEN order, held in a `Position` separate from the encoding.

## Consequences

- ✅ Every accepted line can be classified; every malformed line throws `my_torch::Error` with
  `file:line: reason`, hence exit code 84.
- ✅ Fast: the 723,019 lines of the dataset are read and validated in 0.55 s with 25 MB of memory
  (measured on 2026-10-23, `-O2` build, check program outside the repository); no line rejected.
- ✅ Reading and encoding are separate: the encoding can change (#14, CNN #29) without touching the
  parser and its tests.
- ❌ Illegal but well-formed positions are accepted; they will be flagged by the oracle
  (#37), not by the parser.
- ❌ The oracle will have to scan the 64 squares to find the pieces, more slowly than with
  *bitboards*.
- Follow-up: if the oracle (#37) or data augmentation (#38) turn out too slow with this
  representation, a new ADR will introduce *bitboards* internally, without changing the interface of
  `Position`.
