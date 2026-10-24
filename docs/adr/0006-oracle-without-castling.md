# ADR-0006: Rule-based oracle, move generation by copy and without castling

- **Status**: Accepted
- **Date**: 2026-10-23
- **Deciders**: Ridwane777
- **Issue**: #37

## Context

The oracle (`bonus/my_torch_oracle`) classifies a position by the rules of the game: check, mate,
stalemate or nothing. It is never used to answer in place of the network; it checks the dataset
labels, the positions produced by data augmentation (#38), and enables stalemate detection (#24).

Telling a check from a mate, or a stalemate from a position with nothing, requires knowing whether the
side to move has **at least one legal move**. Moves must therefore be generated, and we must decide
how to discard those that leave one's own king attacked (pinned piece, king fleeing to an attacked
square, en passant capture that uncovers an attack). We must also decide whether **castling** is
generated.

The board representation is fixed by ADR-0005: 64 `char` in FEN order.

## Options considered

**How to discard the moves that leave the king in check?**

| Option | Pros | Cons |
| --- | --- | --- |
| **Play each move on a copy and test whether the king is attacked** | A single rule covers pins, king escapes and discovered en passant captures; little code, easy to check | One position copy and one attack test per move: slower than a generator that only produces legal moves |
| Directly generate only the legal moves (computing pins and checking pieces) | Faster | Many special cases, each a source of bugs; nothing measurable to gain on 723,019 positions |

**Should castling be generated?**

| Option | Pros | Cons |
| --- | --- | --- |
| **Do not generate castling** | Avoids the most complex move (two pieces, four conditions); changes no verdict (see below) | Move counting (*perft*) is wrong on a position where castling is possible; the oracle cannot serve as a general move generator |
| Generate castling | Complete generator, exact *perft* everywhere | Extra code and tests for an identical result |

Why castling changes no verdict:

- **mate**: castling is forbidden when one's king is in check, so castling never escapes a check;
- **stalemate**: castling requires the square next to the king on its path (f1, d1, f8 or d8) to be
  empty and not attacked; the king can then step there, which is already a legal move. Castling is
  therefore never the only legal move.

## Decision

Moves are generated from how the pieces move (en passant and promotions included),
played on a copy, then kept only if the moving side's king is not attacked. Castling is not
generated. The position is classified by combining "king attacked?" and "at least one legal move?";
a position where the king of the side not to move is attacked is flagged as illegal.

## Consequences

- ✅ Checked by *perft* on two positions where no castling is possible: the starting position up to
  4 plies (20, 400, 8,902, 197,281) and an endgame with en passant captures and promotions up to
  5 plies (14, 191, 2,812, 43,238, 674,624), equal to the published reference values.
- ✅ On the 723,019 lines of the dataset: 100% agreement with the labels, no stalemate, no illegal
  position, in 5.2 s (measured on 2026-10-23,
  [`docs/verification-dataset.md`](../verification-dataset.md)).
- ✅ Castling rights are still read and validated by the parser (ADR-0005), so they are available
  for encoding and augmentation (#38).
- ❌ `perft` is only exact on positions with no possible castling; the tests choose their
  positions accordingly.
- ❌ The oracle is not a general move generator: it cannot be used as is by an AI that
  plays (#27) without adding castling.
- Follow-up: if an AI that plays (#27) is built, a new ADR will add castling and *perft* on
  positions that contain it.
