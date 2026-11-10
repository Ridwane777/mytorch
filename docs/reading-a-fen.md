# Reading a dataset line (FEN notation)

A guide to understanding the data without knowing chess. It is the reference for the FEN parser (#13)
and for the position encoding.

## 1. The dataset

The dataset is `dataset.zip` at the repository root (provided by Epitech, redistributed with authorization): `unzip dataset.zip` gives `dataset/` (git-ignored). It contains three folders
(`check/`, `checkmate/`, `nothing/`), each with `10_pieces.txt`, `20_pieces.txt` and `many_pieces.txt`.
One line = one position + the expected answer. Distribution measured on the 723,019 lines:

| Label | Lines |
| --- | --- |
| `Nothing` | 300,000 |
| `Check White` | 135,587 |
| `Check Black` | 113,713 |
| `Checkmate White` | 94,295 |
| `Checkmate Black` | 79,424 |

## 2. The useful chess rules

Two sides, **White** and **Black**, play **in turn**, White first. Each side has 16 pieces.

| Piece | Letter | Movement |
| --- | --- | --- |
| King | K | 1 square in every direction |
| Queen | Q | In a straight line or diagonally, as many squares as wanted |
| Rook | R | In a straight line (horizontal or vertical), as many squares as wanted |
| Bishop | B | Diagonally, as many squares as wanted |
| Knight (*kNight*) | N | In an "L": 2 squares then 1 to the side; the only piece that jumps over others |
| Pawn | P | Moves forward 1 square (2 on its first move), **captures diagonally** 1 square forward |

Pieces, except the knight, do not pass through others: a piece in the way blocks. Capturing a piece means
moving onto its square.

What the network must recognize:

- **Check**: a king is attacked by an enemy piece. The player must get out of it on the next move.
- **Checkmate**: the king is attacked and **no move** gets it out. The game is over.
- **Nothing**: no king is attacked.

"Check White" reads **"White gives check"** (it is the black king that is attacked); "Checkmate Black": Black
gives checkmate.

## 3. The 6 fields of a line

Real example (`dataset/check/10_pieces.txt`):

```
8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59 Check White
└──────────┬─────────────┘ │ │ │ │ └┬┘ └────┬────┘
        field 1            2 3 4 5  6    label (not FEN)
```

| # | Field | Example | Meaning |
| --- | --- | --- | --- |
| 1 | Piece placement | `8/8/R2k4/…` | The 8 ranks, from the 8th to the 1st, separated by `/` |
| 2 | Side to move | `b` | Who must play: `w` White, `b` Black |
| 3 | Castling | `-` | Castling still possible (`KQkq`), `-` if none |
| 4 | En passant | `-` | Square where an en passant capture is possible, otherwise `-` |
| 5 | Half-moves | `7` | Half-moves since the last capture or the last pawn move |
| 6 | Move number | `59` | Number of the current full move, starts at 1 |
| + | Label | `Check White` | Expected answer, added by the dataset; may contain a space |

## 4. Translating field 1, step by step

### Step 1: split at the `/`

You get 8 segments = 8 ranks, **from top to bottom** (rank 8, black side, down to rank 1, white side).

| Segment | `8` | `8` | `R2k4` | `4r1p1` | `8` | `5K2` | `5P2` | `8` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rank | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 |

### Step 2: read each segment from left to right (files a → h)

- a **digit** = that many empty squares;
- a **letter** = a piece on the next square. **Uppercase = white, lowercase = black.**

```
R2k4  : R → white rook on a6; 2 → b6, c6 empty; k → black king on d6; 4 → e6 to h6 empty
4r1p1 : 4 → a5 to d5 empty; r → black rook on e5; 1 → f5 empty; p → black pawn on g5; 1 → h5 empty
5K2   : 5 → a3 to e3 empty; K → white king on f3; 2 → g3, h3 empty
5P2   : 5 → empty; P → white pawn on f2; 2 → empty
```

Check: in each segment, digits + letters = 8 (`4r1p1` → 4 + 1 + 1 + 1 + 1 = 8).

### Step 3: the chessboard

```
8 | .  .  .  .  .  .  .  . |
7 | .  .  .  .  .  .  .  . |
6 | R  .  .  k  .  .  .  . |
5 | .  .  .  .  r  .  p  . |
4 | .  .  .  .  .  .  .  . |
3 | .  .  .  .  .  K  .  . |
2 | .  .  .  .  .  P  .  . |
1 | .  .  .  .  .  .  .  . |
    a  b  c  d  e  f  g  h
```

White: king f3, rook a6, pawn f2. Black: king d6, rook e5, pawn g5.

### Step 4: check the label

The white rook on a6 looks to the right: b6 and c6 are empty, then d6 = black king. The black king is attacked,
White gives check: **Check White**. Consistent with side to move `b`: it is Black's turn to get out of check.

## 5. Fields 2 to 6

### Side to move (field 2)

`w` (*white*) or `b` (*black*): the side that must play. It is usually the side that is in check.

### Castling (field 3)

A special move that moves **the king and a rook at the same time** to shelter the king in a corner: the king moves 2
squares toward the rook, the rook goes to the other side of the king. Kingside castling on the king's side (h-file),
queenside castling on the queen's side (a-file). Allowed only if neither the king nor that rook has moved, if nothing
separates them and **if the king is not in check**.

| Letter | Castling still possible |
| --- | --- |
| `K` | White, kingside castling |
| `Q` | White, queenside castling |
| `k` | Black, kingside castling |
| `q` | Black, queenside castling |
| `-` | None |

### En passant capture (field 4)

A special pawn rule: a pawn that moves **2 squares** at once can be captured by an enemy pawn standing next to it,
**as if it had moved only 1 square**. This right lasts only one move. The field gives the skipped square (`e3`, `d6`…),
always on the 3rd or 6th rank, or `-`.

### Half-moves (field 5) and move number (field 6)

- A **half-move** is the move of **a single** player, whichever it is. A **full move** is a white move followed by the
  black reply.
- **Field 5** counts the half-moves played **since the last capture or the last pawn move**. It goes back to 0 at
  each capture or pawn move. At 100 (50 moves on each side), a player can claim a draw (50-move rule).
- **Field 6** is the number of the **current** full move, captures included. It starts at 1 and increases after
  each Black move.

| Half-move | Player | Action | Field 5 |
| --- | --- | --- | --- |
| 1 | White | Moves a pawn | 0 |
| 2 | Black | Moves a knight | 1 |
| 3 | White | Moves a bishop | 2 |
| 4 | Black | Moves a rook | 3 |
| 5 | White | Its queen captures a piece | 0 |
| 6 | Black | Moves its king | 1 |

In `… b - - 7 59`: 7 half-moves without a capture or pawn move; 58 full moves are finished, White has played
its 59th move and it is Black's turn to play theirs.

## 6. What the network needs

| Field | Usefulness for detecting check and checkmate |
| --- | --- |
| 1. Pieces | Essential |
| 2. Side to move | Useful: the side to move is the one in check |
| 3. Castling | Low: a king in check cannot castle; useful for data augmentation (#38) |
| 4. En passant | Very rare: can be the only move that avoids a checkmate |
| 5–6. Counters | None |

The parser reads and **validates** the 6 fields, because a malformed line must give an error (code 84); the encoding
then chooses what it keeps.

## 7. Exercise

Draw `4k3/8/8/8/8/8/8/4K2R` (3 pieces).

<details>
<summary>Solution</summary>

```
8 | .  .  .  .  k  .  .  . |
7 | .  .  .  .  .  .  .  . |
  …        (ranks 6 to 2 empty)
1 | .  .  .  .  K  .  .  R |
    a  b  c  d  e  f  g  h
```

Black king on e8, white king on e1, white rook on h1.

</details>
