#pragma once

#include <vector>

#include "Fen.hpp"

// Board symmetries that keep the chess label (data augmentation, #38, ADR-0019).
// Square 0 = a8, so a row of 8 board entries is a rank, from rank 8 down to rank 1.
namespace analyzer {

// Left-right mirror: file f becomes 7 - f (a <-> h), pieces and side to move unchanged, the en
// passant file is mirrored. Every castling right is cleared: after the mirror the king is no longer
// on its starting square, so keeping the rights would describe an impossible position. The label
// (check, checkmate, stalemate) does not depend on the files and is kept as is.
Position mirrorFiles(const Position &position);

// Top-bottom flip with the colours swapped: rank r becomes 9 - r, every piece changes colour, the
// side to move changes, castling rights and the en passant square follow (K <-> k, Q <-> q, rank
// 3 <-> 6). The clocks are unchanged. The result is the same game seen from the other side.
Position flipColors(const Position &position);

// Label of the flipped position. "Check White" means White gives check (reading-a-fen.md), so
// White <-> Black swaps for check and checkmate; Nothing and Stalemate are colourless.
Label flipLabel(Label label);

// The same operations on a sample: the label is kept by the mirror, swapped by the flip, and
// stays absent when the sample has none.
Sample mirrorFiles(const Sample &sample);
Sample flipColors(const Sample &sample);

enum class Symmetries { None, Mirror, Flip, All };

// For each input sample, in this order: the original, its mirror, its flip, its mirror and flip
// (only the enabled ones). Symmetries::None returns a copy of the input.
std::vector<Sample> augment(const std::vector<Sample> &samples, Symmetries symmetries);

} // namespace analyzer
