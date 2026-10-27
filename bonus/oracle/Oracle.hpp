#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "Fen.hpp"

// Rule-based oracle (#37): a reference for the network, never answering in its place.
// Castling is not generated: it changes no verdict (ADR-0006).
namespace oracle {

using analyzer::Color;
using analyzer::Position;

struct Move {
    int from;
    int to;
    char promotion = 0;     // FEN letter of the promoted piece ('Q', 'n'...), 0 otherwise
    bool enPassant = false; // the captured pawn is not on the destination square
};

enum class Verdict {
    Nothing,
    CheckWhite,
    CheckBlack,
    CheckmateWhite,
    CheckmateBlack,
    Stalemate, // no check and no legal move (#24)
    Illegal    // the king of the side not to move is attacked
};

Color opponent(Color color);
bool isAttacked(const Position &position, int square, Color by);
// Squares of the pieces of colour `by` that attack `square` (two for a double check). Used by the
// dataset analysis (docs/performance/02-eda.md) to tell which piece gives check.
std::vector<int> attackers(const Position &position, int square, Color by);
int findKing(const Position &position, Color color);
bool inCheck(const Position &position, Color color);

Position play(const Position &position, const Move &move);
std::vector<Move> legalMoves(const Position &position);

Verdict classify(const Position &position);
// The label of each verdict; none for Illegal.
std::optional<analyzer::Label> toLabel(Verdict verdict);
const char *verdictName(Verdict verdict);

// Number of move sequences over depth half-moves. Exact only if no castling is possible.
std::uint64_t perft(const Position &position, int depth);

} // namespace oracle
