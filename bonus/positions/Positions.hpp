#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "Oracle.hpp"

// Labeled position generator for the stalemate bonus (#24, ADR-0015). The dataset holds no
// stalemate, so they are generated: sparse random boards (two kings and 1 to 6 other pieces),
// kept when legal, labeled by the oracle. Non-stalemate boards are drawn the same way, so that
// the number of pieces does not give the answer away.
namespace positions {

using analyzer::Position;

std::string toFen(const Position &position);

// A random sparse board: kings on two squares, `pieces` other pieces (no pawn on the first or last
// rank), side to move at random, no castling, no en passant. It may be illegal.
Position randomBoard(std::uint64_t &state, int pieces);

struct Labeled {
    Position position;
    oracle::Verdict verdict;
};

struct Report {
    std::uint64_t drawn = 0;      // boards drawn
    std::uint64_t legal = 0;      // of which legal
    std::uint64_t stalemates = 0; // of which stalemates
};

// count labeled positions, round(count · stalemateShare) of them stalemates, the others legal
// non-stalemates, in a shuffled order. The same seed gives the same positions.
// Throws my_torch::Error if stalemateShare is outside [0, 1].
std::vector<Labeled> generate(std::uint64_t count, std::uint64_t seed, double stalemateShare,
                              Report *report = nullptr);

} // namespace positions
