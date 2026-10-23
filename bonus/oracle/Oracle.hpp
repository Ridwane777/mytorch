#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "Fen.hpp"

// Oracle par les règles du jeu (#37) : sert de référence au réseau, jamais à répondre à sa place.
// Le roque n'est pas généré : il ne change aucun verdict (ADR-0006).
namespace oracle {

using analyzer::Color;
using analyzer::Position;

struct Move {
    int from;
    int to;
    char promotion = 0;     // lettre FEN de la pièce promue ('Q', 'n'…), 0 sinon
    bool enPassant = false; // le pion pris n'est pas sur la case d'arrivée
};

enum class Verdict {
    Nothing,
    CheckWhite,
    CheckBlack,
    CheckmateWhite,
    CheckmateBlack,
    Stalemate, // pas d'échec et aucun coup légal (#24)
    Illegal    // le roi du camp qui n'a pas le trait est attaqué
};

Color opponent(Color color);
bool isAttacked(const Position &position, int square, Color by);
int findKing(const Position &position, Color color);
bool inCheck(const Position &position, Color color);

Position play(const Position &position, const Move &move);
std::vector<Move> legalMoves(const Position &position);

Verdict classify(const Position &position);
// Les 5 labels du sujet ; rien pour Stalemate et Illegal.
std::optional<analyzer::Label> toLabel(Verdict verdict);
const char *verdictName(Verdict verdict);

// Nombre de suites de coups sur depth demi-coups. Exact seulement si aucun roque n'est possible.
std::uint64_t perft(const Position &position, int depth);

} // namespace oracle
