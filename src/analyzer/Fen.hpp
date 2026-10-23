#pragma once

#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace analyzer {

enum class Color { White, Black };

enum class Label { Nothing, CheckWhite, CheckBlack, CheckmateWhite, CheckmateBlack };

struct Castling {
    bool whiteKingside = false;
    bool whiteQueenside = false;
    bool blackKingside = false;
    bool blackQueenside = false;
};

// Case 0 = a8, 7 = h8, 56 = a1, 63 = h1 : l'ordre de lecture de la FEN.
// Guide : docs/lire-une-fen.md.
struct Position {
    std::array<char, 64> board{}; // '.' pour une case vide, sinon la lettre FEN ('K', 'p'…)
    Color sideToMove = Color::White;
    Castling castling;
    std::optional<int> enPassant;
    int halfmoveClock = 0;
    int fullmoveNumber = 1;
};

struct Sample {
    Position position;
    std::optional<Label> label;
};

// --train exige la réponse attendue, --predict l'accepte sans l'exiger.
enum class LabelPolicy { Required, Optional };

// Indice de case à partir de la colonne (0 = a) et de la rangée (1 à 8).
constexpr int square(int file, int rank)
{
    return (8 - rank) * 8 + file;
}

// Valident le format FEN, pas la légalité échiquéenne (rôle de l'oracle, #37). Exception : un roi
// de chaque couleur est exigé, sans quoi échec et mat n'ont pas de sens. Lèvent my_torch::Error.
Position parseFen(std::string_view fen);
Sample parseLine(std::string_view line, LabelPolicy policy);
// Ignore les lignes vides ; les erreurs sont préfixées par "fichier:ligne: ".
std::vector<Sample> parseFile(const std::string &path, LabelPolicy policy);

Label parseLabel(std::string_view text);
std::string_view labelName(Label label);

} // namespace analyzer
