#pragma once

#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace analyzer {

enum class Color { White, Black };

// The five outputs of the subject, plus Stalemate for the stalemate bonus (#24, ADR-0015).
enum class Label { Nothing, CheckWhite, CheckBlack, CheckmateWhite, CheckmateBlack, Stalemate };

struct Castling {
    bool whiteKingside = false;
    bool whiteQueenside = false;
    bool blackKingside = false;
    bool blackQueenside = false;
};

// Square 0 = a8, 7 = h8, 56 = a1, 63 = h1: the FEN reading order.
// Guide: docs/reading-a-fen.md.
struct Position {
    std::array<char, 64> board{}; // '.' for an empty square, otherwise the FEN letter ('K', 'p'...)
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

// --train requires the expected answer, --predict accepts it without requiring it.
enum class LabelPolicy { Required, Optional };

// Square index from the file (0 = a) and the rank (1 to 8).
constexpr int square(int file, int rank)
{
    return (8 - rank) * 8 + file;
}

// Validate the FEN format, not chess legality (the oracle's job, #37). Exception: one king
// of each colour is required, otherwise check and checkmate are meaningless.
// Throw my_torch::Error.
Position parseFen(std::string_view fen);
Sample parseLine(std::string_view line, LabelPolicy policy);
// Skips empty lines; errors are prefixed with "file:line: ".
std::vector<Sample> parseFile(const std::string &path, LabelPolicy policy);

Label parseLabel(std::string_view text);
std::string_view labelName(Label label);

} // namespace analyzer
