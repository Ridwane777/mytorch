#include "Symmetry.hpp"

#include <cctype>

namespace analyzer {

namespace {

char swapCase(char piece)
{
    if (piece == '.')
        return piece;
    return static_cast<char>(std::isupper(static_cast<unsigned char>(piece))
                                 ? std::tolower(static_cast<unsigned char>(piece))
                                 : std::toupper(static_cast<unsigned char>(piece)));
}

} // namespace

Position mirrorFiles(const Position &position)
{
    Position result = position;

    for (int row = 0; row < 8; row++)
        for (int file = 0; file < 8; file++)
            result.board[row * 8 + (7 - file)] = position.board[row * 8 + file];
    result.castling = Castling{};
    if (position.enPassant)
        result.enPassant = *position.enPassant / 8 * 8 + (7 - *position.enPassant % 8);
    return result;
}

Position flipColors(const Position &position)
{
    Position result = position;

    for (int row = 0; row < 8; row++)
        for (int file = 0; file < 8; file++)
            result.board[(7 - row) * 8 + file] = swapCase(position.board[row * 8 + file]);
    result.sideToMove = position.sideToMove == Color::White ? Color::Black : Color::White;
    result.castling.whiteKingside = position.castling.blackKingside;
    result.castling.whiteQueenside = position.castling.blackQueenside;
    result.castling.blackKingside = position.castling.whiteKingside;
    result.castling.blackQueenside = position.castling.whiteQueenside;
    if (position.enPassant)
        result.enPassant = (7 - *position.enPassant / 8) * 8 + *position.enPassant % 8;
    return result;
}

Label flipLabel(Label label)
{
    switch (label) {
    case Label::CheckWhite:
        return Label::CheckBlack;
    case Label::CheckBlack:
        return Label::CheckWhite;
    case Label::CheckmateWhite:
        return Label::CheckmateBlack;
    case Label::CheckmateBlack:
        return Label::CheckmateWhite;
    case Label::Nothing:
    case Label::Stalemate:
        break;
    }
    return label;
}

Sample mirrorFiles(const Sample &sample)
{
    return Sample{mirrorFiles(sample.position), sample.label};
}

Sample flipColors(const Sample &sample)
{
    Sample result{flipColors(sample.position), std::nullopt};

    if (sample.label)
        result.label = flipLabel(*sample.label);
    return result;
}

std::vector<Sample> augment(const std::vector<Sample> &samples, Symmetries symmetries)
{
    const bool mirror = symmetries == Symmetries::Mirror || symmetries == Symmetries::All;
    const bool flip = symmetries == Symmetries::Flip || symmetries == Symmetries::All;
    std::vector<Sample> result;

    result.reserve(samples.size() * (1 + (mirror ? 1 : 0) + (flip ? 1 : 0) + (mirror && flip)));
    for (const Sample &sample : samples) {
        result.push_back(sample);
        if (mirror)
            result.push_back(mirrorFiles(sample));
        if (flip)
            result.push_back(flipColors(sample));
        if (mirror && flip)
            result.push_back(mirrorFiles(flipColors(sample)));
    }
    return result;
}

} // namespace analyzer
