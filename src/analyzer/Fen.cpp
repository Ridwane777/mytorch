#include "Fen.hpp"

#include <charconv>
#include <fstream>
#include <string>

#include "my_torch/Error.hpp"

namespace analyzer {

namespace {

using my_torch::Error;

constexpr std::string_view pieces = "KQRBNPkqrbnp";
constexpr std::array<std::string_view, 5> labelNames = {
    "Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"};

std::vector<std::string_view> split(std::string_view text, char separator)
{
    std::vector<std::string_view> parts;
    std::size_t start = 0;

    while (true) {
        const std::size_t end = text.find(separator, start);
        parts.push_back(text.substr(start, end - start));
        if (end == std::string_view::npos)
            return parts;
        start = end + 1;
    }
}

// Découpe aux espaces et tabulations en ignorant les séparateurs répétés.
std::vector<std::string_view> words(std::string_view text)
{
    std::vector<std::string_view> result;
    std::size_t i = 0;

    while (i < text.size()) {
        while (i < text.size() && (text[i] == ' ' || text[i] == '\t'))
            i++;
        const std::size_t start = i;
        while (i < text.size() && text[i] != ' ' && text[i] != '\t')
            i++;
        if (i > start)
            result.push_back(text.substr(start, i - start));
    }
    return result;
}

int parseCounter(std::string_view text, int minimum, const char *name)
{
    int value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);

    if (error != std::errc{} || end != text.data() + text.size() || value < minimum)
        throw Error(std::string(name) + " must be an integer >= " + std::to_string(minimum) +
                    ", got '" + std::string(text) + "'");
    return value;
}

bool isEmptyCount(char c)
{
    return c >= '1' && c <= '8';
}

void parseRank(std::string_view text, int rank, Position &position)
{
    int squares = 0;

    // Première passe : valider les caractères et compter les cases avant d'écrire quoi que ce soit.
    for (const char c : text) {
        if (!isEmptyCount(c) && pieces.find(c) == std::string_view::npos)
            throw Error("unknown piece '" + std::string(1, c) + "' on rank " +
                        std::to_string(rank));
        squares += isEmptyCount(c) ? c - '0' : 1;
    }
    if (squares != 8)
        throw Error("rank " + std::to_string(rank) + " has " + std::to_string(squares) +
                    " squares, expected 8");
    int file = 0;
    for (const char c : text) {
        if (isEmptyCount(c))
            for (int n = 0; n < c - '0'; n++)
                position.board[square(file++, rank)] = '.';
        else
            position.board[square(file++, rank)] = c;
    }
}

void parseBoard(std::string_view field, Position &position)
{
    const auto ranks = split(field, '/');

    if (ranks.size() != 8)
        throw Error("board must have 8 ranks, got " + std::to_string(ranks.size()));
    for (int r = 0; r < 8; r++)
        parseRank(ranks[r], 8 - r, position);
    int whiteKings = 0;
    int blackKings = 0;
    for (const char c : position.board) {
        whiteKings += c == 'K';
        blackKings += c == 'k';
    }
    if (whiteKings != 1 || blackKings != 1)
        throw Error("board must have exactly one king per color, got " +
                    std::to_string(whiteKings) + " white and " + std::to_string(blackKings) +
                    " black");
}

Color parseSide(std::string_view field)
{
    if (field == "w")
        return Color::White;
    if (field == "b")
        return Color::Black;
    throw Error("side to move must be 'w' or 'b', got '" + std::string(field) + "'");
}

Castling parseCastling(std::string_view field)
{
    Castling castling;

    if (field == "-")
        return castling;
    if (field.empty() || field.size() > 4)
        throw Error("invalid castling field '" + std::string(field) + "'");
    for (const char c : field) {
        bool *flag = c == 'K'   ? &castling.whiteKingside
                     : c == 'Q' ? &castling.whiteQueenside
                     : c == 'k' ? &castling.blackKingside
                     : c == 'q' ? &castling.blackQueenside
                                : nullptr;
        if (flag == nullptr || *flag)
            throw Error("invalid castling field '" + std::string(field) + "'");
        *flag = true;
    }
    return castling;
}

std::optional<int> parseEnPassant(std::string_view field)
{
    if (field == "-")
        return std::nullopt;
    if (field.size() != 2 || field[0] < 'a' || field[0] > 'h' ||
        (field[1] != '3' && field[1] != '6'))
        throw Error("invalid en passant square '" + std::string(field) + "'");
    return square(field[0] - 'a', field[1] - '0');
}

Position parseFields(const std::vector<std::string_view> &fields)
{
    Position position;

    parseBoard(fields[0], position);
    position.sideToMove = parseSide(fields[1]);
    position.castling = parseCastling(fields[2]);
    position.enPassant = parseEnPassant(fields[3]);
    position.halfmoveClock = parseCounter(fields[4], 0, "halfmove clock");
    position.fullmoveNumber = parseCounter(fields[5], 1, "fullmove number");
    return position;
}

std::string_view trimLineEnd(std::string_view line)
{
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
        line.remove_suffix(1);
    return line;
}

} // namespace

Position parseFen(std::string_view fen)
{
    const auto fields = words(trimLineEnd(fen));

    if (fields.size() != 6)
        throw Error("FEN must have 6 fields, got " + std::to_string(fields.size()));
    return parseFields(fields);
}

Sample parseLine(std::string_view line, LabelPolicy policy)
{
    const auto fields = words(trimLineEnd(line));
    Sample sample;

    if (fields.size() < 6)
        throw Error("FEN must have 6 fields, got " + std::to_string(fields.size()));
    sample.position = parseFields(fields);
    if (fields.size() > 6) {
        std::string label(fields[6]);
        for (std::size_t i = 7; i < fields.size(); i++)
            label += " " + std::string(fields[i]);
        sample.label = parseLabel(label);
    } else if (policy == LabelPolicy::Required) {
        throw Error("missing expected output after the FEN");
    }
    return sample;
}

std::vector<Sample> parseFile(const std::string &path, LabelPolicy policy)
{
    std::ifstream file(path);
    std::vector<Sample> samples;
    std::string line;
    std::size_t number = 0;

    if (!file)
        throw Error("cannot open '" + path + "'");
    while (std::getline(file, line)) {
        number++;
        if (words(trimLineEnd(line)).empty())
            continue;
        try {
            samples.push_back(parseLine(line, policy));
        } catch (const Error &error) {
            throw Error(path + ":" + std::to_string(number) + ": " + error.what());
        }
    }
    return samples;
}

Label parseLabel(std::string_view text)
{
    for (std::size_t i = 0; i < labelNames.size(); i++)
        if (text == labelNames[i])
            return static_cast<Label>(i);
    throw Error("unknown expected output '" + std::string(text) + "'");
}

std::string_view labelName(Label label)
{
    return labelNames[static_cast<std::size_t>(label)];
}

} // namespace analyzer
