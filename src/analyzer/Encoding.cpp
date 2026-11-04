#include "Encoding.hpp"

#include <array>
#include <cctype>
#include <memory>
#include <string_view>

#include "my_torch/Error.hpp"
#include "my_torch/Parallel.hpp"

namespace analyzer {

namespace {

constexpr std::string_view pieceOrder = "PNBRQKpnbrqk";

// Calls visit(plane, square) for every piece of the board; plane indexes pieceOrder.
template <typename Visit> void forEachPiece(const Position &position, Visit visit)
{
    for (std::size_t square = 0; square < 64; square++) {
        const std::size_t plane = pieceOrder.find(position.board[square]);
        if (plane != std::string_view::npos)
            visit(plane, square);
    }
}

// The 768 piece inputs shared by every one-hot layout.
void writePieces(const Position &position, my_torch::Scalar *row)
{
    forEachPiece(position,
                 [row](std::size_t plane, std::size_t square) { row[plane * 64 + square] = 1; });
}

// Side to move then K Q k q castling rights: the 5 inputs after the pieces.
void writeGameState(const Position &position, my_torch::Scalar *row)
{
    row[0] = position.sideToMove == Color::White ? 1 : 0;
    row[1] = position.castling.whiteKingside ? 1 : 0;
    row[2] = position.castling.whiteQueenside ? 1 : 0;
    row[3] = position.castling.blackKingside ? 1 : 0;
    row[4] = position.castling.blackQueenside ? 1 : 0;
}

// Squares attacked by a colour: those a piece of that colour could capture on, ignoring pins and
// whether the king would be left in check. A slider attacks up to and including the first
// occupied square on each ray, so the squares of defended pieces count as attacked. Square
// index as in Position: row = square / 8 (0 = rank 8), column = square % 8 (0 = file a).
std::array<bool, 64> attackedBy(const Position &position, Color color)
{
    constexpr int knightMoves[8][2] = {{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
                                       {1, -2},  {1, 2},  {2, -1},  {2, 1}};
    constexpr int straight[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    constexpr int diagonal[4][2] = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
    std::array<bool, 64> attacked{};

    const auto mark = [&attacked](int row, int column) {
        if (row >= 0 && row < 8 && column >= 0 && column < 8)
            attacked[static_cast<std::size_t>(row * 8 + column)] = true;
    };
    const auto slide = [&](int row, int column, const int (&directions)[4][2]) {
        for (const auto &direction : directions) {
            int r = row + direction[0];
            int c = column + direction[1];
            while (r >= 0 && r < 8 && c >= 0 && c < 8) {
                mark(r, c);
                if (position.board[static_cast<std::size_t>(r * 8 + c)] != '.')
                    break;
                r += direction[0];
                c += direction[1];
            }
        }
    };

    for (int square = 0; square < 64; square++) {
        const char piece = position.board[static_cast<std::size_t>(square)];
        if (piece == '.' ||
            (std::isupper(static_cast<unsigned char>(piece)) != 0) != (color == Color::White))
            continue;
        const int row = square / 8;
        const int column = square % 8;
        switch (std::toupper(static_cast<unsigned char>(piece))) {
        case 'P': {
            // White moves towards row 0 (square - 8), Black towards row 7.
            const int ahead = color == Color::White ? row - 1 : row + 1;
            mark(ahead, column - 1);
            mark(ahead, column + 1);
            break;
        }
        case 'N':
            for (const auto &move : knightMoves)
                mark(row + move[0], column + move[1]);
            break;
        case 'B':
            slide(row, column, diagonal);
            break;
        case 'R':
            slide(row, column, straight);
            break;
        case 'Q':
            slide(row, column, straight);
            slide(row, column, diagonal);
            break;
        case 'K':
            for (const auto &move : diagonal)
                mark(row + move[0], column + move[1]);
            for (const auto &move : straight)
                mark(row + move[0], column + move[1]);
            break;
        default:
            break;
        }
    }
    return attacked;
}

const std::vector<std::unique_ptr<Encoding>> &encodings()
{
    static const auto all = [] {
        std::vector<std::unique_ptr<Encoding>> list;
        list.push_back(std::make_unique<OneHot768>());
        list.push_back(std::make_unique<OneHot773>());
        list.push_back(std::make_unique<Signed385>());
        list.push_back(std::make_unique<Attacks901>());
        return list;
    }();
    return all;
}

} // namespace

std::string OneHot768::name() const
{
    return "fen-onehot-768";
}

std::size_t OneHot768::width() const
{
    return 768;
}

void OneHot768::encode(const Position &position, my_torch::Scalar *row) const
{
    writePieces(position, row);
}

std::string OneHot773::name() const
{
    return "fen-onehot-773";
}

std::size_t OneHot773::width() const
{
    return 773;
}

void OneHot773::encode(const Position &position, my_torch::Scalar *row) const
{
    writePieces(position, row);
    writeGameState(position, row + 768);
}

std::string Signed385::name() const
{
    return "fen-signed-385";
}

std::size_t Signed385::width() const
{
    return 385;
}

void Signed385::encode(const Position &position, my_torch::Scalar *row) const
{
    forEachPiece(position, [row](std::size_t plane, std::size_t square) {
        // pieceOrder lists the six white pieces first, then the same six in black.
        row[(plane % 6) * 64 + square] = plane < 6 ? 1 : -1;
    });
    row[384] = position.sideToMove == Color::White ? 1 : 0;
}

std::string Attacks901::name() const
{
    return "fen-attacks-901";
}

std::size_t Attacks901::width() const
{
    return 901;
}

void Attacks901::encode(const Position &position, my_torch::Scalar *row) const
{
    writePieces(position, row);
    writeGameState(position, row + 768);
    const std::array<bool, 64> white = attackedBy(position, Color::White);
    const std::array<bool, 64> black = attackedBy(position, Color::Black);
    for (std::size_t square = 0; square < 64; square++) {
        row[773 + square] = white[square] ? 1 : 0;
        row[837 + square] = black[square] ? 1 : 0;
    }
}

const Encoding &findEncoding(const std::string &name)
{
    std::string known;
    for (const auto &encoding : encodings()) {
        if (encoding->name() == name)
            return *encoding;
        known += (known.empty() ? "" : ", ") + encoding->name();
    }
    throw my_torch::Error("unknown input encoding '" + name + "' (known: " + known + ")");
}

my_torch::Matrix encodeRows(const Encoding &encoding, const std::vector<Sample> &samples,
                            std::size_t begin, std::size_t end)
{
    my_torch::Matrix rows(end - begin, encoding.width());

    for (std::size_t i = begin; i < end; i++)
        encoding.encode(samples[i].position, rows.data() + (i - begin) * encoding.width());
    return rows;
}

PositionSource::PositionSource(const std::vector<Sample> &samples,
                               const std::vector<std::size_t> &classes, std::size_t classCount,
                               const Encoding &encoding, std::vector<std::size_t> rows)
    : _samples(samples), _classes(classes), _classCount(classCount), _encoding(encoding),
      _rows(std::move(rows))
{
}

std::size_t PositionSource::size() const
{
    return _rows.size();
}

std::size_t PositionSource::inputWidth() const
{
    return _encoding.width();
}

std::size_t PositionSource::targetWidth() const
{
    return _classCount;
}

void PositionSource::fill(const std::vector<std::size_t> &rows, my_torch::Matrix &inputs,
                          my_torch::Matrix &targets) const
{
    // Each batch row is encoded on its own: encoding is pure and every row writes only its own
    // part of the matrices, so the rows are spread over the threads of the library (the encoding
    // of the attack planes would otherwise run on one thread while the others wait).
    my_torch::parallelFor(rows.size(), [&](std::size_t begin, std::size_t end) {
        for (std::size_t r = begin; r < end; r++) {
            const std::size_t sample = _rows[rows[r]];
            _encoding.encode(_samples[sample].position, inputs.data() + r * inputs.cols());
            targets(r, _classes[sample]) = 1;
        }
    });
}

} // namespace analyzer
