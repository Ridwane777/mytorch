#include "Encoding.hpp"

#include <memory>
#include <string_view>

#include "my_torch/Error.hpp"

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

const std::vector<std::unique_ptr<Encoding>> &encodings()
{
    static const auto all = [] {
        std::vector<std::unique_ptr<Encoding>> list;
        list.push_back(std::make_unique<OneHot768>());
        list.push_back(std::make_unique<OneHot773>());
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

} // namespace analyzer
