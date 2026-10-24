#include "Encoding.hpp"

#include <memory>
#include <string_view>

#include "my_torch/Error.hpp"

namespace analyzer {

namespace {

constexpr std::string_view pieceOrder = "PNBRQKpnbrqk";

const std::vector<std::unique_ptr<Encoding>> &encodings()
{
    static const auto all = [] {
        std::vector<std::unique_ptr<Encoding>> list;
        list.push_back(std::make_unique<OneHot768>());
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
    for (std::size_t square = 0; square < 64; square++) {
        const std::size_t plane = pieceOrder.find(position.board[square]);
        if (plane != std::string_view::npos)
            row[plane * 64 + square] = 1;
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

} // namespace analyzer
