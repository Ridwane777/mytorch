#include "Saliency.hpp"

#include <string>
#include <vector>

#include "my_torch/Error.hpp"
#include "my_torch/Metrics.hpp"

namespace gui {

namespace {

// Inputs of one square: 12 piece planes of 64 (fen-onehot-768, -773, fen-attacks-901), then for
// fen-attacks-901 the White and Black attack maps at 773 and 837.
std::vector<std::size_t> squareInputs(const analyzer::Encoding &encoding, std::size_t square)
{
    std::vector<std::size_t> inputs;
    for (std::size_t plane = 0; plane < 12; plane++)
        inputs.push_back(plane * 64 + square);
    if (encoding.width() == 901) {
        inputs.push_back(773 + square);
        inputs.push_back(837 + square);
    }
    return inputs;
}

} // namespace

SquareSaliency saliency(my_torch::Network &network, const analyzer::Encoding &encoding,
                        const analyzer::Position &position)
{
    const std::string name = encoding.name();
    if (name.rfind("fen-onehot", 0) != 0 && name != "fen-attacks-901")
        throw my_torch::Error("saliency needs a fen-onehot or fen-attacks encoding, not " + name);
    SquareSaliency result;

    // Row 0: the position; then one row per occupied square, with its piece removed.
    std::vector<std::size_t> occupied;
    for (std::size_t square = 0; square < 64; square++)
        if (position.board[square] != '.')
            occupied.push_back(square);
    my_torch::Matrix rows(1 + occupied.size(), encoding.width());
    encoding.encode(position, &rows(0, 0));
    for (std::size_t k = 0; k < occupied.size(); k++) {
        analyzer::Position without = position;
        without.board[occupied[k]] = '.';
        encoding.encode(without, &rows(k + 1, 0));
    }
    const my_torch::Matrix output = network.predict(rows);
    result.predicted = my_torch::classesOf(output)[0];
    result.probability = output(0, result.predicted);
    for (std::size_t k = 0; k < occupied.size(); k++)
        result.occlusion[occupied[k]] = result.probability - output(k + 1, result.predicted);

    my_torch::Matrix input(1, encoding.width());
    for (std::size_t i = 0; i < encoding.width(); i++)
        input(0, i) = rows(0, i);
    const my_torch::Matrix gradient = network.inputGradient(input, result.predicted);
    for (std::size_t square = 0; square < 64; square++)
        for (const std::size_t i : squareInputs(encoding, square))
            result.gradient[square] += gradient(0, i) * input(0, i);
    return result;
}

} // namespace gui
