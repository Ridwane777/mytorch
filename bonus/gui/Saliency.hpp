#pragma once

#include <array>
#include <cstddef>

#include "Encoding.hpp"
#include "Fen.hpp"
#include "my_torch/Network.hpp"

// Saliency per square (ADR-0035): which squares made the network give its answer. Used by the GUI
// and by scripts/inspect.cpp.
namespace gui {

struct SquareSaliency {
    std::size_t predicted = 0; // class of the network
    my_torch::Scalar probability = 0;
    // Square index as in analyzer::Position (0 = a8, 63 = h1).
    // gradient: Σ gradient × input over the inputs of the square (its piece and, for
    // fen-attacks-901, whether White and Black attack it), the gradient being that of the
    // probability of the predicted class: positive where the square supports the answer.
    std::array<my_torch::Scalar, 64> gradient{};
    // occlusion: probability of the predicted class minus the same probability with the piece of
    // the square removed (attack maps recomputed); 0 on an empty square.
    std::array<my_torch::Scalar, 64> occlusion{};
};

// Throws my_torch::Error for an encoding without one input per piece and square (the fen-onehot
// and fen-attacks encodings have them).
SquareSaliency saliency(my_torch::Network &network, const analyzer::Encoding &encoding,
                        const analyzer::Position &position);

} // namespace gui
