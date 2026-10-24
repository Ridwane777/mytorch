#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "Fen.hpp"
#include "my_torch/Matrix.hpp"

namespace analyzer {

// Turns a position into the input row of a network (ADR-0014). The network file names its
// encoding; the analyzer looks the name up here, so encodings can be added (#14) without
// touching the command line.
class Encoding {
  public:
    virtual ~Encoding() = default;

    virtual std::string name() const = 0;
    virtual std::size_t width() const = 0;
    // Writes width() values into row, which the caller has zeroed.
    virtual void encode(const Position &position, my_torch::Scalar *row) const = 0;
};

// One-hot of the pieces: 12 planes of 64 squares, in the order P N B R Q K p n b r q k, square
// index as in Position (a8 = 0). 768 inputs, nothing else (no side to move, no castling).
class OneHot768 : public Encoding {
  public:
    std::string name() const override;
    std::size_t width() const override;
    void encode(const Position &position, my_torch::Scalar *row) const override;
};

// Throws my_torch::Error for an unknown name, listing the known ones.
const Encoding &findEncoding(const std::string &name);

// Rows [begin, end) of samples, encoded.
my_torch::Matrix encodeRows(const Encoding &encoding, const std::vector<Sample> &samples,
                            std::size_t begin, std::size_t end);

} // namespace analyzer
