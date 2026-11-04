#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "Fen.hpp"
#include "my_torch/Matrix.hpp"
#include "my_torch/Network.hpp"

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

// OneHot768 plus the state of the game, 773 inputs: [0, 768) as OneHot768, [768] = 1 when White
// is to move, [769..772] = 1 for each castling right still available, in the order K Q k q.
class OneHot773 : public Encoding {
  public:
    std::string name() const override;
    std::size_t width() const override;
    void encode(const Position &position, my_torch::Scalar *row) const override;
};

// Six planes of 64 squares, in the order P N B R Q K: +1 for a white piece, -1 for a black one, 0
// for an empty square (a piece type shares its plane between the colours), then [384] = 1 when
// White is to move. 385 inputs, no castling.
class Signed385 : public Encoding {
  public:
    std::string name() const override;
    std::size_t width() const override;
    void encode(const Position &position, my_torch::Scalar *row) const override;
};

// OneHot773 plus attack maps, 901 inputs: [0, 773) as OneHot773, [773..836] = 1 for each square
// attacked by a White piece, [837..900] = 1 for each square attacked by a Black piece (square
// index as in Position). A square is attacked when a piece could capture there, pins ignored; a
// slider attacks up to and including the first piece in its way. This is feature extraction (it
// hands the network some of the rules of the game), not a check verdict, and it is computed here
// because the analyzer does not depend on the bonus oracle.
class Attacks901 : public Encoding {
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

// Training data for my_torch::Network::fit built on demand (ADR-0021): keeps the positions and
// the class of each one, and encodes only the rows of the batch fit asks for. rows selects which
// samples belong to this set (the training or the validation part).
class PositionSource : public my_torch::DataSource {
  public:
    // Keeps references: samples, classes and encoding must outlive the source.
    PositionSource(const std::vector<Sample> &samples, const std::vector<std::size_t> &classes,
                   std::size_t classCount, const Encoding &encoding, std::vector<std::size_t> rows);

    std::size_t size() const override;
    std::size_t inputWidth() const override;
    std::size_t targetWidth() const override;
    void fill(const std::vector<std::size_t> &rows, my_torch::Matrix &inputs,
              my_torch::Matrix &targets) const override;

  private:
    const std::vector<Sample> &_samples;
    const std::vector<std::size_t> &_classes;
    std::size_t _classCount;
    const Encoding &_encoding;
    std::vector<std::size_t> _rows;
};

} // namespace analyzer
