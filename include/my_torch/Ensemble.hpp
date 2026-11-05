#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "my_torch/Matrix.hpp"
#include "my_torch/Network.hpp"

namespace my_torch {

// How an Ensemble combines its members (ADR-0045). Every member outputs probabilities over the
// same labels, in the same order.
// - Average: the weighted mean of the members' probabilities (soft voting).
// - Vote: each member votes for its most probable class with its weight; the score of a class is
//   its share of the total weight. A tie goes to the tied class with the highest weighted mean
//   probability, then to the lowest index.
// - Stack: a combiner network reads the members' probabilities side by side (member 0's labels,
//   then member 1's…) and outputs the probabilities; it is trained apart, on data the members did
//   not learn (stacking).
enum class Combination { Average, Vote, Stack };

// Name in files and on command lines: "average", "vote", "stack". combinationFromName throws
// my_torch::Error on an unknown name.
const char *combinationName(Combination combination) noexcept;
Combination combinationFromName(const std::string &name);

// Several networks used as one (#40). Members may read different encodings of the same examples:
// predict takes one input matrix per member.
class Ensemble {
  public:
    explicit Ensemble(Combination combination = Combination::Average);

    // Throws my_torch::Error if the weight is not finite and > 0, or if the member's labels are
    // not those of the first member.
    void add(Network member, Scalar weight = 1);
    // Stack only: input width members × labels, one output per label. Checked by predict and save.
    void setCombiner(Network combiner);

    Combination combination() const noexcept;
    std::size_t size() const noexcept;
    Network &member(std::size_t index);
    const Network &member(std::size_t index) const;
    Scalar weight(std::size_t index) const;
    const std::vector<std::string> &labels() const;

    // One row of scores per example: probabilities (Average, Stack) or vote shares (Vote).
    // inputs[i] goes to member i; all have the same number of rows. Throws my_torch::Error on an
    // empty ensemble, a wrong number of inputs or mismatched rows, or a missing combiner.
    Matrix predict(const std::vector<Matrix> &inputs);
    // Same input for every member.
    Matrix predict(const Matrix &input);
    // Predicted class of each row, with the tie rule of Vote.
    std::vector<std::size_t> classify(const std::vector<Matrix> &inputs);

    // The members' probabilities side by side, the input of a Stack combiner.
    Matrix memberOutputs(const std::vector<Matrix> &inputs);

    // One binary file holding the combination, the weights, every member (each a complete binary
    // network file, ADR-0042) and the combiner, with a signature, a version and a checksum
    // (ADR-0045). load throws my_torch::Error on a malformed, truncated or corrupted file.
    void save(const std::string &path) const;
    static Ensemble load(const std::string &path);
    // True if the file starts with the signature of an ensemble file (false if it cannot be read).
    static bool isEnsembleFile(const std::string &path);

  private:
    void check() const;
    std::vector<Matrix> outputs(const std::vector<Matrix> &inputs);

    Combination _combination;
    std::vector<Network> _members;
    std::vector<Scalar> _weights;
    std::vector<Network> _combiner; // empty, or the single combiner of Stack
};

} // namespace my_torch
