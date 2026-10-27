#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "my_torch/Matrix.hpp"
#include "my_torch/Network.hpp"

namespace my_torch {

// Generic data splits (ADR-0027), on the model of scikit-learn's stratified splitters and PyTorch's
// Subset. They only see class indices, never the examples themselves. The randomness is
// permutation(count, seed): the same seed gives the same split on every toolchain. Every function
// throws my_torch::Error on invalid arguments.

// Splits the indices 0..classes.size()-1 into one part per proportion, each part keeping the
// class proportions: within a class, part j gets floor(proportion_j × count) examples, and the
// remaining ones go one each to the parts with the largest fractional remainders (lowest index on
// a tie). Proportions must be positive and sum to 1 (within 1e-9). Each part is sorted.
std::vector<std::vector<std::size_t>> stratifiedSplit(const std::vector<std::size_t> &classes,
                                                      const std::vector<Scalar> &proportions,
                                                      std::uint64_t seed);

struct Fold {
    std::vector<std::size_t> train;      // sorted
    std::vector<std::size_t> validation; // sorted
};

// k folds: every index is in the validation part of exactly one fold and in the training part of
// all the others. The examples of each class are dealt in turn over the folds, so every fold holds
// the same share of each class within one example. Needs 2 <= k <= classes.size().
std::vector<Fold> stratifiedKFold(const std::vector<std::size_t> &classes, std::size_t k,
                                  std::uint64_t seed);

// A view on some rows of another source, without copying: row i is row indices[i] of the source.
// Keeps a reference: the source must outlive the subset. Throws if an index is out of range.
class SubsetSource : public DataSource {
  public:
    SubsetSource(const DataSource &source, std::vector<std::size_t> indices);

    std::size_t size() const override;
    std::size_t inputWidth() const override;
    std::size_t targetWidth() const override;
    void fill(const std::vector<std::size_t> &rows, Matrix &inputs, Matrix &targets) const override;

  private:
    const DataSource &_source;
    std::vector<std::size_t> _indices;
};

// The class of every row of a source (classesOf its targets: argmax, or >= 0.5 for one column),
// read EVALUATION_CHUNK rows at a time.
std::vector<std::size_t> targetClasses(const DataSource &source);

} // namespace my_torch
