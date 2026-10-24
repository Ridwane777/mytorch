#pragma once

#include <vector>

#include "my_torch/Loss.hpp"

namespace my_torch {

// Loss functions. All of them average over the batch (one row per example) and throw
// my_torch::Error if prediction and target differ in shape or the batch is empty.

// Squared error: batch mean of Σ_j (p_j - t_j)². For regression.
class MSE : public Loss {
  public:
    Scalar value(const Matrix &prediction, const Matrix &target) const override;
    Matrix gradient(const Matrix &prediction, const Matrix &target) const override;
    std::string type() const override;
};

// Absolute error: batch mean of Σ_j |p_j - t_j|. Less sensitive to outliers.
class MAE : public Loss {
  public:
    Scalar value(const Matrix &prediction, const Matrix &target) const override;
    Matrix gradient(const Matrix &prediction, const Matrix &target) const override;
    std::string type() const override;
};

// Quadratic below delta, linear beyond: MSE near the target, MAE far from it.
class Huber : public Loss {
  public:
    explicit Huber(Scalar delta = 1.0);

    Scalar value(const Matrix &prediction, const Matrix &target) const override;
    Matrix gradient(const Matrix &prediction, const Matrix &target) const override;
    std::string type() const override;

  private:
    Scalar _delta;
};

// Binary cross-entropy on probabilities (the output of a Sigmoid): batch mean of
// -Σ_j [t_j·log p_j + (1 - t_j)·log(1 - p_j)]. Probabilities are bounded to [ε, 1 - ε].
class BinaryCrossEntropy : public Loss {
  public:
    Scalar value(const Matrix &prediction, const Matrix &target) const override;
    Matrix gradient(const Matrix &prediction, const Matrix &target) const override;
    std::string type() const override;
};

// Cross-entropy coupled to Softmax (ADR-0007). prediction = the final Softmax output, target = one
// distribution per row (one-hot). gradient() returns (p·W - w⊙q) / batch, i.e. (p - q) / batch
// without weights: the gradient with respect to the Softmax *input*, which never divides by p
// (that division blows up when the right class has a probability close to 0).
//
// - labelSmoothing in [0, 1[: the target becomes q = (1 - s)·t + s / classes, which stops the
//   network from pushing its scores to infinity.
// - classWeights: one weight per class (empty = all 1), for class imbalance (#16). The mean is
//   still divided by the batch size, not by the sum of the weights.
class CrossEntropy : public Loss {
  public:
    explicit CrossEntropy(Scalar labelSmoothing = 0, std::vector<Scalar> classWeights = {});

    Scalar value(const Matrix &prediction, const Matrix &target) const override;
    Matrix gradient(const Matrix &prediction, const Matrix &target) const override;
    std::string type() const override;
    bool fusesSoftmax() const noexcept override;

  private:
    Matrix smoothed(const Matrix &prediction, const Matrix &target) const;
    Scalar weight(std::size_t cls) const noexcept;

    Scalar _smoothing;
    std::vector<Scalar> _weights;
};

} // namespace my_torch
