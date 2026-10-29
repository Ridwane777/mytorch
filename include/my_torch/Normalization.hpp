#pragma once

#include <cstddef>

#include "my_torch/Layer.hpp"

namespace my_torch {

// Normalization layers (ADR-0033), placed between a Dense layer and its activation. Each feature is
// normalized, x̂ = (x - mean) / sqrt(variance + epsilon), then scaled and shifted by two trained
// parameters per feature: y = gamma·x̂ + beta (gamma starts at 1, beta at 0, so a new layer only
// normalizes). They keep the width of their input.

// Batch normalization (Ioffe and Szegedy, 2015): mean and variance of each column over the rows of
// the batch. In training they come from the batch, and running averages are updated:
// running = (1 - momentum)·running + momentum·batch, with the unbiased variance (batch of m rows:
// sum of squares / (m - 1)), as PyTorch does. In prediction the running averages are used, so one
// position gives the same answer alone or in a batch. A training batch of one row normalizes to 0
// and leaves the running averages unchanged.
class BatchNorm : public Layer {
  public:
    // Throws my_torch::Error unless features > 0, 0 < momentum <= 1 and epsilon > 0.
    explicit BatchNorm(std::size_t features, Scalar momentum = 0.1, Scalar epsilon = 1e-5);

    Matrix forward(const Matrix &input, bool training) override;
    // dgamma += Σ g·x̂, dbeta += Σ g; dX from the batch statistics of the last training forward.
    Matrix backward(const Matrix &gradOutput) override;
    std::vector<Parameter> parameters() override;
    std::string type() const override;
    // "batch_norm <features> momentum=… epsilon=…", then the lines gamma, beta, running mean and
    // running variance.
    void save(std::ostream &out) const override;
    // The running averages, which are not parameters, for checkpoints (ADR-0032).
    void saveState(std::ostream &out) const override;
    void loadState(std::istream &in) override;

    std::size_t features() const noexcept;
    Scalar momentum() const noexcept;
    Scalar epsilon() const noexcept;
    Matrix &gamma() noexcept;
    Matrix &beta() noexcept;
    Matrix &runningMean() noexcept;
    Matrix &runningVariance() noexcept;

  private:
    Scalar _momentum;
    Scalar _epsilon;
    Matrix _gamma;
    Matrix _beta;
    Matrix _gammaGrad;
    Matrix _betaGrad;
    Matrix _runningMean;
    Matrix _runningVariance;
    Matrix _normalized; // x̂ of the last training forward
    Matrix _inverseStd; // 1 / sqrt(variance + epsilon) per column, same forward
};

// Layer normalization (Ba, Kiros and Hinton, 2016): mean and variance of each row over its
// features. The same computation in training and in prediction, independent of the batch size.
class LayerNorm : public Layer {
  public:
    // Throws my_torch::Error unless features > 0 and epsilon > 0.
    explicit LayerNorm(std::size_t features, Scalar epsilon = 1e-5);

    Matrix forward(const Matrix &input, bool training) override;
    Matrix backward(const Matrix &gradOutput) override;
    std::vector<Parameter> parameters() override;
    std::string type() const override;
    // "layer_norm <features> epsilon=…", then the lines gamma and beta.
    void save(std::ostream &out) const override;

    std::size_t features() const noexcept;
    Scalar epsilon() const noexcept;
    Matrix &gamma() noexcept;
    Matrix &beta() noexcept;

  private:
    Scalar _epsilon;
    Matrix _gamma;
    Matrix _beta;
    Matrix _gammaGrad;
    Matrix _betaGrad;
    Matrix _normalized; // x̂ of the last forward
    Matrix _inverseStd; // 1 / sqrt(variance + epsilon) per row, same forward
};

} // namespace my_torch
