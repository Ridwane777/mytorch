#pragma once

#include <cstddef>
#include <cstdint>

#include "my_torch/Layer.hpp"

namespace my_torch {

// Weight penalty of a Dense layer (ADR-0010): l1·Σ|w| + l2·Σw². Biases are never penalized.
struct Regularization {
    Scalar l1 = 0;
    Scalar l2 = 0;
};

// Fully connected layer: Z = X·W + b, with W (inputs × neurons) and one bias per neuron,
// b (1 × neurons), shared by every row of the batch.
class Dense : public Layer {
  public:
    // Temporary uniform initialization in ±sqrt(1 / inputs); Xavier and He come with #6.
    // Throws my_torch::Error if a regularization coefficient is negative.
    Dense(std::size_t inputs, std::size_t outputs, std::uint32_t seed = 42,
          Regularization regularization = {});

    Matrix forward(const Matrix &input, bool training) override;
    // dW += Xᵀ·G, db += sum of the rows of G, returns dX = G·Wᵀ. Gradients accumulate:
    // the training loop must reset them to zero before each batch.
    Matrix backward(const Matrix &gradOutput) override;
    std::vector<Parameter> parameters() override;
    std::string type() const override;
    // l1·Σ|w| + l2·Σw²; backward adds l1·sign(w) + 2·l2·w to dW.
    Scalar penalty() const override;
    const Regularization &regularization() const noexcept;

    Matrix &weights() noexcept;
    Matrix &bias() noexcept;

  private:
    Matrix _weights;
    Matrix _bias;
    Matrix _weightsGrad;
    Matrix _biasGrad;
    Matrix _input;
    Regularization _regularization;
};

} // namespace my_torch
