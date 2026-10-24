#pragma once

#include <cstddef>
#include <cstdint>

#include "my_torch/Layer.hpp"

namespace my_torch {

// Fully connected layer: Z = X·W + b, with W (inputs × neurons) and one bias per neuron,
// b (1 × neurons), shared by every row of the batch.
class Dense : public Layer {
  public:
    // Temporary uniform initialization in ±sqrt(1 / inputs); Xavier and He come with #6.
    Dense(std::size_t inputs, std::size_t outputs, std::uint32_t seed = 42);

    Matrix forward(const Matrix &input, bool training) override;
    // dW += Xᵀ·G, db += sum of the rows of G, returns dX = G·Wᵀ. Gradients accumulate:
    // the training loop must reset them to zero before each batch.
    Matrix backward(const Matrix &gradOutput) override;
    std::vector<Parameter> parameters() override;
    std::string type() const override;

    Matrix &weights() noexcept;
    Matrix &bias() noexcept;

  private:
    Matrix _weights;
    Matrix _bias;
    Matrix _weightsGrad;
    Matrix _biasGrad;
    Matrix _input;
};

} // namespace my_torch
