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

// How the weights of a Dense layer are drawn (ADR-0020). With fan_in = inputs and
// fan_out = outputs, the weights have zero mean and a variance of:
//   Lecun*   1 / fan_in          Xavier*   2 / (fan_in + fan_out)          He*   2 / fan_in
// *Uniform draws from U(±sqrt(3·variance)), *Normal from the untruncated N(0, variance).
// Uniform is the original default, U(±sqrt(1 / fan_in)): variance 1 / (3·fan_in).
// Orthogonal gives W orthonormal columns (inputs >= outputs) or rows (inputs < outputs), gain 1.
enum class Init {
    Uniform,
    LecunUniform,
    LecunNormal,
    XavierUniform,
    XavierNormal,
    HeUniform,
    HeNormal,
    Orthogonal
};

// Fully connected layer: Z = X·W + b, with W (inputs × neurons) and one bias per neuron,
// b (1 × neurons), shared by every row of the batch.
class Dense : public Layer {
  public:
    // Weights drawn from std::mt19937(seed) following `init`, biases at zero.
    // Throws my_torch::Error if a regularization coefficient is negative.
    Dense(std::size_t inputs, std::size_t outputs, std::uint32_t seed = 42,
          Regularization regularization = {}, Init init = Init::Uniform);

    Matrix forward(const Matrix &input, bool training) override;
    // dW += Xᵀ·G, db += sum of the rows of G, returns dX = G·Wᵀ. Gradients accumulate:
    // the training loop must reset them to zero before each batch.
    Matrix backward(const Matrix &gradOutput) override;
    std::vector<Parameter> parameters() override;
    std::string type() const override;
    // l1·Σ|w| + l2·Σw²; backward adds l1·sign(w) + 2·l2·w to dW.
    Scalar penalty() const override;
    // "dense <inputs> <outputs> l1=… l2=…", then one line of weights per input, then the biases.
    void save(std::ostream &out) const override;
    const Regularization &regularization() const noexcept;

    Matrix &weights() noexcept;
    const Matrix &weights() const noexcept;
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
