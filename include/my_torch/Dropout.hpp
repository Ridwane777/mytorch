#pragma once

#include <cstdint>
#include <random>

#include "my_torch/Layer.hpp"

namespace my_torch {

// Inverted dropout (ADR-0010). In training, each value is zeroed with probability rate and the
// others are divided by 1 - rate, so the mean is unchanged; in prediction the layer is the
// identity. backward reuses the mask of the last training forward.
class Dropout : public Layer {
  public:
    // Throws my_torch::Error unless 0 <= rate < 1.
    explicit Dropout(Scalar rate, std::uint64_t seed = 42);

    Matrix forward(const Matrix &input, bool training) override;
    Matrix backward(const Matrix &gradOutput) override;
    std::string type() const override;
    Scalar rate() const noexcept;

  private:
    Scalar _rate;
    std::mt19937_64 _rng;
    Matrix _mask; // 0 or 1 / (1 - rate) per value; empty after a prediction forward
};

} // namespace my_torch
