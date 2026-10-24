#pragma once

#include "my_torch/Layer.hpp"

namespace my_torch {

// Activations: parameter-free layers applied after a Dense. Without them, stacking Dense
// layers amounts to a single linear layer.

// max(0, z). Derivative 1 if z > 0, otherwise 0.
class ReLU : public Layer {
  public:
    Matrix forward(const Matrix &input, bool training) override;
    Matrix backward(const Matrix &gradOutput) override;
    std::string type() const override;

  private:
    Matrix _input;
};

// z if z > 0, otherwise slope·z: lets a little gradient through to avoid dead neurons.
class LeakyReLU : public Layer {
  public:
    explicit LeakyReLU(Scalar slope = 0.01);

    Matrix forward(const Matrix &input, bool training) override;
    Matrix backward(const Matrix &gradOutput) override;
    std::string type() const override;
    Scalar slope() const noexcept;

  private:
    Scalar _slope;
    Matrix _input;
};

// 1 / (1 + e^-z), in ]0, 1[. Derivative s·(1 - s), at most 0.25.
class Sigmoid : public Layer {
  public:
    Matrix forward(const Matrix &input, bool training) override;
    Matrix backward(const Matrix &gradOutput) override;
    std::string type() const override;

  private:
    Matrix _output;
};

// In ]-1, 1[, centred on 0. Derivative 1 - t².
class Tanh : public Layer {
  public:
    Matrix forward(const Matrix &input, bool training) override;
    Matrix backward(const Matrix &gradOutput) override;
    std::string type() const override;

  private:
    Matrix _output;
};

// Probabilities per row: e^z_i / Σ e^z_j, computed after subtracting the row maximum
// so it never overflows. Backward : dZ_i = p_i·(G_i - Σ_j p_j·G_j).
class Softmax : public Layer {
  public:
    Matrix forward(const Matrix &input, bool training) override;
    Matrix backward(const Matrix &gradOutput) override;
    std::string type() const override;

  private:
    Matrix _output;
};

} // namespace my_torch
