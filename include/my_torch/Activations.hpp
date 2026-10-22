#pragma once

#include "my_torch/Layer.hpp"

namespace my_torch {

// Activations : couches sans paramètre appliquées après une Dense. Sans elles, empiler des
// couches Dense revient à une seule couche linéaire.

// max(0, z). Dérivée 1 si z > 0, sinon 0.
class ReLU : public Layer {
  public:
    Matrix forward(const Matrix &input, bool training) override;
    Matrix backward(const Matrix &gradOutput) override;
    std::string type() const override;

  private:
    Matrix _input;
};

// z si z > 0, sinon slope·z : laisse passer un peu de gradient pour éviter les neurones morts.
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

// 1 / (1 + e^-z), dans ]0, 1[. Dérivée s·(1 - s), au plus 0.25.
class Sigmoid : public Layer {
  public:
    Matrix forward(const Matrix &input, bool training) override;
    Matrix backward(const Matrix &gradOutput) override;
    std::string type() const override;

  private:
    Matrix _output;
};

// Dans ]-1, 1[, centrée sur 0. Dérivée 1 - t².
class Tanh : public Layer {
  public:
    Matrix forward(const Matrix &input, bool training) override;
    Matrix backward(const Matrix &gradOutput) override;
    std::string type() const override;

  private:
    Matrix _output;
};

// Probabilités par ligne : e^z_i / Σ e^z_j, calculé après soustraction du maximum de la ligne
// pour ne jamais déborder. Backward : dZ_i = p_i·(G_i - Σ_j p_j·G_j).
class Softmax : public Layer {
  public:
    Matrix forward(const Matrix &input, bool training) override;
    Matrix backward(const Matrix &gradOutput) override;
    std::string type() const override;

  private:
    Matrix _output;
};

} // namespace my_torch
