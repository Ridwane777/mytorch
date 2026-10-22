#pragma once

#include <cstddef>
#include <cstdint>

#include "my_torch/Layer.hpp"

namespace my_torch {

// Couche entièrement connectée : Z = X·W + b, avec W (entrées × neurones) et un biais par neurone,
// b (1 × neurones), partagé par toutes les lignes du batch.
class Dense : public Layer {
  public:
    // Initialisation provisoire uniforme dans ±sqrt(1 / entrées) ; Xavier et He arrivent avec #6.
    Dense(std::size_t inputs, std::size_t outputs, std::uint32_t seed = 42);

    Matrix forward(const Matrix &input, bool training) override;
    // dW += Xᵀ·G, db += somme des lignes de G, renvoie dX = G·Wᵀ. Les gradients s'accumulent :
    // c'est à la boucle d'entraînement de les remettre à zéro avant chaque batch.
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
