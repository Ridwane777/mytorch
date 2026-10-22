#pragma once

#include <string>
#include <vector>

#include "my_torch/Matrix.hpp"

namespace my_torch {

// Un paramètre entraînable et son gradient, exposés à l'optimiseur.
struct Parameter {
    Matrix *value;
    Matrix *grad;
};

// Toute couche : dense, activation, dropout, convolution… Les activations sont des couches
// sans paramètre, ce qui garde un réseau simple à empiler et à sérialiser.
class Layer {
  public:
    virtual ~Layer() = default;

    // training = false désactive le dropout et fige les statistiques de normalisation.
    virtual Matrix forward(const Matrix &input, bool training) = 0;
    // Reçoit dL/dsortie, accumule les gradients des paramètres, renvoie dL/dentrée.
    virtual Matrix backward(const Matrix &gradOutput) = 0;

    virtual std::vector<Parameter> parameters()
    {
        return {};
    }
    // Nom stable écrit dans le fichier du réseau (#8), par exemple "dense" ou "relu".
    virtual std::string type() const = 0;
};

} // namespace my_torch
