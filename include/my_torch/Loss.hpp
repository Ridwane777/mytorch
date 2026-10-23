#pragma once

#include <string>

#include "my_torch/Matrix.hpp"

namespace my_torch {

class Loss {
  public:
    virtual ~Loss() = default;

    // Coût moyen sur le batch.
    virtual Scalar value(const Matrix &prediction, const Matrix &target) const = 0;
    // dL/dprédiction, de la même forme que prediction. Si fusesSoftmax() est vrai, c'est dL/dz,
    // où z est l'entrée de la couche Softmax finale (voir ci-dessous).
    virtual Matrix gradient(const Matrix &prediction, const Matrix &target) const = 0;
    virtual std::string type() const = 0;

    // Vrai pour l'entropie croisée couplée à Softmax (ADR-0007) : prediction est la sortie du
    // Softmax, mais gradient() renvoie directement le gradient par rapport à son entrée. La
    // rétropropagation doit alors sauter le backward de cette dernière couche Softmax.
    virtual bool fusesSoftmax() const noexcept
    {
        return false;
    }
};

} // namespace my_torch
