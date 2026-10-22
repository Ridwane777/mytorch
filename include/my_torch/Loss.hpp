#pragma once

#include <string>

#include "my_torch/Matrix.hpp"

namespace my_torch {

class Loss {
  public:
    virtual ~Loss() = default;

    // Coût moyen sur le batch.
    virtual Scalar value(const Matrix &prediction, const Matrix &target) const = 0;
    // dL/dprédiction, de la même forme que prediction.
    virtual Matrix gradient(const Matrix &prediction, const Matrix &target) const = 0;
    virtual std::string type() const = 0;
};

} // namespace my_torch
