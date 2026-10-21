#pragma once

#include <string>
#include <vector>

#include "my_torch/Layer.hpp"

namespace my_torch {

class Optimizer {
  public:
    virtual ~Optimizer() = default;

    // Met à jour chaque paramètre à partir de son gradient.
    virtual void step(const std::vector<Parameter> &parameters) = 0;
    virtual void setLearningRate(Scalar rate) = 0;
    virtual Scalar learningRate() const = 0;
    virtual std::string type() const = 0;
};

} // namespace my_torch
