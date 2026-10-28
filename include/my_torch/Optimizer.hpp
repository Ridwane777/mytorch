#pragma once

#include <istream>
#include <ostream>
#include <string>
#include <vector>

#include "my_torch/Layer.hpp"

namespace my_torch {

class Optimizer {
  public:
    virtual ~Optimizer() = default;

    // Updates every parameter from its gradient.
    virtual void step(const std::vector<Parameter> &parameters) = 0;
    virtual void setLearningRate(Scalar rate) = 0;
    virtual Scalar learningRate() const = 0;
    virtual std::string type() const = 0;
    // The optimizer's memory (velocity, running averages, step count), for checkpoints
    // (ADR-0032). loadState reads what saveState wrote and throws my_torch::Error on malformed
    // text; the shapes are checked against the parameters at the next step.
    virtual void saveState(std::ostream &out) const = 0;
    virtual void loadState(std::istream &in) = 0;
};

} // namespace my_torch
