#pragma once

#include <string>

#include "my_torch/Matrix.hpp"

namespace my_torch {

class Loss {
  public:
    virtual ~Loss() = default;

    // Mean cost over the batch.
    virtual Scalar value(const Matrix &prediction, const Matrix &target) const = 0;
    // dL/dprediction, same shape as prediction. If fusesSoftmax() is true, this is dL/dz instead,
    // where z is the input of the final Softmax layer (see below).
    virtual Matrix gradient(const Matrix &prediction, const Matrix &target) const = 0;
    virtual std::string type() const = 0;

    // True for the cross-entropy coupled to Softmax (ADR-0007): prediction is the Softmax output,
    // but gradient() returns the gradient with respect to its input directly. Backpropagation must
    // then skip the backward of that final Softmax layer, and must throw my_torch::Error if the
    // last layer is not a Softmax.
    virtual bool fusesSoftmax() const noexcept
    {
        return false;
    }
};

} // namespace my_torch
