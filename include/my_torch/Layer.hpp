#pragma once

#include <string>
#include <vector>

#include "my_torch/Matrix.hpp"

namespace my_torch {

// A trainable parameter and its gradient, exposed to the optimizer.
struct Parameter {
    Matrix *value;
    Matrix *grad;
};

// Any layer: dense, activation, dropout, convolution... Activations are parameter-free
// layers, which keeps a network simple to stack and to serialize.
class Layer {
  public:
    virtual ~Layer() = default;

    // training = false disables dropout and freezes the normalization statistics.
    virtual Matrix forward(const Matrix &input, bool training) = 0;
    // Receives dL/doutput, accumulates the parameter gradients, returns dL/dinput.
    virtual Matrix backward(const Matrix &gradOutput) = 0;

    virtual std::vector<Parameter> parameters()
    {
        return {};
    }
    // Stable name written in the network file (#8), for example "dense" or "relu".
    virtual std::string type() const = 0;
};

} // namespace my_torch
