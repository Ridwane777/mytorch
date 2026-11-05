#pragma once

#include <cstddef>
#include <istream>
#include <ostream>
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
    // Regularization term added to the training objective (L1/L2 on the weights, ADR-0010). The
    // layer adds its derivative to its own gradients in backward; fit reports it per epoch.
    virtual Scalar penalty() const
    {
        return 0;
    }
    // Writes the layer in the network file (ADR-0012): its line ("type key=value…") and, for layers
    // with parameters, the numbers that follow. Parameter-free layers keep this default.
    virtual void save(std::ostream &out) const
    {
        out << type() << '\n';
    }
    // Training state other than the parameters, for checkpoints (ADR-0032): the generator of a
    // dropout layer. Parameter-free, stateless layers keep these defaults. loadState reads what
    // saveState wrote and throws my_torch::Error on malformed text.
    virtual void saveState(std::ostream &) const
    {
    }
    virtual void loadState(std::istream &)
    {
    }
    // Width of the output rows for input rows of `inputWidth` columns (ADR-0037). The network file
    // and modelSummary follow the widths through the layers with it. Layers that keep the width
    // (activations, dropout, normalizations) keep this default; a layer that needs a fixed input
    // width (dense, convolution, pooling) throws my_torch::Error ("expects N inputs but receives
    // M") when it differs.
    virtual std::size_t outputWidth(std::size_t inputWidth) const
    {
        return inputWidth;
    }
    // Stable name written in the network file (#8), for example "dense" or "relu".
    virtual std::string type() const = 0;

    // The layers a container layer holds (Residual, Concat, ADR-0046), in order; empty for every
    // other layer. Code that must see every layer walks them.
    virtual std::vector<Layer *> children() const
    {
        return {};
    }
    // Forward with only the dropout layers in training mode (Monte Carlo dropout, ADR-0034):
    // prediction mode for every other layer. Dropout and the containers override it.
    virtual Matrix forwardMonteCarlo(const Matrix &input)
    {
        return forward(input, false);
    }
};

} // namespace my_torch
