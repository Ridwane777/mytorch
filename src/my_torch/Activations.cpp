#include "my_torch/Activations.hpp"

#include <algorithm>
#include <cmath>

namespace my_torch {

Matrix ReLU::forward(const Matrix &input, bool /*training*/)
{
    _input = input;
    return input.map([](Scalar z) { return std::max(z, Scalar{0}); });
}

Matrix ReLU::backward(const Matrix &gradOutput)
{
    return gradOutput.hadamard(_input.map([](Scalar z) { return z > 0 ? 1.0 : 0.0; }));
}

std::string ReLU::type() const
{
    return "relu";
}

LeakyReLU::LeakyReLU(Scalar slope) : _slope(slope)
{
}

Matrix LeakyReLU::forward(const Matrix &input, bool /*training*/)
{
    _input = input;
    return input.map([this](Scalar z) { return z > 0 ? z : _slope * z; });
}

Matrix LeakyReLU::backward(const Matrix &gradOutput)
{
    return gradOutput.hadamard(_input.map([this](Scalar z) { return z > 0 ? 1.0 : _slope; }));
}

std::string LeakyReLU::type() const
{
    return "leaky_relu";
}

Scalar LeakyReLU::slope() const noexcept
{
    return _slope;
}

Matrix Sigmoid::forward(const Matrix &input, bool /*training*/)
{
    _output = input.map([](Scalar z) { return 1.0 / (1.0 + std::exp(-z)); });
    return _output;
}

Matrix Sigmoid::backward(const Matrix &gradOutput)
{
    return gradOutput.hadamard(_output.map([](Scalar s) { return s * (1.0 - s); }));
}

std::string Sigmoid::type() const
{
    return "sigmoid";
}

Matrix Tanh::forward(const Matrix &input, bool /*training*/)
{
    _output = input.map([](Scalar z) { return std::tanh(z); });
    return _output;
}

Matrix Tanh::backward(const Matrix &gradOutput)
{
    return gradOutput.hadamard(_output.map([](Scalar t) { return 1.0 - t * t; }));
}

std::string Tanh::type() const
{
    return "tanh";
}

Matrix Softmax::forward(const Matrix &input, bool /*training*/)
{
    _output = Matrix(input.rows(), input.cols());
    for (std::size_t r = 0; r < input.rows(); r++) {
        Scalar max = input(r, 0);
        for (std::size_t c = 1; c < input.cols(); c++)
            max = std::max(max, input(r, c));
        Scalar sum = 0;
        for (std::size_t c = 0; c < input.cols(); c++) {
            _output(r, c) = std::exp(input(r, c) - max);
            sum += _output(r, c);
        }
        for (std::size_t c = 0; c < input.cols(); c++)
            _output(r, c) /= sum;
    }
    return _output;
}

Matrix Softmax::backward(const Matrix &gradOutput)
{
    Matrix result(gradOutput.rows(), gradOutput.cols());

    for (std::size_t r = 0; r < gradOutput.rows(); r++) {
        Scalar dot = 0;
        for (std::size_t c = 0; c < gradOutput.cols(); c++)
            dot += _output(r, c) * gradOutput(r, c);
        for (std::size_t c = 0; c < gradOutput.cols(); c++)
            result(r, c) = _output(r, c) * (gradOutput(r, c) - dot);
    }
    return result;
}

std::string Softmax::type() const
{
    return "softmax";
}

} // namespace my_torch
