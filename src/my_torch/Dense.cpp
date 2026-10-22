#include "my_torch/Dense.hpp"

#include <cmath>
#include <random>

namespace my_torch {

Dense::Dense(std::size_t inputs, std::size_t outputs, std::uint32_t seed)
    : _weights(inputs, outputs), _bias(1, outputs), _weightsGrad(inputs, outputs),
      _biasGrad(1, outputs)
{
    const Scalar limit = std::sqrt(1.0 / static_cast<Scalar>(inputs));
    std::mt19937 rng(seed);
    std::uniform_real_distribution<Scalar> dist(-limit, limit);

    for (std::size_t i = 0; i < _weights.size(); i++)
        _weights.data()[i] = dist(rng);
}

Matrix Dense::forward(const Matrix &input, bool /*training*/)
{
    _input = input;
    return input.matmul(_weights).addRowVector(_bias);
}

Matrix Dense::backward(const Matrix &gradOutput)
{
    _weightsGrad += _input.transpose().matmul(gradOutput);
    _biasGrad += gradOutput.sumRows();
    return gradOutput.matmul(_weights.transpose());
}

std::vector<Parameter> Dense::parameters()
{
    return {{&_weights, &_weightsGrad}, {&_bias, &_biasGrad}};
}

std::string Dense::type() const
{
    return "dense";
}

Matrix &Dense::weights() noexcept
{
    return _weights;
}

Matrix &Dense::bias() noexcept
{
    return _bias;
}

} // namespace my_torch
