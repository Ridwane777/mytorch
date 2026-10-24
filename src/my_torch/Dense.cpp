#include "my_torch/Dense.hpp"

#include <cmath>
#include <random>

#include "my_torch/Error.hpp"

namespace my_torch {

Dense::Dense(std::size_t inputs, std::size_t outputs, std::uint32_t seed,
             Regularization regularization)
    : _weights(inputs, outputs), _bias(1, outputs), _weightsGrad(inputs, outputs),
      _biasGrad(1, outputs), _regularization(regularization)
{
    if (!(regularization.l1 >= 0) || !(regularization.l2 >= 0))
        throw Error("dense: regularization coefficients must be non-negative");
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
    if (_regularization.l1 > 0 || _regularization.l2 > 0)
        for (std::size_t i = 0; i < _weights.size(); i++) {
            const Scalar w = _weights.data()[i];
            const Scalar sign = w > 0 ? 1.0 : (w < 0 ? -1.0 : 0.0);
            _weightsGrad.data()[i] += _regularization.l1 * sign + 2 * _regularization.l2 * w;
        }
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

Scalar Dense::penalty() const
{
    Scalar total = 0;

    for (std::size_t i = 0; i < _weights.size(); i++) {
        const Scalar w = _weights.data()[i];
        total += _regularization.l1 * std::abs(w) + _regularization.l2 * w * w;
    }
    return total;
}

const Regularization &Dense::regularization() const noexcept
{
    return _regularization;
}

} // namespace my_torch
