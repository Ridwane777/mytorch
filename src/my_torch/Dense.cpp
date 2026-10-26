#include "my_torch/Dense.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "my_torch/Error.hpp"

namespace my_torch {

namespace {

// A column is redrawn when its norm after projection is below this: it was (almost) dependent.
constexpr Scalar degenerateNorm = 1e-12;

// Fills W with an orthogonal matrix: a Gaussian a × b matrix (a = max, b = min of the two sizes)
// whose columns are orthonormalised by modified Gram-Schmidt, transposed when inputs < outputs.
void fillOrthogonal(Matrix &weights, std::mt19937 &rng)
{
    const std::size_t rows = weights.rows();
    const std::size_t cols = weights.cols();
    const std::size_t a = std::max(rows, cols);
    const std::size_t b = std::min(rows, cols);
    std::normal_distribution<Scalar> gauss(0, 1);
    // q[j] is column j, of length a.
    std::vector<std::vector<Scalar>> q(b, std::vector<Scalar>(a));

    for (std::size_t j = 0; j < b; j++) {
        for (;;) {
            for (Scalar &v : q[j])
                v = gauss(rng);
            for (std::size_t k = 0; k < j; k++) {
                Scalar dot = 0;
                for (std::size_t i = 0; i < a; i++)
                    dot += q[k][i] * q[j][i];
                for (std::size_t i = 0; i < a; i++)
                    q[j][i] -= dot * q[k][i];
            }
            Scalar norm = 0;
            for (Scalar v : q[j])
                norm += v * v;
            norm = std::sqrt(norm);
            if (norm < degenerateNorm)
                continue;
            for (Scalar &v : q[j])
                v /= norm;
            break;
        }
    }
    for (std::size_t r = 0; r < rows; r++)
        for (std::size_t c = 0; c < cols; c++)
            weights(r, c) = rows >= cols ? q[c][r] : q[r][c];
}

// Distribution of one weight: N(0, spread²) when `normal`, else U(±spread).
struct Law {
    bool normal;
    Scalar spread;
};

Law lawOf(Init init, Scalar fanIn, Scalar fanOut)
{
    const Scalar lecun = 1.0 / fanIn;
    const Scalar xavier = 2.0 / (fanIn + fanOut);
    const Scalar he = 2.0 / fanIn;

    switch (init) {
    case Init::LecunUniform:
        return {false, std::sqrt(3 * lecun)};
    case Init::LecunNormal:
        return {true, std::sqrt(lecun)};
    case Init::XavierUniform:
        return {false, std::sqrt(3 * xavier)};
    case Init::XavierNormal:
        return {true, std::sqrt(xavier)};
    case Init::HeUniform:
        return {false, std::sqrt(3 * he)};
    case Init::HeNormal:
        return {true, std::sqrt(he)};
    case Init::Uniform:
    case Init::Orthogonal:
        break;
    }
    // The original default, kept draw for draw: U(±sqrt(1 / fan_in)).
    return {false, std::sqrt(lecun)};
}

void fillRandom(Matrix &weights, Init init, std::mt19937 &rng)
{
    if (init == Init::Orthogonal)
        return fillOrthogonal(weights, rng);
    const Law law =
        lawOf(init, static_cast<Scalar>(weights.rows()), static_cast<Scalar>(weights.cols()));

    if (law.normal) {
        std::normal_distribution<Scalar> dist(0, law.spread);
        for (std::size_t i = 0; i < weights.size(); i++)
            weights.data()[i] = dist(rng);
    } else {
        std::uniform_real_distribution<Scalar> dist(-law.spread, law.spread);
        for (std::size_t i = 0; i < weights.size(); i++)
            weights.data()[i] = dist(rng);
    }
}

} // namespace

Dense::Dense(std::size_t inputs, std::size_t outputs, std::uint32_t seed,
             Regularization regularization, Init init)
    : _weights(inputs, outputs), _bias(1, outputs), _weightsGrad(inputs, outputs),
      _biasGrad(1, outputs), _regularization(regularization)
{
    if (!(regularization.l1 >= 0) || !(regularization.l2 >= 0))
        throw Error("dense: regularization coefficients must be non-negative");
    std::mt19937 rng(seed);

    fillRandom(_weights, init, rng);
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

const Matrix &Dense::weights() const noexcept
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

void Dense::save(std::ostream &out) const
{
    out << "dense " << _weights.rows() << ' ' << _weights.cols() << " l1=" << _regularization.l1
        << " l2=" << _regularization.l2 << '\n';
    for (std::size_t r = 0; r < _weights.rows(); r++)
        for (std::size_t c = 0; c < _weights.cols(); c++)
            out << _weights(r, c) << (c + 1 < _weights.cols() ? ' ' : '\n');
    for (std::size_t c = 0; c < _bias.cols(); c++)
        out << _bias(0, c) << (c + 1 < _bias.cols() ? ' ' : '\n');
}

} // namespace my_torch
