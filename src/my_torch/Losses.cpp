#include "my_torch/Losses.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

#include "my_torch/Error.hpp"

namespace my_torch {

namespace {

// Bounds probabilities before a logarithm or a division.
constexpr Scalar epsilon = 1e-15;

void requireSameShape(const char *loss, const Matrix &prediction, const Matrix &target)
{
    if (prediction.rows() != target.rows() || prediction.cols() != target.cols())
        throw Error(std::string(loss) + ": prediction " + std::to_string(prediction.rows()) + "x" +
                    std::to_string(prediction.cols()) + ", target " +
                    std::to_string(target.rows()) + "x" + std::to_string(target.cols()));
    if (prediction.rows() == 0)
        throw Error(std::string(loss) + ": empty batch");
}

// Sum of f(p, t) over every cell, divided by the batch size.
template <typename F> Scalar batchMean(const Matrix &prediction, const Matrix &target, F f)
{
    Scalar sum = 0;

    for (std::size_t i = 0; i < prediction.size(); i++)
        sum += f(prediction.data()[i], target.data()[i]);
    return sum / static_cast<Scalar>(prediction.rows());
}

// Matrix of f(p, t) / batch size.
template <typename F> Matrix batchGradient(const Matrix &prediction, const Matrix &target, F f)
{
    Matrix result(prediction.rows(), prediction.cols());
    const Scalar batch = static_cast<Scalar>(prediction.rows());

    for (std::size_t i = 0; i < prediction.size(); i++)
        result.data()[i] = f(prediction.data()[i], target.data()[i]) / batch;
    return result;
}

} // namespace

Scalar MSE::value(const Matrix &prediction, const Matrix &target) const
{
    requireSameShape("mse", prediction, target);
    return batchMean(prediction, target, [](Scalar p, Scalar t) { return (p - t) * (p - t); });
}

Matrix MSE::gradient(const Matrix &prediction, const Matrix &target) const
{
    requireSameShape("mse", prediction, target);
    return batchGradient(prediction, target, [](Scalar p, Scalar t) { return 2 * (p - t); });
}

std::string MSE::type() const
{
    return "mse";
}

Scalar MAE::value(const Matrix &prediction, const Matrix &target) const
{
    requireSameShape("mae", prediction, target);
    return batchMean(prediction, target, [](Scalar p, Scalar t) { return std::abs(p - t); });
}

// Subgradient: 0 at p = t, where |x| is not differentiable.
Matrix MAE::gradient(const Matrix &prediction, const Matrix &target) const
{
    requireSameShape("mae", prediction, target);
    return batchGradient(prediction, target,
                         [](Scalar p, Scalar t) { return static_cast<Scalar>((p > t) - (p < t)); });
}

std::string MAE::type() const
{
    return "mae";
}

Huber::Huber(Scalar delta) : _delta(delta)
{
    if (!(delta > 0))
        throw Error("huber: delta must be positive");
}

Scalar Huber::value(const Matrix &prediction, const Matrix &target) const
{
    requireSameShape("huber", prediction, target);
    return batchMean(prediction, target, [this](Scalar p, Scalar t) {
        const Scalar d = std::abs(p - t);
        return d <= _delta ? 0.5 * d * d : _delta * (d - 0.5 * _delta);
    });
}

Matrix Huber::gradient(const Matrix &prediction, const Matrix &target) const
{
    requireSameShape("huber", prediction, target);
    return batchGradient(prediction, target,
                         [this](Scalar p, Scalar t) { return std::clamp(p - t, -_delta, _delta); });
}

std::string Huber::type() const
{
    return "huber";
}

Scalar BinaryCrossEntropy::value(const Matrix &prediction, const Matrix &target) const
{
    requireSameShape("binary_cross_entropy", prediction, target);
    return batchMean(prediction, target, [](Scalar p, Scalar t) {
        p = std::clamp(p, epsilon, 1 - epsilon);
        return -(t * std::log(p) + (1 - t) * std::log(1 - p));
    });
}

Matrix BinaryCrossEntropy::gradient(const Matrix &prediction, const Matrix &target) const
{
    requireSameShape("binary_cross_entropy", prediction, target);
    return batchGradient(prediction, target, [](Scalar p, Scalar t) {
        p = std::clamp(p, epsilon, 1 - epsilon);
        return (p - t) / (p * (1 - p));
    });
}

std::string BinaryCrossEntropy::type() const
{
    return "binary_cross_entropy";
}

CrossEntropy::CrossEntropy(Scalar labelSmoothing, std::vector<Scalar> classWeights)
    : _smoothing(labelSmoothing), _weights(std::move(classWeights))
{
    if (!(labelSmoothing >= 0 && labelSmoothing < 1))
        throw Error("cross_entropy: label smoothing must be in [0, 1[");
    for (Scalar w : _weights)
        if (!(w >= 0))
            throw Error("cross_entropy: class weights must be non-negative");
}

Scalar CrossEntropy::weight(std::size_t cls) const noexcept
{
    return _weights.empty() ? 1.0 : _weights[cls];
}

// Smoothed target q = (1 - s)·t + s / classes.
Matrix CrossEntropy::smoothed(const Matrix &prediction, const Matrix &target) const
{
    requireSameShape("cross_entropy", prediction, target);
    if (!_weights.empty() && _weights.size() != prediction.cols())
        throw Error("cross_entropy: " + std::to_string(_weights.size()) + " class weights for " +
                    std::to_string(prediction.cols()) + " classes");
    const Scalar uniform = _smoothing / static_cast<Scalar>(prediction.cols());

    return target.map([this, uniform](Scalar t) { return (1 - _smoothing) * t + uniform; });
}

Scalar CrossEntropy::value(const Matrix &prediction, const Matrix &target) const
{
    const Matrix q = smoothed(prediction, target);
    Scalar sum = 0;

    for (std::size_t r = 0; r < q.rows(); r++)
        for (std::size_t c = 0; c < q.cols(); c++)
            sum -= weight(c) * q(r, c) * std::log(std::max(prediction(r, c), epsilon));
    return sum / static_cast<Scalar>(q.rows());
}

// L = -Σ_c w_c·q_c·log p_c with p = softmax(z) gives dL/dz_j = p_j·W - w_j·q_j, where W = Σ_c
// w_c·q_c.
Matrix CrossEntropy::gradient(const Matrix &prediction, const Matrix &target) const
{
    const Matrix q = smoothed(prediction, target);
    Matrix result(q.rows(), q.cols());
    const Scalar batch = static_cast<Scalar>(q.rows());

    for (std::size_t r = 0; r < q.rows(); r++) {
        Scalar total = 0;
        for (std::size_t c = 0; c < q.cols(); c++)
            total += weight(c) * q(r, c);
        for (std::size_t c = 0; c < q.cols(); c++)
            result(r, c) = (prediction(r, c) * total - weight(c) * q(r, c)) / batch;
    }
    return result;
}

std::string CrossEntropy::type() const
{
    return "cross_entropy";
}

bool CrossEntropy::fusesSoftmax() const noexcept
{
    return true;
}

} // namespace my_torch
