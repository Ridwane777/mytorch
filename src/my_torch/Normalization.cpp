#include "my_torch/Normalization.hpp"

#include <cmath>
#include <string>

#include "my_torch/Checkpoint.hpp"
#include "my_torch/Error.hpp"

namespace my_torch {

namespace {

void checkWidth(const char *layer, const Matrix &m, std::size_t features)
{
    if (m.cols() != features)
        throw Error(std::string(layer) + ": expected " + std::to_string(features) +
                    " features, got " + std::to_string(m.cols()));
}

// y = gamma·x̂ + beta, row by row.
Matrix scaleAndShift(const Matrix &normalized, const Matrix &gamma, const Matrix &beta)
{
    Matrix out(normalized.rows(), normalized.cols());
    for (std::size_t i = 0; i < out.rows(); i++)
        for (std::size_t j = 0; j < out.cols(); j++)
            out(i, j) = gamma(0, j) * normalized(i, j) + beta(0, j);
    return out;
}

// dgamma += Σ g·x̂ and dbeta += Σ g over the rows; returns d̂ = g·gamma, the gradient of x̂.
Matrix parameterGradients(const Matrix &grad, const Matrix &normalized, const Matrix &gamma,
                          Matrix &gammaGrad, Matrix &betaGrad)
{
    Matrix dNormalized(grad.rows(), grad.cols());
    for (std::size_t i = 0; i < grad.rows(); i++)
        for (std::size_t j = 0; j < grad.cols(); j++) {
            gammaGrad(0, j) += grad(i, j) * normalized(i, j);
            betaGrad(0, j) += grad(i, j);
            dNormalized(i, j) = grad(i, j) * gamma(0, j);
        }
    return dNormalized;
}

} // namespace

BatchNorm::BatchNorm(std::size_t features, Scalar momentum, Scalar epsilon)
    : _momentum(momentum), _epsilon(epsilon), _gamma(1, features, 1), _beta(1, features),
      _gammaGrad(1, features), _betaGrad(1, features), _runningMean(1, features),
      _runningVariance(1, features, 1)
{
    if (features == 0)
        throw Error("batch_norm: the number of features must be positive");
    if (!(momentum > 0 && momentum <= 1))
        throw Error("batch_norm: momentum must be in ]0, 1]");
    if (!(epsilon > 0))
        throw Error("batch_norm: epsilon must be positive");
}

Matrix BatchNorm::forward(const Matrix &input, bool training)
{
    checkWidth("batch_norm", input, features());
    const std::size_t m = input.rows();
    const std::size_t n = input.cols();
    Matrix normalized(m, n);

    if (!training) {
        _normalized = Matrix();
        _inverseStd = Matrix(1, n);
        for (std::size_t j = 0; j < n; j++) {
            const Scalar inverse = 1 / std::sqrt(_runningVariance(0, j) + _epsilon);
            _inverseStd(0, j) = inverse;
            for (std::size_t i = 0; i < m; i++)
                normalized(i, j) = (input(i, j) - _runningMean(0, j)) * inverse;
        }
        return scaleAndShift(normalized, _gamma, _beta);
    }
    _inverseStd = Matrix(1, n);
    for (std::size_t j = 0; j < n; j++) {
        Scalar mean = 0;
        for (std::size_t i = 0; i < m; i++)
            mean += input(i, j);
        mean /= static_cast<Scalar>(m);
        Scalar squares = 0;
        for (std::size_t i = 0; i < m; i++)
            squares += (input(i, j) - mean) * (input(i, j) - mean);
        const Scalar variance = squares / static_cast<Scalar>(m);
        _inverseStd(0, j) = 1 / std::sqrt(variance + _epsilon);
        for (std::size_t i = 0; i < m; i++)
            normalized(i, j) = (input(i, j) - mean) * _inverseStd(0, j);
        if (m > 1) {
            const Scalar unbiased = squares / static_cast<Scalar>(m - 1);
            _runningMean(0, j) = (1 - _momentum) * _runningMean(0, j) + _momentum * mean;
            _runningVariance(0, j) =
                (1 - _momentum) * _runningVariance(0, j) + _momentum * unbiased;
        }
    }
    _normalized = normalized;
    return scaleAndShift(normalized, _gamma, _beta);
}

Matrix BatchNorm::backward(const Matrix &gradOutput)
{
    if (_normalized.size() == 0) {
        // After a prediction forward the statistics are constants: dx = g·gamma / σ_running. Used
        // for input gradients (saliency, ADR-0035); the parameters get no gradient.
        if (_inverseStd.cols() != gradOutput.cols())
            throw Error("batch_norm: backward needs a forward first");
        Matrix grad(gradOutput.rows(), gradOutput.cols());
        for (std::size_t i = 0; i < grad.rows(); i++)
            for (std::size_t j = 0; j < grad.cols(); j++)
                grad(i, j) = gradOutput(i, j) * _gamma(0, j) * _inverseStd(0, j);
        return grad;
    }
    if (gradOutput.rows() != _normalized.rows() || gradOutput.cols() != _normalized.cols())
        throw Error("batch_norm: the gradient does not match the last forward");
    const Matrix d = parameterGradients(gradOutput, _normalized, _gamma, _gammaGrad, _betaGrad);
    const std::size_t m = d.rows();
    const Scalar count = static_cast<Scalar>(m);
    Matrix grad(m, d.cols());

    // dx = (1 / (m·σ))·(m·d̂ - Σ d̂ - x̂·Σ d̂·x̂), the sums over the batch, per column.
    for (std::size_t j = 0; j < d.cols(); j++) {
        Scalar sum = 0;
        Scalar dot = 0;
        for (std::size_t i = 0; i < m; i++) {
            sum += d(i, j);
            dot += d(i, j) * _normalized(i, j);
        }
        for (std::size_t i = 0; i < m; i++)
            grad(i, j) =
                _inverseStd(0, j) / count * (count * d(i, j) - sum - _normalized(i, j) * dot);
    }
    return grad;
}

std::vector<Parameter> BatchNorm::parameters()
{
    return {{&_gamma, &_gammaGrad}, {&_beta, &_betaGrad}};
}

std::string BatchNorm::type() const
{
    return "batch_norm";
}

void BatchNorm::save(std::ostream &out) const
{
    out << "batch_norm " << _gamma.cols() << " momentum=" << _momentum << " epsilon=" << _epsilon
        << '\n';
    for (const Matrix *row : {&_gamma, &_beta, &_runningMean, &_runningVariance})
        for (std::size_t j = 0; j < row->cols(); j++)
            out << (*row)(0, j) << (j + 1 < row->cols() ? ' ' : '\n');
}

void BatchNorm::saveState(std::ostream &out) const
{
    writeMatrices(out, {_runningMean, _runningVariance});
}

void BatchNorm::loadState(std::istream &in)
{
    std::vector<Matrix> running = readMatrices(in);
    if (running.size() != 2 || running[0].rows() != 1 || running[0].cols() != features() ||
        running[1].rows() != 1 || running[1].cols() != features())
        throw Error("batch_norm: the saved running averages do not match the layer");
    _runningMean = running[0];
    _runningVariance = running[1];
}

std::size_t BatchNorm::features() const noexcept
{
    return _gamma.cols();
}

Scalar BatchNorm::momentum() const noexcept
{
    return _momentum;
}

Scalar BatchNorm::epsilon() const noexcept
{
    return _epsilon;
}

Matrix &BatchNorm::gamma() noexcept
{
    return _gamma;
}

Matrix &BatchNorm::beta() noexcept
{
    return _beta;
}

Matrix &BatchNorm::runningMean() noexcept
{
    return _runningMean;
}

Matrix &BatchNorm::runningVariance() noexcept
{
    return _runningVariance;
}

LayerNorm::LayerNorm(std::size_t features, Scalar epsilon)
    : _epsilon(epsilon), _gamma(1, features, 1), _beta(1, features), _gammaGrad(1, features),
      _betaGrad(1, features)
{
    if (features == 0)
        throw Error("layer_norm: the number of features must be positive");
    if (!(epsilon > 0))
        throw Error("layer_norm: epsilon must be positive");
}

Matrix LayerNorm::forward(const Matrix &input, bool)
{
    checkWidth("layer_norm", input, features());
    const std::size_t m = input.rows();
    const std::size_t n = input.cols();
    _normalized = Matrix(m, n);
    _inverseStd = Matrix(m, 1);

    for (std::size_t i = 0; i < m; i++) {
        Scalar mean = 0;
        for (std::size_t j = 0; j < n; j++)
            mean += input(i, j);
        mean /= static_cast<Scalar>(n);
        Scalar squares = 0;
        for (std::size_t j = 0; j < n; j++)
            squares += (input(i, j) - mean) * (input(i, j) - mean);
        _inverseStd(i, 0) = 1 / std::sqrt(squares / static_cast<Scalar>(n) + _epsilon);
        for (std::size_t j = 0; j < n; j++)
            _normalized(i, j) = (input(i, j) - mean) * _inverseStd(i, 0);
    }
    return scaleAndShift(_normalized, _gamma, _beta);
}

Matrix LayerNorm::backward(const Matrix &gradOutput)
{
    if (gradOutput.rows() != _normalized.rows() || gradOutput.cols() != _normalized.cols())
        throw Error("layer_norm: the gradient does not match the last forward");
    const Matrix d = parameterGradients(gradOutput, _normalized, _gamma, _gammaGrad, _betaGrad);
    const std::size_t n = d.cols();
    const Scalar count = static_cast<Scalar>(n);
    Matrix grad(d.rows(), n);

    // The same formula as BatchNorm, the sums over the features of each row.
    for (std::size_t i = 0; i < d.rows(); i++) {
        Scalar sum = 0;
        Scalar dot = 0;
        for (std::size_t j = 0; j < n; j++) {
            sum += d(i, j);
            dot += d(i, j) * _normalized(i, j);
        }
        for (std::size_t j = 0; j < n; j++)
            grad(i, j) =
                _inverseStd(i, 0) / count * (count * d(i, j) - sum - _normalized(i, j) * dot);
    }
    return grad;
}

std::vector<Parameter> LayerNorm::parameters()
{
    return {{&_gamma, &_gammaGrad}, {&_beta, &_betaGrad}};
}

std::string LayerNorm::type() const
{
    return "layer_norm";
}

void LayerNorm::save(std::ostream &out) const
{
    out << "layer_norm " << _gamma.cols() << " epsilon=" << _epsilon << '\n';
    for (const Matrix *row : {&_gamma, &_beta})
        for (std::size_t j = 0; j < row->cols(); j++)
            out << (*row)(0, j) << (j + 1 < row->cols() ? ' ' : '\n');
}

std::size_t LayerNorm::features() const noexcept
{
    return _gamma.cols();
}

Scalar LayerNorm::epsilon() const noexcept
{
    return _epsilon;
}

Matrix &LayerNorm::gamma() noexcept
{
    return _gamma;
}

Matrix &LayerNorm::beta() noexcept
{
    return _beta;
}

const Matrix &LayerNorm::gamma() const noexcept
{
    return _gamma;
}

const Matrix &LayerNorm::beta() const noexcept
{
    return _beta;
}

} // namespace my_torch
