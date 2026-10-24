#include "my_torch/Optimizers.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include "my_torch/Error.hpp"

namespace my_torch {

namespace {

void requirePositive(const char *optimizer, const char *name, Scalar value)
{
    if (!(value > 0))
        throw Error(std::string(optimizer) + ": " + name + " must be positive");
}

void requireNonNegative(const char *optimizer, const char *name, Scalar value)
{
    if (!(value >= 0))
        throw Error(std::string(optimizer) + ": " + name + " must be non-negative");
}

void requireInUnit(const char *optimizer, const char *name, Scalar value)
{
    if (!(value >= 0 && value < 1))
        throw Error(std::string(optimizer) + ": " + name + " must be in [0, 1[");
}

void requireSameShape(const char *optimizer, const Matrix &a, const Matrix &b)
{
    if (a.rows() != b.rows() || a.cols() != b.cols())
        throw Error(std::string(optimizer) +
                    ": parameter and its state or gradient differ in shape");
}

// Creates the optimizer's memory on the first step, then checks it still matches the parameters
// received (same count, same shapes, same order) and that each gradient has its parameter's shape.
void prepare(const char *optimizer, std::vector<Matrix> &state,
             const std::vector<Parameter> &params)
{
    if (state.empty())
        for (const Parameter &p : params)
            state.emplace_back(p.value->rows(), p.value->cols());
    if (state.size() != params.size())
        throw Error(std::string(optimizer) + ": the number of parameters changed between steps");
    for (std::size_t i = 0; i < params.size(); i++) {
        requireSameShape(optimizer, *params[i].value, state[i]);
        requireSameShape(optimizer, *params[i].value, *params[i].grad);
    }
}

} // namespace

SGD::SGD(Scalar learningRate, Scalar momentum, bool nesterov)
    : _rate(learningRate), _momentum(momentum), _nesterov(nesterov)
{
    requirePositive("sgd", "learning rate", learningRate);
    requireInUnit("sgd", "momentum", momentum);
    if (nesterov && momentum == 0)
        throw Error("sgd: nesterov needs a momentum above 0");
}

void SGD::step(const std::vector<Parameter> &parameters)
{
    prepare("sgd", _velocity, parameters);
    for (std::size_t i = 0; i < parameters.size(); i++) {
        Scalar *w = parameters[i].value->data();
        const Scalar *g = parameters[i].grad->data();
        Scalar *v = _velocity[i].data();

        for (std::size_t k = 0; k < parameters[i].value->size(); k++) {
            v[k] = _momentum * v[k] + g[k];
            w[k] -= _rate * (_nesterov ? g[k] + _momentum * v[k] : v[k]);
        }
    }
}

void SGD::setLearningRate(Scalar rate)
{
    requireNonNegative("sgd", "learning rate", rate);
    _rate = rate;
}

Scalar SGD::learningRate() const
{
    return _rate;
}

std::string SGD::type() const
{
    return "sgd";
}

Adam::Adam(Scalar learningRate, Scalar beta1, Scalar beta2, Scalar epsilon, bool amsgrad)
    : Adam(learningRate, 0, beta1, beta2, epsilon, amsgrad, "adam")
{
}

Adam::Adam(Scalar learningRate, Scalar weightDecay, Scalar beta1, Scalar beta2, Scalar epsilon,
           bool amsgrad, const char *name)
    : _rate(learningRate), _weightDecay(weightDecay), _beta1(beta1), _beta2(beta2),
      _epsilon(epsilon), _amsgrad(amsgrad), _name(name)
{
    requirePositive(name, "learning rate", learningRate);
    requirePositive(name, "epsilon", epsilon);
    requireInUnit(name, "beta1", beta1);
    requireInUnit(name, "beta2", beta2);
    if (!(weightDecay >= 0))
        throw Error(std::string(name) + ": weight decay must be non-negative");
}

void Adam::step(const std::vector<Parameter> &parameters)
{
    prepare(_name, _m, parameters);
    prepare(_name, _v, parameters);
    if (_amsgrad)
        prepare(_name, _vmax, parameters);
    _steps++;
    // Bias correction: m and v start at 0 and underestimate the first averages.
    const Scalar correction1 = 1 - std::pow(_beta1, static_cast<Scalar>(_steps));
    const Scalar correction2 = 1 - std::pow(_beta2, static_cast<Scalar>(_steps));

    for (std::size_t i = 0; i < parameters.size(); i++) {
        Scalar *w = parameters[i].value->data();
        const Scalar *g = parameters[i].grad->data();
        Scalar *m = _m[i].data();
        Scalar *v = _v[i].data();

        for (std::size_t k = 0; k < parameters[i].value->size(); k++) {
            m[k] = _beta1 * m[k] + (1 - _beta1) * g[k];
            v[k] = _beta2 * v[k] + (1 - _beta2) * g[k] * g[k];
            Scalar second = v[k];
            if (_amsgrad) {
                Scalar &peak = _vmax[i].data()[k];
                peak = std::max(peak, v[k]);
                second = peak;
            }
            w[k] -= _rate * _weightDecay * w[k];
            w[k] -= _rate * (m[k] / correction1) / (std::sqrt(second / correction2) + _epsilon);
        }
    }
}

void Adam::setLearningRate(Scalar rate)
{
    requireNonNegative(_name, "learning rate", rate);
    _rate = rate;
}

Scalar Adam::learningRate() const
{
    return _rate;
}

std::string Adam::type() const
{
    return _name;
}

AdamW::AdamW(Scalar learningRate, Scalar weightDecay, Scalar beta1, Scalar beta2, Scalar epsilon,
             bool amsgrad)
    : Adam(learningRate, weightDecay, beta1, beta2, epsilon, amsgrad, "adamw")
{
}

RMSProp::RMSProp(Scalar learningRate, Scalar decay, Scalar epsilon)
    : _rate(learningRate), _decay(decay), _epsilon(epsilon)
{
    requirePositive("rmsprop", "learning rate", learningRate);
    requirePositive("rmsprop", "epsilon", epsilon);
    requireInUnit("rmsprop", "decay", decay);
}

void RMSProp::step(const std::vector<Parameter> &parameters)
{
    prepare("rmsprop", _square, parameters);
    for (std::size_t i = 0; i < parameters.size(); i++) {
        Scalar *w = parameters[i].value->data();
        const Scalar *g = parameters[i].grad->data();
        Scalar *s = _square[i].data();

        for (std::size_t k = 0; k < parameters[i].value->size(); k++) {
            s[k] = _decay * s[k] + (1 - _decay) * g[k] * g[k];
            w[k] -= _rate * g[k] / (std::sqrt(s[k]) + _epsilon);
        }
    }
}

void RMSProp::setLearningRate(Scalar rate)
{
    requireNonNegative("rmsprop", "learning rate", rate);
    _rate = rate;
}

Scalar RMSProp::learningRate() const
{
    return _rate;
}

std::string RMSProp::type() const
{
    return "rmsprop";
}

Adagrad::Adagrad(Scalar learningRate, Scalar epsilon) : _rate(learningRate), _epsilon(epsilon)
{
    requirePositive("adagrad", "learning rate", learningRate);
    requirePositive("adagrad", "epsilon", epsilon);
}

void Adagrad::step(const std::vector<Parameter> &parameters)
{
    prepare("adagrad", _sum, parameters);
    for (std::size_t i = 0; i < parameters.size(); i++) {
        Scalar *w = parameters[i].value->data();
        const Scalar *g = parameters[i].grad->data();
        Scalar *s = _sum[i].data();

        for (std::size_t k = 0; k < parameters[i].value->size(); k++) {
            s[k] += g[k] * g[k];
            w[k] -= _rate * g[k] / (std::sqrt(s[k]) + _epsilon);
        }
    }
}

void Adagrad::setLearningRate(Scalar rate)
{
    requireNonNegative("adagrad", "learning rate", rate);
    _rate = rate;
}

Scalar Adagrad::learningRate() const
{
    return _rate;
}

std::string Adagrad::type() const
{
    return "adagrad";
}

} // namespace my_torch
