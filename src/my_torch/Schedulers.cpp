#include "my_torch/Schedulers.hpp"

#include <cmath>
#include <numbers>
#include <utility>

#include "my_torch/Error.hpp"

namespace my_torch {

namespace {

void requirePositive(const char *scheduler, const char *name, Scalar value)
{
    if (!(value > 0))
        throw Error(std::string(scheduler) + ": " + name + " must be positive");
}

} // namespace

ConstantRate::ConstantRate(Scalar base) : _base(base)
{
    requirePositive("constant", "rate", base);
}

Scalar ConstantRate::rate(std::size_t /*epoch*/) const
{
    return _base;
}

std::string ConstantRate::type() const
{
    return "constant";
}

StepDecay::StepDecay(Scalar base, std::size_t stepSize, Scalar gamma)
    : _base(base), _stepSize(stepSize), _gamma(gamma)
{
    requirePositive("step", "rate", base);
    requirePositive("step", "gamma", gamma);
    if (stepSize == 0)
        throw Error("step: step size must be positive");
}

Scalar StepDecay::rate(std::size_t epoch) const
{
    return _base * std::pow(_gamma, static_cast<Scalar>(epoch / _stepSize));
}

std::string StepDecay::type() const
{
    return "step";
}

ExponentialDecay::ExponentialDecay(Scalar base, Scalar gamma) : _base(base), _gamma(gamma)
{
    requirePositive("exponential", "rate", base);
    requirePositive("exponential", "gamma", gamma);
}

Scalar ExponentialDecay::rate(std::size_t epoch) const
{
    return _base * std::pow(_gamma, static_cast<Scalar>(epoch));
}

std::string ExponentialDecay::type() const
{
    return "exponential";
}

CosineAnnealing::CosineAnnealing(Scalar base, std::size_t totalEpochs, Scalar minimum)
    : _base(base), _totalEpochs(totalEpochs), _minimum(minimum)
{
    requirePositive("cosine", "rate", base);
    if (totalEpochs == 0)
        throw Error("cosine: total epochs must be positive");
    if (!(minimum >= 0 && minimum <= base))
        throw Error("cosine: minimum must be in [0, rate]");
}

Scalar CosineAnnealing::rate(std::size_t epoch) const
{
    if (epoch >= _totalEpochs)
        return _minimum;
    const Scalar progress = static_cast<Scalar>(epoch) / static_cast<Scalar>(_totalEpochs);

    return _minimum + (_base - _minimum) * (1 + std::cos(std::numbers::pi * progress)) / 2;
}

std::string CosineAnnealing::type() const
{
    return "cosine";
}

Warmup::Warmup(std::unique_ptr<Scheduler> inner, std::size_t warmupEpochs)
    : _inner(std::move(inner)), _warmupEpochs(warmupEpochs)
{
    if (!_inner)
        throw Error("warmup: missing scheduler");
    if (warmupEpochs == 0)
        throw Error("warmup: warmup epochs must be positive");
}

Scalar Warmup::rate(std::size_t epoch) const
{
    if (epoch >= _warmupEpochs)
        return _inner->rate(epoch - _warmupEpochs);
    return _inner->rate(0) * static_cast<Scalar>(epoch + 1) / static_cast<Scalar>(_warmupEpochs);
}

std::string Warmup::type() const
{
    return "warmup_" + _inner->type();
}

} // namespace my_torch
