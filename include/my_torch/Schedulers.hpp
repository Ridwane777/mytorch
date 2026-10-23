#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "my_torch/Matrix.hpp"

namespace my_torch {

// Learning-rate schedulers: a pure function of the epoch (the first one is 0). The training loop
// applies it with optimizer.setLearningRate(scheduler.rate(epoch)) at the start of each epoch. A
// scheduler may return 0 (cosine after its last epoch, or an exponential decay that underflows),
// which every optimizer accepts. Being stateless, a scheduler gives the same value for the same
// epoch, so a training run can be resumed (#35) without saving anything. A high rate early on
// moves fast; lowering it later avoids oscillating around the minimum.
class Scheduler {
  public:
    virtual ~Scheduler() = default;

    virtual Scalar rate(std::size_t epoch) const = 0;
    virtual std::string type() const = 0;
};

// Always the same rate.
class ConstantRate : public Scheduler {
  public:
    explicit ConstantRate(Scalar base);

    Scalar rate(std::size_t epoch) const override;
    std::string type() const override;

  private:
    Scalar _base;
};

// Multiplies the rate by gamma every stepSize epochs: base·gamma^⌊epoch / stepSize⌋.
class StepDecay : public Scheduler {
  public:
    StepDecay(Scalar base, std::size_t stepSize, Scalar gamma = 0.1);

    Scalar rate(std::size_t epoch) const override;
    std::string type() const override;

  private:
    Scalar _base;
    std::size_t _stepSize;
    Scalar _gamma;
};

// Continuous decay: base·gamma^epoch.
class ExponentialDecay : public Scheduler {
  public:
    ExponentialDecay(Scalar base, Scalar gamma);

    Scalar rate(std::size_t epoch) const override;
    std::string type() const override;

  private:
    Scalar _base;
    Scalar _gamma;
};

// Goes from base down to minimum along a half cosine over totalEpochs epochs:
// minimum + (base - minimum)·(1 + cos(π·epoch / totalEpochs)) / 2, then stays at minimum.
// Slow at the start and at the end, fast in the middle.
class CosineAnnealing : public Scheduler {
  public:
    CosineAnnealing(Scalar base, std::size_t totalEpochs, Scalar minimum = 0);

    Scalar rate(std::size_t epoch) const override;
    std::string type() const override;

  private:
    Scalar _base;
    std::size_t _totalEpochs;
    Scalar _minimum;
};

// Warm-up: climbs linearly to the starting rate of `inner` over warmupEpochs epochs
// (inner.rate(0)·(epoch + 1) / warmupEpochs), then lets `inner` take over from its own epoch 0.
// The last warm-up epoch and the first handed-over epoch therefore both use inner.rate(0).
// Avoids Adam's overly large first steps, while its averages are still empty.
class Warmup : public Scheduler {
  public:
    Warmup(std::unique_ptr<Scheduler> inner, std::size_t warmupEpochs);

    Scalar rate(std::size_t epoch) const override;
    std::string type() const override;

  private:
    std::unique_ptr<Scheduler> _inner;
    std::size_t _warmupEpochs;
};

} // namespace my_torch
