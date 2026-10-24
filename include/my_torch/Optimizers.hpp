#pragma once

#include <cstddef>
#include <vector>

#include "my_torch/Optimizer.hpp"

namespace my_torch {

// Optimizers: at each step they read every parameter's gradient and move its value.
// Those with a memory (velocity, running averages of gradients...) keep it in matrices shaped like
// the parameters, created on the first call and indexed by position: the same parameter vector, in
// the same order, must therefore be passed at every step, otherwise my_torch::Error is thrown.
// Gradients are not zeroed here: that is the training loop's job. Every constructor throws
// my_torch::Error on an out-of-range hyperparameter; setLearningRate accepts 0 (a scheduler may
// reach it) but not a negative rate.

// Gradient descent: w -= lr·v, with v = momentum·v + g.
// - momentum = 0: plain SGD, w -= lr·g.
// - momentum > 0: the velocity v accumulates past gradients, which speeds up travel along valleys
//   and damps oscillations.
// - nesterov: move with g + momentum·v instead of v, i.e. look slightly ahead.
class SGD : public Optimizer {
  public:
    explicit SGD(Scalar learningRate, Scalar momentum = 0, bool nesterov = false);

    void step(const std::vector<Parameter> &parameters) override;
    void setLearningRate(Scalar rate) override;
    Scalar learningRate() const override;
    std::string type() const override;

  private:
    Scalar _rate;
    Scalar _momentum;
    bool _nesterov;
    std::vector<Matrix> _velocity;
};

// Adam: a per-weight step size. m is the running average of gradients, v that of their squares
// (both corrected for the start-up bias), and w -= lr·m̂ / (√v̂ + ε). A weight with a steadily large
// gradient moves by about lr per step, whatever the gradient's scale.
// - amsgrad: v never decreases (maximum of the past v), so steps never grow back.
class Adam : public Optimizer {
  public:
    explicit Adam(Scalar learningRate = 0.001, Scalar beta1 = 0.9, Scalar beta2 = 0.999,
                  Scalar epsilon = 1e-8, bool amsgrad = false);

    void step(const std::vector<Parameter> &parameters) override;
    void setLearningRate(Scalar rate) override;
    Scalar learningRate() const override;
    std::string type() const override;

  protected:
    Adam(Scalar learningRate, Scalar weightDecay, Scalar beta1, Scalar beta2, Scalar epsilon,
         bool amsgrad, const char *name);

  private:
    Scalar _rate;
    Scalar _weightDecay;
    Scalar _beta1;
    Scalar _beta2;
    Scalar _epsilon;
    bool _amsgrad;
    const char *_name;
    std::size_t _steps = 0;
    std::vector<Matrix> _m;
    std::vector<Matrix> _v;
    std::vector<Matrix> _vmax;
};

// Adam with decoupled weight decay: w -= lr·weightDecay·w, independently of the gradient.
// An L2 penalty added to Adam's gradient would be rescaled like the gradient and lose its effect.
class AdamW : public Adam {
  public:
    explicit AdamW(Scalar learningRate = 0.001, Scalar weightDecay = 0.01, Scalar beta1 = 0.9,
                   Scalar beta2 = 0.999, Scalar epsilon = 1e-8, bool amsgrad = false);
};

// RMSProp: divides the gradient by the root of a running average of its squares, which calms
// weights with violent gradients and speeds up those with weak ones.
class RMSProp : public Optimizer {
  public:
    explicit RMSProp(Scalar learningRate = 0.001, Scalar decay = 0.9, Scalar epsilon = 1e-8);

    void step(const std::vector<Parameter> &parameters) override;
    void setLearningRate(Scalar rate) override;
    Scalar learningRate() const override;
    std::string type() const override;

  private:
    Scalar _rate;
    Scalar _decay;
    Scalar _epsilon;
    std::vector<Matrix> _square;
};

// Adagrad: divides by the root of the sum of all past squares. Each weight's step only shrinks,
// which suits convex or sparse problems but eventually freezes the learning of a deep network.
class Adagrad : public Optimizer {
  public:
    explicit Adagrad(Scalar learningRate = 0.01, Scalar epsilon = 1e-10);

    void step(const std::vector<Parameter> &parameters) override;
    void setLearningRate(Scalar rate) override;
    Scalar learningRate() const override;
    std::string type() const override;

  private:
    Scalar _rate;
    Scalar _epsilon;
    std::vector<Matrix> _sum;
};

} // namespace my_torch
