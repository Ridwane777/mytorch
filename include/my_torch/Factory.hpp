#pragma once

#include <memory>

#include "my_torch/Loss.hpp"
#include "my_torch/Network.hpp"
#include "my_torch/Optimizer.hpp"
#include "my_torch/Schedulers.hpp"

namespace my_torch {

// Build the training objects from their description (ADR-0012). Missing keys take the default of
// the matching constructor; an unknown type, an unknown key or an unreadable value throws
// my_torch::Error.
//
// Losses:     mse, mae, huber (delta), binary_cross_entropy,
//             cross_entropy (smoothing, weights=w1,w2,…)
// Optimizers: sgd (lr, momentum, nesterov=0|1), adam and adamw (lr, beta1, beta2, epsilon,
//             amsgrad=0|1; adamw also weight_decay), rmsprop (lr, decay, epsilon),
//             adagrad (lr, epsilon)
// Schedulers: none (no object), constant (base), step (base, step, gamma),
//             exponential (base, gamma), cosine (base, epochs, min); each accepts warmup=N
std::unique_ptr<Loss> makeLoss(const ComponentSpec &spec);
std::unique_ptr<Optimizer> makeOptimizer(const ComponentSpec &spec);
std::unique_ptr<Scheduler> makeScheduler(const ComponentSpec &spec);

} // namespace my_torch
